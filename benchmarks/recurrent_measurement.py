"""Measured decode windows for native completion streams; no model-specific algebra."""
import ctypes
import ctypes.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent / "server-ab"))
from server_ab import SSEAccumulator, iter_sse_payloads  # noqa: E402


def decode_window(rows):
    if not rows:
        raise ValueError("no completed streams")
    streams = []
    for row in rows:
        times = row.get("token_times", [])
        if len(times) < 2 or len(times) != len(row.get("tokens", [])) or any(not math.isfinite(t) for t in times):
            raise ValueError("missing or incomplete token timeline")
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("token timeline is not monotonic")
        streams.append(times)
    start = max(times[0] for times in streams)
    end = min(times[-1] for times in streams)
    if end <= start:
        raise ValueError("streams have no common decode interval")
    counts = [sum(start < t <= end for t in times) for times in streams]
    if not all(counts):
        raise ValueError("a stream emitted no tokens during the common interval")
    return {"start": start, "end": end, "seconds": end - start, "tokens": counts,
            "slots": len(rows), "tokens_per_second": sum(counts) / (end - start)}


def stream_completions(url, payload, count, timeout):
    """A single vector-prompt request admits the cohort together; index identifies each stream."""
    body = dict(payload, stream=True, return_tokens=True)
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                    headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.monotonic()
    accumulators = [SSEAccumulator(started) for _ in range(count)]
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for event in iter_sse_payloads(line.decode("utf-8") for line in response):
            index = event.get("index", 0 if count == 1 else None)
            if type(index) is not int or not 0 <= index < count:
                raise RuntimeError("completion stream has an invalid cohort index")
            accumulators[index].feed(event, time.monotonic())
    finished = time.monotonic()
    rows = []
    for accumulator in accumulators:
        row = accumulator.finish(finished)
        row["id_slot"] = accumulator.final.get("id_slot")
        row["token_times"] = accumulator.token_times
        row["request_started_at"] = started
        rows.append(row)
    for row in rows:
        decode_window([row])
    try:
        window = decode_window(rows)
    except ValueError:
        # The fitter also validates intentionally serialized cohorts. They have
        # no parallel score; its independent selection proof still checks routing.
        window = None
    for row in rows:
        row["common_decode_window"] = window
    return rows


class MemorySampler:
    """Optional whole-device NVML samples. This is a sampled peak, including display/driver use."""
    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, device=0, period=0.02):
        self.device = device
        self.period = period
        self.samples = []
        self.error = None
        self.library = None
        self.handle = ctypes.c_void_p()
        self.stopped = threading.Event()
        self.thread = None

    def __enter__(self):
        try:
            name = "nvml.dll" if os.name == "nt" else ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            self.library = ctypes.CDLL(name)
            if self.library.nvmlInit_v2() != 0:
                raise RuntimeError("NVML initialization failed")
            self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            self.library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Memory)]
            if self.library.nvmlDeviceGetHandleByIndex_v2(self.device, ctypes.byref(self.handle)) != 0:
                raise RuntimeError("NVML device lookup failed")
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except (OSError, RuntimeError, AttributeError) as error:
            self.error = str(error)
        return self

    def _run(self):
        while True:
            value = self.Memory()
            if self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(value)) != 0:
                self.error = "NVML memory query failed"
                return
            self.samples.append({"time": time.monotonic(), "used": value.used, "total": value.total})
            if self.stopped.wait(self.period):
                return

    def __exit__(self, *_):
        self.stopped.set()
        if self.thread:
            self.thread.join()
        if self.library and hasattr(self.library,"nvmlShutdown"):
            self.library.nvmlShutdown()

    def report(self):
        return {"kind": "whole-device-sampled", "period_seconds": self.period, "error": self.error,
                "peak_bytes": max((row["used"] for row in self.samples), default=None), "samples": self.samples}
