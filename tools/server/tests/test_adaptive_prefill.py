#!/usr/bin/env python3
"""Offline two-slot prefill qualification with progress-gated arrivals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request

TEST_SOURCE = Path(__file__).read_bytes()


def request(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.load(response)


def instrumented_server_pid(parent, executable):
    pending = [parent]
    while pending:
        pid = pending.pop()
        try:
            if Path(f"/proc/{pid}/exe").resolve() == executable:
                return pid
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            pending.extend(int(child) for child in children)
        except OSError:
            pass
    return None


def qualify(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness-source.py").write_bytes(TEST_SOURCE)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [str(args.server.resolve()), "--model", str(args.model.resolve()), "--parallel", str(args.parallel),
               "--ctx-size", str(args.context_size), "--batch-size", "256", "--ubatch-size", "256", "--n-gpu-layers", "999",
               "--flash-attn", "on", "--cache-type-k", args.cache_type_k, "--cache-type-v", args.cache_type_v,
               "--shared-device-memory-mib", str(args.arena_mib), "--fit", "off", "--no-mmproj", "--offline",
               "--no-context-shift", "--no-warmup", "--cache-ram", "0", "--no-webui", "--threads", "4",
               "--threads-batch", "4", "--host", "127.0.0.1", "--port", str(port), "-lv", "3"]
    if args.context_checkpoints is not None:
        command += ["--ctx-checkpoints", str(args.context_checkpoints)]
    if args.parallel == 2:
        command += ["--kv-unified"]
    if args.mmproj:
        command.remove("--no-mmproj")
        command += ["--mmproj", str(args.mmproj.resolve()), "--image-min-tokens", "64", "--image-max-tokens", "256"]
    if args.mode == "target":
        command += ["--spec-type", "none"]
    else:
        command += ["--spec-type", "draft-mtp" if args.mode == "mtp" else "draft-dflash,ngram-map-k4v" if args.mode == "dflash7k4v" else "draft-dflash",
                    "--spec-draft-n-max", "3" if args.mode == "mtp" else "7", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"]
        if args.model_draft:
            command += ["--model-draft", str(args.model_draft.resolve()), "--spec-draft-ngl", "999"]
        if args.mode == "dflash7k4v":
            command += ["--spec-ngram-map-k4v-size-n", "12", "--spec-ngram-map-k4v-size-m", "7",
                        "--spec-ngram-map-k4v-min-hits", "2"]
    if args.sanitizer:
        command = [str(args.sanitizer), "--tool", "memcheck", "--report-api-errors", "all", "--error-exitcode", "99"] + command
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.server.resolve().parent) + ":/usr/local/cuda/lib64",
               CUDA_MODULE_LOADING="LAZY", LLAMA_MEDIA_MARKER="<__media__>")
    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    env.pop("LD_PRELOAD", None)
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    report = {"passed": False, "mode": args.mode, "parallel": args.parallel, "arrival_phase": args.arrival_phase,
              "harness_sha256": hashlib.sha256(TEST_SOURCE).hexdigest(), "closed_group_reference": args.closed_group_reference}
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup")
                try:
                    if request(port, "/health").get("status") == "ok":
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("server startup timed out")
            if args.mmproj and not args.late_vision:
                from test_adaptive_vision import png
                image = request(port, "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png(256, 256)}}]}],
                    "temperature": 0, "max_tokens": 32, "ignore_eos": True})
                (args.output / "image-response.json").write_text(json.dumps(image, indent=2) + "\n")
                assert image["usage"]["completion_tokens"] == 32
                if args.mode != "target":
                    assert image["timings"]["draft_n"] > 0
                report["image_prefill_before_text"] = True
            prompts = [request(port, "/tokenize", {"content": text, "add_special": True})["tokens"]
                       for text in ("Explain this source code.\n" + "print('hello')\n" * 1800,
                                    "Explain this different source code.\n" + "console.log('world');\n" * 300)]
            prompts[0] = prompts[0][:args.prompt_tokens_a]
            prompts[1] = prompts[1][:args.prompt_tokens_b]
            for index, source in enumerate((args.tokens_a, args.tokens_b)):
                if source:
                    values = json.loads(source.read_text())
                    assert isinstance(values, list) and all(isinstance(value, int) and value >= 0 for value in values)
                    count = args.prompt_tokens_a if index == 0 else args.prompt_tokens_b
                    prompts[index] = values[-count:]
            late_image = None
            if args.late_vision:
                assert args.mmproj and not args.cached_prefix and args.arrival_phase == "prefill"
                from test_adaptive_vision import png
                late_image = {"prompt_string": "Describe: <__media__> Answer:", "multimodal_data": [png(256, 256)]}
            report["prompt_sha256_each"] = [hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
                for prompt in (prompts[0], late_image if late_image else prompts[1])]
            ready = threading.Event()
            joined = threading.Event()
            traces = [{"tokens": [], "progress": []}, {"tokens": [], "progress": []}]
            case_started = time.monotonic()
            slot_ids = ([1, 0] if args.swap_slots else [0, 1]) if args.parallel == 2 else [0, 0]
            if args.cached_prefix:
                probe = request(port, "/completion", {"prompt": prompts[1][:256], "n_predict": 1, "ignore_eos": True,
                    "id_slot": slot_ids[1], "temperature": 0, "seed": 1234, "cache_prompt": False})
                assert probe["tokens_predicted"] == 1
            if args.arrival_phase == "simultaneous":
                ready.set()

            def stream(index):
                traces[index]["started"] = time.monotonic() - case_started
                body = {"prompt": late_image if index == 1 and late_image else prompts[index], "n_predict": args.outputs, "temperature": 0, "seed": 1234,
                        "ignore_eos": True, "cache_prompt": args.cached_prefix, "stream": True, "return_tokens": True,
                        "return_progress": True, "id_slot": slot_ids[index]}
                if args.check_finite_probs:
                    body["n_probs"] = 16
                req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                events = []
                generated = 0
                with urllib.request.urlopen(req, timeout=args.timeout_seconds) as response:
                    for line in response:
                        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                            continue
                        value = json.loads(line[6:])
                        if not isinstance(value, dict):
                            continue
                        if "error" in value:
                            raise RuntimeError(str(value["error"]))
                        events.append(value)
                        elapsed = time.monotonic() - case_started
                        progress = value.get("prompt_progress")
                        if progress:
                            traces[index]["progress"].append([elapsed, progress["processed"]])
                        elif value.get("tokens") and value.get("tokens_predicted", 0) > 0:
                            traces[index]["tokens"].append([elapsed, value["tokens_predicted"], value["tokens"]])
                        if index == 1 and progress and progress["processed"] > 0:
                            joined.set()
                        generated = max(generated, value.get("tokens_predicted", 0))
                        if index == 0 and args.arrival_phase == "prefill" and progress and 0 < progress["processed"] < progress["total"]:
                            ready.set()
                        if index == 0 and args.arrival_phase == "decode" and generated >= 2:
                            ready.set()
                        if args.cancel_request == index and joined.is_set():
                            break
                traces[index]["finished"] = time.monotonic() - case_started
                return events

            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(stream, 0)
                assert ready.wait(timeout=120), "no unfinished-prefill progress event"
                delay = args.arrival_delay_seconds - (time.monotonic() - case_started)
                if delay > 0:
                    time.sleep(delay)
                second = pool.submit(stream, 1)
                outputs = [first.result(timeout=args.timeout_seconds), second.result(timeout=args.timeout_seconds)]
            (args.output / "client-traces.json").write_text(json.dumps(traces, indent=2) + "\n")
            for index, events in enumerate(outputs):
                (args.output / f"slot{index}.json").write_text(json.dumps(events, indent=2) + "\n")
                if args.cancel_request == index:
                    continue
                final = next(value for value in reversed(events) if value.get("stop"))
                assert final["tokens_predicted"] == args.outputs and final["id_slot"] == slot_ids[index]
                if args.check_finite_probs:
                    probabilities = [entry for value in events for entry in value.get("completion_probabilities", [])
                                     if entry.get("top_logprobs")]
                    assert probabilities, "missing pre-sampling probabilities"
                    assert all(item["logprob"] is not None and math.isfinite(item["logprob"])
                        for entry in probabilities for item in entry["top_logprobs"]), "non-finite model probabilities"
                    report["finite_probabilities"] = True
                if args.mode != "target":
                    assert final["timings"]["draft_n"] > 0
                if args.cached_prefix and index == 1:
                    assert final["timings"]["cache_n"] == 256
                    report["cached_prefix_tokens"] = 256
            text = (args.output / "server.log").read_text()
            prefill = re.findall(r"stream group bind: selected=(\d+) rows=(\d+) phase=prefill", text)
            decode = re.findall(r"stream group bind: selected=(\d+) rows=(\d+) phase=decode", text)
            report.update(prompt_tokens_each=[len(value) for value in prompts], prefill_bindings=prefill, decode_bindings=decode)
            expect_pair = args.parallel == 2 and args.arrival_phase != "decode" and not args.closed_group_reference and not args.late_vision
            assert any(int(selected) == 2 for selected, _ in prefill) == (expect_pair and not args.expect_prefill_fallback), "unexpected shared-prefill scheduling"
            if args.expect_prefill_fallback:
                assert expect_pair and "stream selection capacity: selected=2" in text
                report["prefill_capacity_fallback"] = True
            if args.late_vision:
                stopped = re.search(r"id\s+" + str(slot_ids[0]) + r"\s+\|[^\n]*stop processing", text)
                first_decode = text.index("phase=decode")
                next_prefill = text.index("phase=prefill", first_decode)
                assert stopped and first_decode < stopped.start() < next_prefill
                final = next(value for value in reversed(outputs[1]) if value.get("stop"))
                report["prompt_tokens_each"] = [len(prompts[0]), final["timings"]["prompt_n"] + final["timings"].get("cache_n", 0)]
                report["late_vision_queued"] = True
            if args.cancel_request is None:
                assert any(int(selected) == 2 for selected, _ in decode) == expect_pair, "unexpected joint-decode scheduling"
            else:
                retried = request(port, "/completion", {"prompt": "Write a short greeting.", "id_slot": slot_ids[args.cancel_request],
                                  "n_predict": 16, "ignore_eos": True, "temperature": 0, "return_tokens": True, "cache_prompt": False})
                assert retried["tokens_predicted"] == 16
                report["cancelled_request_retry"] = True
            assert request(port, "/health")["status"] == "ok"
            report["passed"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            if process.poll() is None:
                if args.sanitizer:
                    child = instrumented_server_pid(process.pid, args.server.resolve())
                    if child is not None:
                        os.kill(child, signal.SIGTERM)
                    else:
                        process.send_signal(signal.SIGINT)
                else:
                    process.terminate()
            try:
                process.wait(timeout=45 if args.sanitizer else 15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            if args.sanitizer and report["passed"]:
                text = (args.output / "server.log").read_text()
                summaries = [int(count) for count in re.findall(r"ERROR SUMMARY:\s+(\d+) errors", text)]
                report["sanitizer_zero_errors"] = bool(summaries) and not any(summaries) and process.returncode == 0
                report["sanitizer_returncode"] = process.returncode
                if not report["sanitizer_zero_errors"]:
                    report.update(passed=False, error="missing zero-error CUDA memcheck summary")
            (args.output / "status.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"], report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-draft", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--late-vision", action="store_true")
    parser.add_argument("--mode", choices=["target", "mtp", "dflash7", "dflash7k4v"], default="target")
    parser.add_argument("--arena-mib", type=int, default=1024)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--parallel", type=int, choices=[1, 2], default=2)
    parser.add_argument("--arrival-phase", choices=["prefill", "decode", "simultaneous"], default="prefill")
    parser.add_argument("--swap-slots", action="store_true")
    parser.add_argument("--cancel-request", type=int, choices=[0, 1])
    parser.add_argument("--prompt-tokens-a", type=int, default=8192)
    parser.add_argument("--prompt-tokens-b", type=int, default=2048)
    parser.add_argument("--outputs", type=int, default=64)
    parser.add_argument("--tokens-a", type=Path)
    parser.add_argument("--tokens-b", type=Path)
    parser.add_argument("--arrival-delay-seconds", type=float, default=0)
    parser.add_argument("--closed-group-reference", action="store_true")
    parser.add_argument("--expect-prefill-fallback", action="store_true")
    parser.add_argument("--check-finite-probs", action="store_true")
    parser.add_argument("--cached-prefix", action="store_true")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--context-checkpoints", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--sanitizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    qualify(parser.parse_args())
