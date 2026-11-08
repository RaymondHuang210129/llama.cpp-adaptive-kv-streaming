"""Profile one fresh CUDA object without changing its optimization or architecture flags."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from fork_ci import digest, read_configuration


SOURCES = ("kv-stream-mma-runtime-1-8.cu", "kv-stream-mma-runtime-4-2.cu")


def select_object(build, source):
    entries = json.loads(subprocess.check_output(["ninja", "-t", "compdb", "-x"], cwd=build, text=True))
    selected = [entry for entry in entries if Path(entry["file"]).name == source]
    if source not in SOURCES or len(selected) != 1:
        raise ValueError("expected exactly one supported CUDA source")
    entry = selected[0]
    output = Path(entry["output"])
    if not output.is_absolute():
        output = build / output
    output = output.resolve()
    if not output.is_relative_to(build.resolve()) or output.suffix not in (".o", ".obj"):
        raise ValueError("invalid compiler object output")
    if output.exists():
        raise ValueError("refusing to benchmark an existing object")
    return entry, output


def diagnostic_environment(output):
    environment = dict(os.environ)
    if any(environment.get(name) for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS")):
        raise ValueError("unexpected NVCC environment flags")
    # NVCC parses this variable as command-line flags on both supported platforms.
    environment["NVCC_APPEND_FLAGS"] = subprocess.list2cmdline([
        "--time", str(output / "phases.csv"), "--verbose"])
    return environment


def linux_snapshot(pid):
    queue, seen, processes = [pid], set(), []
    ticks = os.sysconf("SC_CLK_TCK")
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        root = Path("/proc") / str(current)
        try:
            stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            fields = dict(line.split(":", 1) for line in root.joinpath("status").read_text().splitlines())
            queue.extend(map(int, root.joinpath("task", str(current), "children").read_text().split()))
            processes.append({"pid": current, "name": root.joinpath("comm").read_text().strip(),
                              "cpu_seconds": (int(stat[11]) + int(stat[12])) / ticks,
                              "working_set_bytes": int(fields.get("VmRSS", "0").split()[0]) * 1024,
                              "swap_bytes": int(fields.get("VmSwap", "0").split()[0]) * 1024,
                              "major_faults": int(stat[9])})
        except (FileNotFoundError, ProcessLookupError):
            continue
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {"processes": processes,
            "system_memory_kib": {key: int(memory[key].split()[0])
                                  for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")},
            "system_vmstat": {key: int(vmstat[key]) for key in ("pswpin", "pswpout", "pgmajfault")}}


def snapshot(pid):
    if sys.platform == "win32":
        command = ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-File", str(Path(__file__).with_name("cuda_compile_memory.ps1")), "-RootPid", str(pid)]
        return json.loads(subprocess.check_output(command, text=True, encoding="utf-8", timeout=20))
    return linux_snapshot(pid)


def stop(process):
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, timeout=20)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=20)


def valid_telemetry(record):
    return (not record.get("counter_errors") and not record.get("telemetry_error")
            and (sys.platform != "win32" or bool(record.get("counters")))
            and all(counter.get("Status") in (0, 1) for counter in record.get("counters", [])))


def profile(build, source, output, timeout_seconds):
    entry, binary = select_object(build, source)
    output.mkdir(parents=True, exist_ok=False)
    environment = diagnostic_environment(output)
    metadata = {"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "configuration": read_configuration(build), "compiler_entry": entry,
                "source_sha256": digest(Path(entry["file"])), "helper_sha256": digest(Path(__file__)),
                "platform": sys.platform, "logical_cpus": os.cpu_count(),
                "instrumentation": {"NVCC_APPEND_FLAGS": environment["NVCC_APPEND_FLAGS"]},
                "scope": "one fresh object; no GPU execution or final package; sampled timing is instrumented"}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {"source": source, "compile_completed": False, "passed": False}
    try:
        preflight = snapshot(os.getpid())
        (output / "telemetry-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        if not valid_telemetry(preflight):
            raise RuntimeError("resource telemetry preflight failed; no compiler started")
    except BaseException as error:
        result["error"] = repr(error)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    command = ["cmake", "--build", str(build), "--parallel", "1", "--target", entry["output"]]
    start = time.monotonic()
    errors = 0
    with (output / "compiler.log").open("w") as log, (output / "resources.jsonl").open("w") as resources:
        process = subprocess.Popen(command, cwd=build, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=sys.platform != "win32")
        try:
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= timeout_seconds:
                    raise TimeoutError("diagnostic deadline exceeded; incomplete compilation is not a pass")
                record: dict[str, Any]
                try:
                    record = snapshot(process.pid)
                    if not valid_telemetry(record):
                        errors += 1
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    errors += 1
                    record = dict(telemetry_error=repr(error))
                record["elapsed_seconds"] = round(time.monotonic() - start, 3)
                resources.write(json.dumps(record) + "\n")
                resources.flush()
                sys.stdout.write(json.dumps(record) + "\n")
                sys.stdout.flush()
                try:
                    code = process.wait(timeout=min(30, max(0.01, timeout_seconds - (time.monotonic() - start))))
                    break
                except subprocess.TimeoutExpired:
                    continue
            result.update(exit_code=code, compile_completed=code == 0)
            if code:
                raise RuntimeError("CUDA compilation failed: " + str(code))
            if errors:
                raise RuntimeError("compilation succeeded but resource sampling was incomplete")
            result.update(object_bytes=binary.stat().st_size, object_sha256=digest(binary), passed=True)
        except BaseException as error:
            result["error"] = repr(error)
            raise
        finally:
            try:
                stop(process)
            except BaseException as error:
                result.update(passed=False, cleanup_error=repr(error))
                raise
            finally:
                result.update(elapsed_seconds=time.monotonic() - start, telemetry_errors=errors)
                (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--output", type=Path, default=Path("cuda-compile-diagnostics"))
    parser.add_argument("--timeout-minutes", type=int, default=330)
    args = parser.parse_args()
    if not 1 <= args.timeout_minutes <= 330:
        parser.error("diagnostic deadline must leave time for artifact upload within the six-hour job")

    def interrupted(signum, frame):
        raise KeyboardInterrupt("diagnostic interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    profile(args.build.resolve(), args.source, args.output.resolve(), args.timeout_minutes * 60)


if __name__ == "__main__":
    main()
