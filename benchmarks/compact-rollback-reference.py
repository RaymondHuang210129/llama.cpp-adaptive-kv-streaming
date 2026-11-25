#!/usr/bin/env python3
"""Freeze and compare dense rollback references using the arena finder's real probes.

Usage: compact-rollback-reference.py OUTPUT ARENA_MIB --server ... --model ...
The remaining arguments are find-max-arena.py arguments. Both placements use
identical flags, prompts and arena size. This is a same-build reference check,
not an independent stock-server numerical equivalence test.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(native, ram):
    if not native["fits"] or not ram["fits"]:
        raise ValueError("both reference executions must finish")
    if not native.get("prompt_ids_sha256") or native["prompt_ids_sha256"] != ram.get("prompt_ids_sha256"):
        raise ValueError("reference prompt token histories differ")
    if not native.get("profiles") or len(native["profiles"]) != len(ram.get("profiles", [])):
        raise ValueError("reference profile coverage differs or is empty")
    count = 0
    for left, right in zip(native["profiles"], ram["profiles"]):
        if left["label"] != right["label"] or len(left["responses"]) != len(right["responses"]):
            raise ValueError("reference selected-request coverage differs")
        for a, b in zip(left["responses"], right["responses"]):
            if not a.get("tokens") or a["tokens"] != b.get("tokens"):
                raise ValueError("reference output tokens differ or are empty")
            for response in (a, b):
                if response["timings"]["predicted_n"] != len(response["tokens"]):
                    raise ValueError("reference response is incomplete")
            count += len(a["tokens"])
    return count


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    directory, arena = Path(argv[0]).resolve(), int(argv[1])
    directory.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("rollback_finder", Path(__file__).with_name("find-max-arena.py"))
    finder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finder)
    if "--recurrent-rollback-ram" in argv[2:]:
        raise ValueError("this harness selects both placements itself")
    runs = []
    for ram in (False, True):
        args = finder.arguments(argv[2:] + (["--recurrent-rollback-ram"] if ram else []))
        args.output = directory / ("ram" if ram else "native")
        args.output.mkdir()
        record = finder.probe(args, arena)
        profiles = []
        for artifact in sorted(args.output.glob(f"arena-{arena}-*-response.json")):
            value = json.loads(artifact.read_text())
            if "responses" in value:
                profiles.append({"label": artifact.stem, "responses": value["responses"]})
        if not profiles and record.get("response"):
            profiles = [{"label": "serial", "responses": [json.loads(Path(record["response"]).read_text())]}]
        record["profiles"] = profiles
        (args.output / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        runs.append(record)
    files = [args.server, args.model, args.prefill_text] + ([args.model_draft] if args.model_draft else [])
    manifest = {"schema": 1, "reference_kind": "same-build-native-versus-RAM", "arena_mib": arena,
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], text=True).strip(),
                "files": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in files],
                "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
                "matching_tokens": compare(*runs), "runs": runs}
    (directory / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(directory), "matching_tokens": manifest["matching_tokens"]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
