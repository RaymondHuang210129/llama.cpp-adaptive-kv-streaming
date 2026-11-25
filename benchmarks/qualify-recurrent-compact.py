#!/usr/bin/env python3
"""Resume bounded native/RAM versus compact correctness probes from a pinned manifest.

This is a product configuration gate, not an independent stock-model oracle or a
throughput sweep. Operator and forced-prefix tests supply the numerical contract.
The caller owns GPU isolation. A failed case stops the run; it is never retried or
silently counted as qualified. Completed cases are reusable only with the same
runtime-library hashes, manifest and workload.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cases(manifest):
    result, names = [], set()
    for model in manifest["models"]:
        name = model["name"]
        if name in names or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("model labels must be unique filesystem-safe names")
        names.add(name)
        for mode, depths in model["modes"].items():
            allowed = {"none": range(1), "mtp": range(1, 4), "dflash": range(1, 8), "dflash-k4v": range(1, 8)}
            if mode not in allowed or not depths or len(set(depths)) != len(depths):
                raise ValueError("invalid or duplicate mode/depth")
            for depth in depths:
                if type(depth) is not int or depth not in allowed[mode]:
                    raise ValueError("unsupported speculative depth")
                for kv, slots, vision in itertools.product(
                        [("q8_0", "q8_0"), ("q8_0", "q4_0"), ("q4_0", "q4_0")],
                        range(1, 5), [False, True] if model.get("mmproj") else [False]):
                    key = f"{name}-{mode}{depth}-{kv[0]}-{kv[1]}-n{slots}-{'image' if vision else 'text'}"
                    result.append(dict(key=key, model=model, mode=mode, depth=depth, kv=kv,
                                       slots=slots, vision=vision))
    if not result:
        raise ValueError("empty qualification manifest")
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_identity(server):
    paths = {server.resolve(), *server.parent.glob("*.so*"), *server.parent.glob("*.dll")}
    return {str(p.resolve()): digest(p) for p in sorted(paths) if p.is_file()}


def responses(record, directory):
    if record.get("selection_profiles"):
        return {row["profile"]["label"]: sorted(row["responses"], key=lambda x: x["id_slot"])
                for row in record["selection_profiles"]}
    image = directory / f"arena-{record['arena_mib']}-image-group-responses.json"
    return {"main": sorted(json.loads(image.read_text()), key=lambda x: x["id_slot"])
            if image.exists() else [json.loads(Path(record["response"]).read_text())]}


def compare(control, compact):
    if control.keys() != compact.keys():
        raise RuntimeError("mismatched request profiles")
    tokens = 0
    for label, rows in control.items():
        other = compact[label]
        if len(rows) != len(other):
            raise RuntimeError(f"{label}: mismatched request count")
        for slot, (left, right) in enumerate(zip(rows, other)):
            if not left.get("tokens") or left["tokens"] != right.get("tokens"):
                raise RuntimeError(f"{label} request {slot}: token mismatch")
            for key in ("prompt_n", "predicted_n"):
                if left["timings"][key] != right["timings"][key]:
                    raise RuntimeError(f"{label} request {slot}: {key} mismatch")
            tokens += len(left["tokens"])
    return tokens


def run_case(finder, config, case, server, root, identity, image, port):
    model = case["model"]
    folder = root / case["key"]
    folder.mkdir(exist_ok=True)
    expected = {"runtime": identity, "case": case, "decode_tokens": 16, "prompt_tokens": 64,
                "context_size": 8192, "fixture_sha256": digest(image)}
    marker = folder / "complete.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["identity"] != json.loads(json.dumps(expected)):
            raise RuntimeError(f"{case['key']}: completed record belongs to different inputs")
        return prior, False
    arena_key = ("image" if case["vision"] else "text") + ("_serial" if case["slots"] == 1 else "_group")
    arena = model["arena_mib"][arena_key]
    outputs, records = [], []
    for placement in ("control", "compact"):
        directory = folder / placement
        directory.mkdir(exist_ok=True)
        record_path = directory / "record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
        else:
            if list(directory.iterdir()):
                raise RuntimeError(f"{directory}: incomplete attempt; inspect before resuming")
            command = ["--server", str(server), "--model", model["model"], "--context-size", "8192",
                       "--parallel", str(case["slots"]), "--cache-type-k", case["kv"][0],
                       "--cache-type-v", case["kv"][1], "--spec-mode", case["mode"],
                       "--draft-length", str(case["depth"]), "--decode-tokens", "16",
                       "--batch-size", "256", "--ubatch-size", "256", "--probe-mode", "preflight",
                       "--no-webui", "--no-warmup", "--verbosity", "4", "--port", str(port)]
            draft = model.get("draft") if case["mode"].startswith("dflash") else model.get("assistant")
            if draft and case["mode"] != "none":
                command += ["--model-draft", draft]
            if case["mode"]=="dflash-k4v" and "ngram_m" in model:
                command += ["--ngram-m",str(model["ngram_m"])]
            if case["vision"]:
                command += ["--mmproj", model["mmproj"], "--image", str(image)]
                if model.get("image_tokens"):
                    command += ["--image-min-tokens", str(model["image_tokens"]),
                                "--image-max-tokens", str(model["image_tokens"])]
            else:
                command += ["--prompt-tokens", "64", "--distinct-probe-histories"]
            if placement == "compact":
                command += ["--recurrent-rollback-compact"]
            elif model.get("rollback_control") == "ram" and case["mode"] != "none":
                command += ["--recurrent-rollback-ram"]
            args = finder.arguments(command)
            args.output = directory
            from recurrent_measurement import MemorySampler
            with MemorySampler() as sampler:
                record = finder.probe(args, arena)
            record["memory"] = sampler.report()
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        if not record["fits"]:
            raise RuntimeError(f"{case['key']} {placement}: {record.get('error')}")
        required = placement == "compact" and model.get("compact_expected", False) and case["mode"] != "none"
        if record["observed_compact_recurrent"] != required:
            raise RuntimeError(f"{case['key']} {placement}: unexpected representation")
        text = Path(record["log"]).read_text(errors="replace")
        selected = sorted(set(map(int, re.findall(r"stream group bind: selected=(\d+) rows=\d+ phase=decode", text))))
        if case["slots"] > 1 and max(selected, default=0) != case["slots"]:
            raise RuntimeError(f"{case['key']} {placement}: expected selection not exercised: {selected}")
        outputs.append(responses(record, directory))
        records.append({"path": str(record_path), "sha256": digest(record_path),
                        "peak_bytes": record["memory"]["peak_bytes"], "selected": selected,
                        "compact_observed": record["observed_compact_recurrent"]})
    result = {"identity": expected, "matched_tokens": compare(*outputs), "records": records}
    marker.write_text(json.dumps(result, indent=2) + "\n")
    return result, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=1350)
    parser.add_argument("--limit", type=int, help="maximum new cases in this isolated block")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    matrix = cases(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("arena_finder", ROOT / "find-max-arena.py")
    finder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finder)
    image = args.output / "fixture.png"
    fixture = finder.vision_png(64, 64)
    if image.exists() and image.read_bytes() != fixture:
        raise RuntimeError("fixture changed")
    image.write_bytes(fixture)
    identity = runtime_identity(args.server)
    new, done, tokens = 0, 0, 0
    for case in matrix:
        if args.limit is not None and new >= args.limit:
            break
        print("CASE", case["key"], flush=True)
        result, ran = run_case(finder, manifest, case, args.server, args.output, identity, image, args.port)
        new += ran
        done += 1
        tokens += result["matched_tokens"]
        print("PASS", done, "/", len(matrix), case["key"], "tokens", result["matched_tokens"], flush=True)
    (args.output / "summary.json").write_text(json.dumps({"total": len(matrix), "completed": done,
          "new": new, "matched_tokens": tokens, "runtime": identity}, indent=2) + "\n")


if __name__ == "__main__":
    main()
