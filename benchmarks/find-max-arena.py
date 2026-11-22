#!/usr/bin/env python3
"""Find a physical arena limit with runtime probes of a text or image workload.

Does not stop containers, download models, enable UVM, or reserve arbitrary VRAM.
The result is workload-specific, not a guarantee for other image/prompt shapes.
With --mmproj and no --image, report validated landscape and portrait resolutions.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/server-ab"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from server_ab import request_json  # noqa: E402
from run_dflash_controls import ngram_command, ngram_settings, speculative_source_stats, validate_mixed_sources  # noqa: E402

SPEC = importlib.util.spec_from_file_location("arena_sweep", ROOT / "benchmarks/run-fixed-span-sweep.py")
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)
MARKER = "<__media__>"
REMOVED_ENV = ("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_CUDA_PREFER_MODEL_WEIGHTS", "GGML_CUDA_PREFER_KV_HOST",
               "GGML_CUDA_KV_ACCESSED_BY_GPU", "GGML_CUDA_DISABLE_GRAPHS", "LLAMA_RS_HOST_SPILL", "LLAMA_RS_STAGED_PUBLICATION",
               "LD_PRELOAD", "DFLASH_CONTROL_TRACE", "DFLASH_CONTROL_CAPTURE")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    parser.add_argument("--server", type=Path, default=ROOT /"build-v2/bin" /executable)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context-size", type=int, default=262144)
    parser.add_argument("--parallel", type=int, choices=range(1, 257), default=1,
                        help="adaptive slots; each slot and the selected group retain the full context-size limit")
    parser.add_argument("--probe-selected", type=int,
                        help="joint probe request count within configured slots; omitted uses the native batch/context limit")
    parser.add_argument("--cache-type-k", choices=SWEEP.KV_TYPES, default="q8_0")
    parser.add_argument("--cache-type-v", choices=SWEEP.KV_TYPES, default="q4_0")
    parser.add_argument("--spec-mode", choices=("none", "mtp", "dflash", "dflash-k4v"), default="none")
    parser.add_argument("--ngram-n", type=int, help="mixed K4V lookup length (default: 12)")
    parser.add_argument("--ngram-m", type=int, help="mixed K4V maximum proposals (default: 7; supported: 1-7)")
    parser.add_argument("--ngram-min-hits", type=int, help="mixed K4V minimum hits (default: 2)")
    parser.add_argument("--draft-length", type=int)
    parser.add_argument("--model-draft", type=Path)
    parser.add_argument("--mmproj", type=Path, help="enable vision; without --image, report validated per-ratio resolutions")
    parser.add_argument("--image", type=Path, action="append", default=[], help="add image-workload validation; repeat for a batch")
    parser.add_argument("--prompt-file", type=Path, help="literal image-request prompt; one <__media__> per image")
    parser.add_argument("--prefill-text", type=Path, default=ROOT /"benchmarks/data/online-articles-262144-words.txt")
    parser.add_argument("--prompt-tokens", type=int, help="text-only prefill length; default fills context minus outputs and headroom")
    parser.add_argument("--probe-mode", choices=("full", "preflight"), default="full",
                        help="full: test the requested workload for every candidate; preflight: use bounded text history plus real decoding")
    parser.add_argument("--validate-selected", action="store_true",
                        help="after a preflight search, run the full requested workload once at the selected arena; fail if it does not fit")
    parser.add_argument("--copy-probe", action="store_true",
                        help="also validate a separate 32-output copying path; always enabled for mixed K4V, optional for matched non-mixed sizing")
    parser.add_argument("--no-webui", action="store_true", help="disable the test server's web UI")
    parser.add_argument("--no-warmup", action="store_true", help="skip startup warmup; retain all real workload probes")
    parser.add_argument("--verbosity", type=int, default=3, help="server log threshold 0-5; route proofs require at least 4")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--image-min-tokens", type=int, help="minimum image budget (auto: projector metadata; explicit images: 64)")
    parser.add_argument("--image-max-tokens", type=int, help="image-budget search ceiling (auto: projector metadata; explicit images: 4096)")
    parser.add_argument("--initial-arena-mib", type=int, default=2240, help="optional initial search guess, not the final arena size or a safety reserve")
    parser.add_argument("--port", type=int, default=1249)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, default=ROOT /"benchmarks/results/arena-fit")
    args = parser.parse_args(argv)
    if args.image:
        if args.image_min_tokens is None:args.image_min_tokens=64
        if args.image_max_tokens is None:args.image_max_tokens=4096
    if args.validate_selected and args.probe_mode != "preflight":
        parser.error("--validate-selected requires --probe-mode preflight")
    if args.draft_length is None:
        args.draft_length = {"none":0, "mtp":3, "dflash":7, "dflash-k4v":7}[args.spec_mode]
    if not (args.spec_mode == "none" and args.draft_length == 0
            or args.spec_mode == "mtp" and 1 <= args.draft_length <= 3
            or args.spec_mode in ("dflash", "dflash-k4v") and 1 <= args.draft_length <= 7):
        parser.error("target-only uses length 0; MTP supports 1-3; DFlash supports 1-7")
    try:
        ngram_settings(args)
    except ValueError as error:
        parser.error(str(error))
    if args.spec_mode in ("dflash", "dflash-k4v") and not args.model_draft:
        parser.error("--model-draft is required for DFlash")
    if args.spec_mode == "none" and args.model_draft:
        parser.error("--model-draft requires MTP or DFlash")
    if args.image and not args.mmproj:
        parser.error("--image requires --mmproj")
    if args.prompt_file and not args.image or args.prompt_tokens is not None and args.image:
        parser.error("--prompt-file requires image validation; --prompt-tokens is for text probes")
    if args.copy_probe and args.image:
        parser.error("--copy-probe is a text-only resource check; omit it for image validation")
    if (not 512 <= args.context_size <= 262144 or not 8 <= args.decode_tokens < args.context_size -15
            or not 1 <= args.batch_size <= 2**31 -1 or not 0 <= args.ubatch_size <= 2**31 -1
            or not 1 <= args.initial_arena_mib <= 1048576 or not 1 <= args.port <= 65535 or not 0 <= args.verbosity <= 5
            or not math.isfinite(args.request_timeout) or not 1 <= args.request_timeout <= 43200
            or (args.image_min_tokens is not None and not 1 <= args.image_min_tokens <= 65536)
            or (args.image_max_tokens is not None and not 1 <= args.image_max_tokens <= 65536)
            or (args.image_min_tokens is not None and args.image_max_tokens is not None and args.image_min_tokens > args.image_max_tokens)
            or (args.image and args.image_max_tokens is not None and args.image_max_tokens > args.context_size) or len(args.image) > 32):
        parser.error("invalid context, output, batch, image, arena, timeout or port bounds")
    if args.probe_selected is not None and not 1 <= args.probe_selected <= args.parallel:
        parser.error("--probe-selected must be within configured slot capacity")
    if args.prompt_tokens is not None and not 1 <= args.prompt_tokens <= args.context_size -args.decode_tokens -15:
        parser.error("prefill must leave room for outputs and 15 speculative headroom positions")
    return args


def server_command(args, arena_mib):
    command = [str(args.server), "--model", str(args.model), "--ctx-size", str(args.context_size),
               "--batch-size", str(args.batch_size), "--ubatch-size", str(args.ubatch_size), "--parallel", str(getattr(args, "parallel", 1)),
               "--n-gpu-layers", "999", "--flash-attn", "on", "--cache-type-k", args.cache_type_k,
               "--cache-type-v", args.cache_type_v, "--shared-device-memory-mib", str(arena_mib), "--fit", "off",
               "--spec-type", {"none":"none", "mtp":"draft-mtp", "dflash":"draft-dflash", "dflash-k4v":"ngram-map-k4v,draft-dflash"}[args.spec_mode],
               "--host", "127.0.0.1", "--port", str(args.port), "--offline", "--no-context-shift", "--cache-ram", "0",
               "-lv", str(max(args.verbosity, 4 if args.spec_mode == "dflash-k4v" or args.copy_probe else 0))]
    if getattr(args, "parallel", 1) > 1:
        command.append("--kv-unified")
    if args.no_webui:
        command.append("--no-webui")
    if getattr(args, "no_warmup", False):
        command.append("--no-warmup")
    if args.draft_length:
        command += ["--spec-draft-n-max", str(args.draft_length), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"]
    if args.model_draft:
        command += ["--model-draft", str(args.model_draft), "--spec-draft-ngl", "999"]
    command += ngram_command(args)
    if args.mmproj:
        command += ["--mmproj", str(args.mmproj)]
        if args.image_min_tokens is not None:command += ["--image-min-tokens",str(args.image_min_tokens)]
        if args.image_max_tokens is not None:command += ["--image-max-tokens",str(args.image_max_tokens)]
    else:
        command.append("--no-mmproj")
    return command


def environment(server=None):
    env = SWEEP.server_environment(False)
    for name in list(env):
        if name.startswith("LLAMA_ARG_") or name in REMOVED_ENV:
            env.pop(name)
    env["LLAMA_MEDIA_MARKER"] = MARKER
    if server is not None:
        runtime = Path(server).resolve().parent
        variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        paths = []
        for value in env.get(variable, "").split(os.pathsep):
            if not value:
                continue
            directory = Path(value).resolve()
            old_build = directory.is_relative_to(ROOT) and any(part.startswith("build") for part in directory.relative_to(ROOT).parts)
            old_runtime = any(directory.glob("libllama*")) or any(directory.glob("llama.dll")) or any(directory.glob("libggml*"))
            if directory != runtime and not old_build and not old_runtime:
                paths.append(value)
        env[variable] = os.pathsep.join([str(runtime), *paths])
    return env


_runtime_library_hashes = {}


def loaded_runtime_libraries(pid):
    if not isinstance(pid, int):
        return []
    maps = Path(f"/proc/{pid}/maps")
    if not maps.is_file():
        return []
    paths = {Path(parts[5]) for line in maps.read_text().splitlines()
             if len(parts := line.split(maxsplit=5)) == 6 and parts[5].startswith("/")
             and Path(parts[5]).name.startswith(("libllama", "libggml", "libcudart", "libcublas"))}
    result = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        stat = path.stat()
        key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in _runtime_library_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1048576), b""):
                    digest.update(chunk)
            _runtime_library_hashes[key] = digest.hexdigest()
        result.append({"path": str(path), "sha256": _runtime_library_hashes[key]})
    return result


def shell_commands(command):
    def quote(value):
        return "'" +value.replace("'", "''") +"'"
    return {"posix":shlex.join(command), "powershell":"& " +" ".join(quote(value) for value in command)}


def bounded_bytes(path, maximum=64 *1048576):
    with path.open("rb") as source:
        data = source.read(maximum +1)
    if not data or len(data) > maximum:
        raise ValueError(f"input must contain 1 byte to {maximum} bytes: {path}")
    return data


def requested_text_tokens(args):
    return args.prompt_tokens or args.context_size -args.decode_tokens -15


def probe_text_tokens(args):
    requested = requested_text_tokens(args)
    if args.probe_mode == "preflight":
        # Exercise wide prefill and decode transitions without filling the whole history.
        return min(requested, max(2048, 2 *max(args.batch_size, args.ubatch_size)))
    return requested


def context_semantics(args):
    slots = getattr(args, "parallel", 1)
    return {"slots":slots, "per_slot":args.context_size, "selected_group":args.context_size,
            "all_histories":slots *args.context_size}


def validate_context_semantics(args, props):
    if getattr(args, "parallel", 1) <= 1:
        return
    expected = context_semantics(args)
    observed = props.get("stream_context_limits", {})
    if (props.get("total_slots") != expected["slots"]
            or any(observed.get(key) != expected[key] for key in ("per_slot", "selected_group", "all_histories"))
            or type(observed.get("selected_storage")) is not int or observed["selected_storage"] < args.context_size):
        raise RuntimeError(f"server reported incompatible multi-slot context limits: {props}")


def multi_slot_probe_profiles(args):
    if getattr(args, "parallel", 1) <= 1:
        return []
    slots = args.parallel
    count = probe_text_tokens(args)
    active = min(slots, getattr(args, "probe_selected", None) or slots, args.context_size // (args.decode_tokens + 16),
                 args.batch_size // (args.draft_length + 1))
    if not active:
        raise ValueError("context cannot fit even one probe request")
    joint = min(count, (args.context_size - active * (args.decode_tokens + 15)) // active)
    profiles = [{"label":f"single-{slot}", "slots":[slot], "prompt_tokens":[count], "selected_max":1}
                for slot in range(slots)]
    profiles.append({"label":"joint", "slots":list(range(active)), "prompt_tokens":[joint] * active, "selected_max":active})
    if slots > 2 and active > 1:
        profiles.append({"label":"asymmetric", "slots":list(range(active)),
                         "prompt_tokens":[max(1, joint // 2)] + [joint] * (active - 1), "selected_max":active})
    if args.probe_mode == "full" and count * 2 > args.context_size:
        profiles.append({"label":"serialized", "slots":list(range(slots)), "prompt_tokens":[count] * slots, "selected_max":1})
    return profiles


def payload(args, prompt, images):
    if images:
        if not isinstance(prompt, str) or prompt.count(MARKER) != len(images):
            raise ValueError("image prompt needs exactly one media marker per image")
        prompt = {"prompt_string":prompt, "multimodal_data":images}
    return {"prompt":prompt, "n_predict":args.decode_tokens, "temperature":0, "seed":123,
            "ignore_eos":True, "cache_prompt":False, "return_tokens":True, "stream":False}


def validate_result(args, result, expected_prompt):
    timing = result.get("timings", {})
    if len(result.get("tokens", [])) != args.decode_tokens or timing.get("predicted_n") != args.decode_tokens:
        raise RuntimeError("incomplete decode response")
    count = timing.get("prompt_n")
    if type(count) is not int or count <= 0 or count +args.decode_tokens +15 > args.context_size:
        raise RuntimeError("request exceeded the configured context/headroom or omitted prompt timing")
    if expected_prompt is not None and count != expected_prompt:
        raise RuntimeError("incomplete prefill response")
    if args.spec_mode != "none" and (type(timing.get("draft_n")) is not int or timing["draft_n"] <= 0):
        raise RuntimeError("speculative workload completed without generating drafts")


def request_completion_group(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(64 *1048576 +1)
        if len(data) > 64 *1048576:
            raise RuntimeError("completion group response exceeds 64 MiB")
        rows = json.loads(data)
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"completion group request failed for {url}: {error}") from error
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("expected an array of completion objects")
    return rows


def observed_completion(args, url, body, label, grouped=False):
    log_path = getattr(args, "_source_log_path", None)
    before = speculative_source_stats(log_path.read_text(errors="replace")) if log_path else {}
    result = request_completion_group(url +"/completion", body, timeout=args.request_timeout) if grouped else request_json(url +"/completion", body, timeout=args.request_timeout)
    if log_path:
        after = speculative_source_stats(log_path.read_text(errors="replace"))
        fields = ("generated_tokens", "accepted_tokens", "accept_calls", "prediction_forwards")
        delta = {name:{key:row[key] -before.get(name, {}).get(key, 0) for key in fields} for name, row in after.items()}
        rows = result if isinstance(result, list) else [result]
        timing = {key:sum(row["timings"].get(key, 0) for row in rows) for key in ("draft_n", "draft_n_accepted")}
        args._source_groups.append({"label":label, "members":len(rows), "timings":timing, "source_stats":delta})
    return result


def image_slot_requests(args, url, prompt, images):
    body = payload(args, prompt, images)
    slots = getattr(args, "probe_selected", None) or getattr(args, "parallel", 1)
    if slots > 1:
        body["prompt"] = [body["prompt"] for _ in range(slots)]
    result = observed_completion(args, url, body, "image", grouped=slots > 1)
    rows = [result] if isinstance(result, dict) else result
    if not isinstance(rows, list) or len(rows) != slots:
        raise RuntimeError("incomplete image completion group")
    for row in rows:
        validate_result(args, row, None)
    if slots > 1 and sorted(row.get("id_slot", -1) for row in rows) != list(range(slots)):
        raise RuntimeError("image group did not use independent configured slots")
    return rows


def multi_slot_requests(args, url, tokens, candidate, log_path):
    observations = []
    for profile in multi_slot_probe_profiles(args):
        prompts = [tokens[:count] for count in profile["prompt_tokens"]]
        if any(len(prompt) != count for prompt, count in zip(prompts, profile["prompt_tokens"])):
            raise ValueError("corpus is too short for multi-slot probes")
        body = payload(args, prompts[0] if len(prompts) == 1 else prompts, [])
        if len(prompts) == 1:
            body["id_slot"] = profile["slots"][0]
        offset = log_path.stat().st_size
        result = observed_completion(args, url, body, profile["label"], grouped=len(prompts) > 1)
        rows = [result] if isinstance(result, dict) else result
        if not isinstance(rows, list) or len(rows) != len(prompts):
            raise RuntimeError("incomplete multi-slot completion group")
        for row, count in zip(rows, profile["prompt_tokens"]):
            validate_result(args, row, count)
        ids = [row.get("id_slot") for row in rows]
        if any(type(slot) is not int for slot in ids) or sorted(ids) != profile["slots"]:
            raise RuntimeError("multi-slot response did not identify the requested independent slots")
        with log_path.open("rb") as source:
            source.seek(offset)
            text = source.read().decode("utf-8", errors="replace")
        selected = [int(value) for value in re.findall(r"stream group bind: selected=(\d+) rows=\d+ phase=decode", text)]
        if not selected or max(selected) != profile["selected_max"]:
            rejected = re.findall(r"stream selection capacity: selected=" + str(profile["selected_max"]) + r" required=(\d+) available=\d+", text)
            if profile["selected_max"] > 1 and rejected:
                raise SWEEP.ArenaTooSmallError((max(map(int, rejected)) +1048575) //1048576)
            raise RuntimeError(f"multi-slot {profile['label']} selection was not exercised: {selected}")
        artifact = args.output /f"arena-{candidate}-{profile['label']}-response.json"
        artifact.write_text(json.dumps({"profile":profile, "request":body, "responses":rows,
                                       "decode_selections":selected}, indent=2) +"\n")
        observations.append({"profile":profile, "responses":rows, "artifact":str(artifact),
                             "sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(), "decode_selections":selected})
    return observations


def mixed_route_probe(args, url, prompt, candidate):
    settings = ngram_settings(args)
    if settings is None and not args.copy_probe:
        return None
    count = 32
    length = min(len(prompt), args.context_size -count -15)
    tail = 2 *settings["ngram_n"] +settings["ngram_m"] +2 if settings else 33
    if length < tail:
        raise ValueError("mixed route proof needs at least 2*N+M+2 prompt tokens with decode headroom")
    ids = request_json(url +"/tokenize", {"content":"B", "add_special":False}, timeout=args.request_timeout)["tokens"]
    if len(ids) != 1 or type(ids[0]) is not int or not 0 <= ids[0] <= 2**31 -1:
        raise ValueError("mixed route proof requires a single ordinary token for B")
    proof_prompt = list(prompt[:length])
    proof_prompt[-tail:] = ids *tail
    config = argparse.Namespace(**vars(args))
    config.decode_tokens = count
    request = payload(config, proof_prompt, [])
    request["logit_bias"] = [[ids[0], 1000]]
    result = request_json(url +"/completion", request, timeout=args.request_timeout)
    path = args.output /f"arena-{candidate}-mixed-proof.json"
    path.write_text(json.dumps({"payload":request, "result":result}, indent=2) +"\n")
    validate_result(config, result, length)
    if result["tokens"] != ids *count:
        raise RuntimeError("mixed route proof violated the target sampling constraint")
    return {"response":str(path), "response_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt_tokens":length, "decode_tokens":count, "ngram":settings, "timings":result["timings"],
            "scope":"Separate target-bias-constrained text path/resource check; mixed mode additionally proves both sources. Not performance or model quality evidence"}


def mixed_image_route_probe(args, url, images, candidate):
    settings = ngram_settings(args)
    if settings is None or not images:
        raise ValueError("mixed image route proof requires configured K4V and real image inputs")
    tail = 2 *settings["ngram_n"] +settings["ngram_m"] +2
    count = max(32, (args.draft_length +1) *settings["ngram_min_hits"] +settings["ngram_m"] +2)
    if tail +count +15 >= args.context_size:
        raise ValueError("context cannot hold the mixed image proof history and outputs")
    fragment = " B"
    tokenized = request_json(url +"/tokenize", {"content":fragment, "add_special":False}, timeout=args.request_timeout)["tokens"]
    if len(tokenized)!=1 or type(tokenized[0]) is not int or not 0<=tokenized[0]<=2**31 -1:
        raise ValueError("mixed image route proof requires a single ordinary token")
    repeated = request_json(url +"/tokenize", {"content":fragment *tail, "add_special":False}, timeout=args.request_timeout)["tokens"]
    if repeated != tokenized *tail:
        raise ValueError("mixed image proof history does not preserve the repeated token")
    media = " ".join([MARKER] *len(images))
    requests: list[dict[str, Any]] = []
    for route, prompt, outputs in (("image-miss", fragment *tail +" Describe: " +media +" Answer:", args.draft_length +2),
                                   ("image-copy", "Describe: " +media +fragment *tail, count)):
        config = argparse.Namespace(**vars(args))
        config.decode_tokens=outputs
        body = payload(config,prompt,images)
        body["logit_bias"]=[[tokenized[0],1000]]
        result = observed_completion(args,url,body,route)
        path = args.output /f"arena-{candidate}-mixed-{route}.json"
        path.write_text(json.dumps({"payload":body,"result":result},indent=2) +"\n")
        validate_result(config,result,None)
        if result["tokens"] != tokenized *outputs:
            raise RuntimeError("mixed image route proof violated the target sampling constraint")
        requests.append({"route":route,"response":str(path),"response_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "decode_tokens":outputs,"timings":result["timings"]})
    timing = {key:sum(row["timings"].get(key,0) for row in requests) for key in ("draft_n","draft_n_accepted","predicted_n","prompt_n")}
    return {"requests":requests,"timings":timing,"ngram":settings,
            "scope":"Separate image-conditioned target-biased copy and genuine-miss resource checks. Not performance or image quality evidence."}


def validate_completed_image_sources(text, completed, groups=None):
    totals = {key:sum(row["timings"].get(key,0) for row in completed) for key in ("draft_n","draft_n_accepted")}
    result = validate_mixed_sources(text,generated_tokens=totals["draft_n"],accepted_tokens=totals["draft_n_accepted"])
    if groups is not None:
        if sum(group["members"] for group in groups) != len(completed):
            raise RuntimeError("mixed source barriers do not cover every completed response")
        for group in groups:
            delta = group["source_stats"]
            if set(delta) != set(result) or any(value < 0 for row in delta.values() for value in row.values()):
                raise RuntimeError("missing or decreasing mixed source group counters")
            for source_key, timing_key in (("generated_tokens", "draft_n"), ("accepted_tokens", "draft_n_accepted")):
                if sum(row[source_key] for row in delta.values()) != group["timings"].get(timing_key, 0):
                    raise RuntimeError("mixed source group totals differ from completed responses")
            if group["label"] == "image-copy" and not (
                    delta["ngram-map-k4v"]["generated_tokens"] > 0
                    and delta["ngram-map-k4v"]["accept_calls"] > 0
                    and delta["draft-dflash"]["prediction_forwards"] == 0):
                raise RuntimeError("image copy group did not use K4V without a neural forward")
            if group["label"] == "image-miss" and not (
                    delta["draft-dflash"]["generated_tokens"] > 0
                    and delta["draft-dflash"]["prediction_forwards"] > 0
                    and delta["ngram-map-k4v"]["generated_tokens"] == 0):
                raise RuntimeError("image miss group did not exercise genuine neural fallback")
        return result
    snapshots = {}
    for line in text.splitlines():
        for name,row in speculative_source_stats(line).items():
            snapshots.setdefault(row["begin_calls"],{})[name]=row
    fields = ("generated_tokens","accepted_tokens","accept_calls","prediction_forwards")
    for ordinal,request in enumerate(completed,1):
        current,previous = snapshots.get(ordinal),snapshots.get(ordinal -1)
        if not current or set(current)!=set(result) or ordinal>1 and (not previous or set(previous)!=set(result)):
            raise RuntimeError("missing per-request mixed image source counters")
        delta = {name:{key:row[key] -(previous[name][key] if previous else 0) for key in fields} for name,row in current.items()}
        if any(value<0 for row in delta.values() for value in row.values()):
            raise RuntimeError("mixed image source counters moved backward")
        for source_key,timing_key in (("generated_tokens","draft_n"),("accepted_tokens","draft_n_accepted")):
            if sum(row[source_key] for row in delta.values())!=request["timings"].get(timing_key,0):
                raise RuntimeError("per-request mixed image source totals differ from completed response")
        if request["label"]=="image-copy" and not (delta["ngram-map-k4v"]["generated_tokens"]>0 and delta["ngram-map-k4v"]["accept_calls"]>0):
            raise RuntimeError("image copy route did not exercise post-image K4V")
        if request["label"]=="image-miss" and not (delta["draft-dflash"]["generated_tokens"]>0
                                                   and delta["draft-dflash"]["prediction_forwards"]>0 and delta["ngram-map-k4v"]["generated_tokens"]==0):
            raise RuntimeError("image miss route did not exercise genuine neural fallback")
        request["source_stats"] = delta
    return result


def _runtime_probe(args, candidate):
    try:
        with socket.socket() as available:
            if os.name == "posix":
                available.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            available.bind(("127.0.0.1", args.port))
    except OSError as error:
        raise RuntimeError(f"port {args.port} is unavailable; choose --port for this finder") from error
    log_path = args.output /f"arena-{candidate}.log"
    command = server_command(args, candidate)
    failure = None
    result = {}
    mixed_proof = None
    vision_result = None
    multi_profiles = None
    observed_context = None
    runtime_libraries = []
    image_rows = None
    response_path = None
    prompt_sha256 = None
    completed: list[dict[str, Any]] = []
    mixed_image = bool(args.mmproj and args.spec_mode=="dflash-k4v")
    expected_phases = []
    if getattr(args, "parallel", 1) > 1 and args.spec_mode == "dflash-k4v":
        args = argparse.Namespace(**vars(args))
        args._source_log_path, args._source_groups = log_path, []
    with log_path.open("x", encoding="utf-8") as log:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment(args.server), creationflags=flags)
        try:
            url = f"http://127.0.0.1:{args.port}"
            SWEEP.wait_ready(url, process, log_path)
            runtime_libraries = loaded_runtime_libraries(getattr(process, "pid", None))
            if getattr(args, "parallel", 1) > 1:
                props = request_json(url +"/props", timeout=args.request_timeout)
                validate_context_semantics(args, props)
                observed_context = props["stream_context_limits"]
            expected = None
            if getattr(args,"_vision_validate_text_first",False):
                prelude=argparse.Namespace(**vars(args))
                prelude.probe_mode="preflight"
                count=probe_text_tokens(prelude)
                corpus=bounded_bytes(args.prefill_text).decode("utf-8")
                ids=request_json(url +"/tokenize",{"content":corpus,"add_special":True},timeout=args.request_timeout)["tokens"]
                if len(ids)<count:raise ValueError("corpus is too short for vision transition validation")
                response=observed_completion(args,url,payload(args,ids[:count],[]),"text-prelude")
                validate_result(args,response,count)
                completed.append({"label":"text-prelude","timings":response["timings"]})
                (args.output /f"arena-{candidate}-text-prelude.json").write_text(json.dumps(response,indent=2) +"\n")
            if args.image:
                images = [base64.b64encode(bounded_bytes(path)).decode("ascii") for path in args.image]
                prompt = bounded_bytes(args.prompt_file).decode("utf-8") if args.prompt_file else \
                    "Describe these images: " + " ".join([MARKER] *len(images))
            else:
                expected = probe_text_tokens(args)
                corpus = bounded_bytes(args.prefill_text).decode("utf-8")
                tokens = request_json(url +"/tokenize", {"content":corpus, "add_special":True}, timeout=args.request_timeout)["tokens"]
                if len(tokens) < expected:
                    raise ValueError("corpus is too short; provide more text or set --prompt-tokens")
                prompt, images = tokens[:expected], []
            if getattr(args, "parallel", 1) > 1 and not images:
                multi_profiles = multi_slot_requests(args, url, tokens, candidate, log_path)
                result = multi_profiles[0]["responses"][0]
            elif getattr(args, "parallel", 1) > 1 and images:
                image_rows = image_slot_requests(args, url, prompt, images)
                result = image_rows[0]
            else:
                result = request_json(url +"/completion", payload(args,prompt,images), timeout=args.request_timeout)
            response_path = args.output /f"arena-{candidate}-response.json"
            response_path.write_text(json.dumps(result, indent=2) +"\n")
            if not args.image:
                prompt_sha256 = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            validate_result(args, result, expected)
            if image_rows:
                completed.extend({"label":"image-slot-" +str(row["id_slot"]), "timings":row["timings"]} for row in image_rows)
                (args.output /f"arena-{candidate}-image-group-responses.json").write_text(json.dumps(image_rows,indent=2) +"\n")
            elif multi_profiles:
                completed.extend({"label":profile["profile"]["label"], "timings":row["timings"]}
                                 for profile in multi_profiles for row in profile["responses"])
            else:
                completed.append({"label":"image" if args.image else "text","timings":result["timings"]})
            if getattr(args,"_expected_vision_shape",None):expected_phases.extend([args._expected_vision_shape] *len(image_rows or [result]))
            if not mixed_image:
                mixed_proof = mixed_route_probe(args, url, prompt, candidate)
                if mixed_proof:completed.append({"label":"text-proof","timings":mixed_proof["timings"]})
            bootstrap=getattr(args,"_vision_bootstrap",None)
            if bootstrap:
                vision_result=[]
                for item in bootstrap:
                    data=base64.b64encode(bounded_bytes(Path(item["image"]))).decode("ascii")
                    rows=image_slot_requests(args,url,"Describe: " +MARKER,[data])
                    response=rows[0]
                    completed.extend({"label":"vision-" +item["ratio"],"timings":row["timings"]} for row in rows)
                    expected_phases.extend([item["shape"]] *len(rows))
                    path=args.output /f"arena-{candidate}-vision-{item['ratio'].replace(':','x')}-response.json"
                    path.write_text(json.dumps(response,indent=2) +"\n")
                    vision_result.append({"ratio":item["ratio"],"response":str(path),
                                          "response_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"timings":response["timings"],
                                          "responses":rows,"slots":len(rows),"image":item["image"],"image_sha256":hashlib.sha256(Path(item["image"]).read_bytes()).hexdigest()})
            if mixed_image:
                proof_images=images if args.image else [base64.b64encode(bounded_bytes(Path(bootstrap[0]["image"]))).decode("ascii")] if bootstrap else []
                mixed_proof=mixed_image_route_probe(args,url,proof_images,candidate)
                for row in mixed_proof["requests"]:completed.append({"label":row["route"],"timings":row["timings"]})
                if args.image:
                    shape=getattr(args,"_expected_vision_shape",None)
                else:
                    assert bootstrap
                    shape=bootstrap[0]["shape"]
                if shape:expected_phases.extend([shape] *(2 *len(proof_images)))
        except Exception as error:
            failure = error
        finally:
            SWEEP.stop_process(process)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"illegal memory access|misaligned address|device-side assert|unspecified launch failure", text, re.IGNORECASE):
        raise RuntimeError(f"probe reported a device fault, not an arena capacity boundary; see {log_path}")
    if isinstance(failure, SWEEP.ArenaTooSmallError):
        raise failure
    if failure is None:
        try:
            SWEEP.validate_execution_log(text)
            if expected_phases:check_vision_phases(text,expected_phases,candidate)
            elif mixed_image:
                grouped=getattr(args,"parallel",1)>1
                phases=[dict(re.findall(r"([a-z][a-z0-9_]*)=([^\s]+)",line)) for line in text.splitlines()
                        if ("vision_group_phase:" if grouped else "vision_phase:") in line]
                if grouped:
                    returned=len(phases)>=4 and all(row.get("grants_after")=="0" and row.get("selected_state_bytes")=="0" for row in phases)
                else:
                    returned=len(phases)==3 *len(args.image) and all(row.get("borrowed_after")=="0" and row.get("suspended_after")=="0" and row.get("diagnostics")=="1" for row in phases)
                if not returned:
                    raise RuntimeError("mixed image probes did not complete and return all encoder loans")
            if args.spec_mode == "dflash-k4v":
                if mixed_proof is None:
                    raise RuntimeError("missing completed mixed route probe")
                mixed_proof["source_stats"] = (
                    validate_completed_image_sources(text, completed, groups=getattr(args, "_source_groups", None))
                    if mixed_image else validate_mixed_sources(
                        text, generated_tokens=sum(row["timings"].get("draft_n", 0) for row in completed),
                        accepted_tokens=sum(row["timings"].get("draft_n_accepted", 0) for row in completed)))
            elif mixed_proof is not None:
                mixed_proof["source_stats"] = speculative_source_stats(text)
                for source_key, timing_key in (("generated_tokens", "draft_n"), ("accepted_tokens", "draft_n_accepted")):
                    expected = sum(row["timings"].get(timing_key,0) for row in completed)
                    if sum(row[source_key] for row in mixed_proof["source_stats"].values()) != expected:
                        raise ValueError("copy-path source totals differ from completed request counters")
        except (RuntimeError, ValueError) as error:
            failure = error
    minimum = SWEEP.arena_minimum_mib(text, include_mtp_fallback_hint=failure is None or SWEEP.is_arena_capacity_failure(text))
    if minimum is not None:
        raise SWEEP.ArenaTooSmallError(minimum)
    context_limit=bool(failure and getattr(args,"_vision_capacity_probe",False)
                       and re.search(r"request .*exceeds the available context size|request .*exceeds.*context size",text,re.I))
    if failure and not context_limit and not SWEEP.is_arena_capacity_failure(text):
        raise RuntimeError(f"probe failed; not a physical OOM: {failure}; see {log_path}") from failure
    return {"arena_mib":candidate, "fits":failure is None, "log":str(log_path), "command":command,
            "log_sha256":hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "loaded_runtime_libraries":runtime_libraries,
            "response":str(response_path) if response_path else None,
            "response_sha256":hashlib.sha256(response_path.read_bytes()).hexdigest() if response_path else None,
            "prompt_ids_sha256":prompt_sha256, "mixed_proof":mixed_proof, "completed_requests":completed,
            "source_groups":getattr(args,"_source_groups",None),
            "context_semantics":context_semantics(args), "observed_context_limits":observed_context, "selection_profiles":multi_profiles,
            "probe_mode":args.probe_mode, "timings":result.get("timings"), "error":str(failure) if failure else None,
            "vision_bootstrap":vision_result,"vision_capacity_requested":bool(args.mmproj),
            "limiting_factor":"context" if context_limit else "runtime_capacity" if failure else None}


VISION_RATIOS = ("1:1", "4:3", "3:4", "16:9", "9:16", "2:1", "1:2")


def vision_png(width, height):
    import struct
    import zlib
    if type(width) is not int or type(height) is not int or not 1 <= width <= 65536 or not 1 <= height <= 65536 or width *height > 64 *1024 *1024:
        raise ValueError("vision capacity fixture dimensions are out of range")

    def chunk(kind, data):
        return struct.pack(">I", len(data)) +kind +data +struct.pack(">I", zlib.crc32(kind +data) &0xffffffff)
    compressor = zlib.compressobj()
    encoded = bytearray()
    row = b"\x00" +bytes((48, 112, 176)) *width
    for _ in range(height):
        encoded.extend(compressor.compress(row))
    encoded.extend(compressor.flush())
    return b"\x89PNG\r\n\x1a\n" +chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +chunk(b"IDAT", bytes(encoded)) +chunk(b"IEND", b"")


def validate_vision_workspace(workspace, compute_bytes):
    names=("graph_bytes", "operator_bytes", "library_bytes", "padding_bytes", "total_bytes")
    if (not isinstance(workspace, dict) or any(type(workspace.get(key)) is not int or workspace[key] < 0 for key in names)
            or not workspace["graph_bytes"] or not workspace["total_bytes"]
            or sum(workspace[key] for key in names[:-1]) != workspace["total_bytes"] or workspace["total_bytes"] != compute_bytes):
        raise RuntimeError("invalid or incomplete vision workspace components")
    return workspace


def validate_vision_plan(plan, arena_mib):
    if not isinstance(plan, dict) or plan.get("schema") != "vision-memory-v2" or plan.get("arena_bytes") != arena_mib *1048576:
        raise RuntimeError("missing or mismatched server vision-memory report; use the matching server revision")

    def integer(value):
        return type(value) is int and value > 0
    if (plan.get("family") not in ("qwen3vl", "gemma4") or not integer(plan.get("alignment"))
            or not integer(plan.get("min_image_tokens")) or not integer(plan.get("effective_image_max_tokens"))
            or not 1 <= plan["min_image_tokens"] <= plan["effective_image_max_tokens"] <= 65536 or plan.get("images_per_probe") != 1):
        raise RuntimeError("invalid vision geometry report")
    if (plan.get("controlled_workspace") != ["graph", "operator", "library"]
            or plan.get("opaque_runtime_allocations_included") is not False or not integer(plan.get("weight_bytes"))):
        raise RuntimeError("vision report does not cover the complete controlled workspace")
    rows = plan.get("ratios")
    if not isinstance(rows, list) or len(rows) != len(VISION_RATIOS) or {r.get("ratio") for r in rows if isinstance(r, dict)} != set(VISION_RATIOS):
        raise RuntimeError("incomplete or repeated vision ratio coverage")
    for row in rows:
        p, q = map(int, row["ratio"].split(":"))
        seen = set()
        identifiers = set()
        last = None
        if not isinstance(row.get("candidates"), list):
            raise RuntimeError("invalid vision shape list")
        for shape in row["candidates"]:
            fields = ("width", "height", "input_width", "input_height", "image_tokens", "image_max_tokens", "weight_bytes", "compute_bytes", "required_arena_bytes")
            if any(not integer(shape.get(key)) for key in fields):
                raise RuntimeError("invalid vision shape values")
            validate_vision_workspace(shape.get("workspace"), shape["compute_bytes"])
            if shape["weight_bytes"] != plan["weight_bytes"]:
                raise RuntimeError("vision shape weight storage differs from the configured projector")
            width, height, alignment = shape["width"], shape["height"], plan["alignment"]
            if (max(width,height,shape["input_width"],shape["input_height"]) > 65536 or width %alignment or height %alignment
                    or width //alignment *(height //alignment) != shape["image_tokens"]
                    or shape["input_width"] *q != shape["input_height"] *p
                    or not plan["min_image_tokens"] <= shape["image_max_tokens"] <= plan["effective_image_max_tokens"]
                    or shape["image_tokens"] > shape["image_max_tokens"]
                    or shape["weight_bytes"] +shape["compute_bytes"] > shape["required_arena_bytes"]
                    or shape["required_arena_bytes"] > plan["arena_bytes"] or (width,height) in seen
                    or (last is not None and shape["image_tokens"] > last)):
                raise RuntimeError("vision shape violates its native geometry or arena admission")
            if not integer(shape.get("shape_id")) or shape["shape_id"] in identifiers:
                raise RuntimeError("invalid vision shape identity")
            identifiers.add(shape["shape_id"])
            seen.add((width,height))
            last = shape["image_tokens"]
        indices=row.get("budget_shape_ids")
        if not isinstance(indices,list) or len(indices)!=plan["effective_image_max_tokens"] -plan["min_image_tokens"] +1 or any(type(i) is not int or i!=-1 and i not in identifiers for i in indices):
            raise RuntimeError("incomplete vision budget-to-shape coverage")
    return plan


def check_vision_phase(text, expected, arena_mib):
    return check_vision_phases(text,[expected],arena_mib)[0]


def check_vision_phases(text, expected, arena_mib):
    phases=[line for line in text.splitlines() if "vision_phase:" in line or "vision_group_phase:" in line]
    if len(phases)!=len(expected):raise RuntimeError("vision probe did not complete the planned encoder phases")
    return [check_vision_fields(dict(re.findall(r"([a-z][a-z0-9_]*)=([^\s]+)",line)),shape,arena_mib)
            for line,shape in zip(phases,expected)]


def check_vision_fields(phase, expected, arena_mib):
    fields = {"image_width":expected["width"], "image_height":expected["height"], "image_tokens":expected["image_tokens"], "parent":arena_mib *1048576}
    grouped="grants_after" in phase
    returned=(phase.get("grants_after")=="0" and phase.get("selected_state_bytes")=="0") if grouped else (
        phase.get("borrowed_after")=="0" and phase.get("suspended_after")=="0" and phase.get("diagnostics")=="1")
    if any(phase.get(key) != str(value) for key,value in fields.items()) or not returned:
        raise RuntimeError("actual vision shape or phase return differs from the server plan")
    if phase.get("weights") != str(expected["weight_bytes"]) or phase.get("compute") != str(expected["compute_bytes"]):
        raise RuntimeError("executed vision allocations differ from the planned shape")
    workspace=validate_vision_workspace(expected.get("workspace"),expected["compute_bytes"])
    components={"graph":"graph_bytes", "operator":"operator_bytes", "library":"library_bytes", "padding":"padding_bytes"}
    if any(phase.get(key) != str(workspace[value]) for key,value in components.items()):
        raise RuntimeError("executed vision workspace components differ from the controlled plan")
    lifetime=("grants_after","selected_state_bytes") if grouped else ("borrowed_after","suspended_after")
    return {key:int(phase[key]) for key in (*fields,"weights","compute",*components,*lifetime)}


def vision_plan_probe(args, candidate):
    command = server_command(args,candidate) +["--vision-memory-report"]
    path = args.output /f"arena-{candidate}-vision-plan.json"
    log_path = args.output /f"arena-{candidate}-vision-plan.log"
    with socket.socket() as available:
        if os.name == "posix":available.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        available.bind(("127.0.0.1",args.port))
    process = None
    with path.open("xb") as output, log_path.open("x") as log:
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(command,stdout=output,stderr=log,env=environment(args.server),creationflags=flags)
            code = process.wait(timeout=args.request_timeout)
        finally:
            if process is not None:SWEEP.stop_process(process)
    text = log_path.read_text(errors="replace")
    if re.search(r"illegal memory access|misaligned address|device-side assert|unspecified launch failure",text,re.I):
        raise RuntimeError("vision planning reported a device fault: " +str(log_path))
    if code:
        shared_vision_rejected = "unsupported vision arena model/projector capabilities" in text.lower()
        minimum = SWEEP.arena_minimum_mib(text,include_mtp_fallback_hint=SWEEP.is_arena_capacity_failure(text) or shared_vision_rejected)
        if minimum is not None:raise SWEEP.ArenaTooSmallError(minimum)
        if SWEEP.is_arena_capacity_failure(text):return None
        raise RuntimeError("server vision planning failed; see " +str(log_path))
    plan = validate_vision_plan(json.loads(bounded_bytes(path,32 *1048576)),candidate)
    plan["artifact"] = str(path)
    plan["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan["log"] = str(log_path)
    return plan


def probe_with_vision(args, candidate):
    if not args.mmproj or args.image or getattr(args,"_vision_bootstrap",None):
        return _runtime_probe(args,candidate)
    plan = vision_plan_probe(args,candidate)
    if plan is None:
        log=args.output /f"arena-{candidate}-vision-plan.log"
        return {"arena_mib":candidate,"fits":False,"stage":"vision_model_initialization",
                "command":server_command(args,candidate) +["--vision-memory-report"],"log":str(log),
                "log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()}
    shapes = [shape for row in plan["ratios"] for shape in row["candidates"]]
    if not shapes:
        minimum = plan["minimum_arena_bytes_lower_bound"]
        rejected = [v for row in plan["ratios"] if (v:=row.get("minimum_rejected_bytes")) is not None]
        if minimum <= candidate *1048576 and rejected:minimum = min(rejected)
        if minimum <= candidate *1048576:
            raise RuntimeError("no supported image shape is admitted by the selected configuration")
        raise SWEEP.ArenaTooSmallError((minimum +1048575) //1048576)
    offset=next((i for i in range(plan["effective_image_max_tokens"] -plan["min_image_tokens"] +1)
                 if all(row["budget_shape_ids"][i]>0 for row in plan["ratios"])),None)
    if offset is None:raise RuntimeError("no common image budget is admitted for all listed ratios")
    budget=plan["min_image_tokens"] +offset
    bootstrap=[]
    for row in plan["ratios"]:
        shape=vision_shape_at_budget(plan,row,budget)
        image=args.output /f"arena-{candidate}-bootstrap-{row['ratio'].replace(':','x')}.png"
        image.write_bytes(vision_png(shape["input_width"],shape["input_height"]))
        bootstrap.append({"ratio":row["ratio"],"shape":shape,"image":str(image)})
    config = argparse.Namespace(**vars(args))
    config.image_min_tokens = plan["min_image_tokens"]
    config.image_max_tokens = budget
    config._vision_bootstrap = bootstrap
    result = _runtime_probe(config,candidate)
    result["vision_plan"] = plan
    return result


def vision_shape_at_budget(plan, row, budget):
    identity=row["budget_shape_ids"][budget -plan["min_image_tokens"]]
    shape=dict(next(value for value in row["candidates"] if value["shape_id"]==identity))
    p,q=map(int,row["ratio"].split(":"))
    scale=math.isqrt(budget //(p *q)) +1
    shape.update(image_max_tokens=budget,input_width=plan["alignment"] *p *scale,input_height=plan["alignment"] *q *scale)
    return shape


def vision_runtime_boundary(values, fits):
    if not values:return None
    if fits(values[-1]):return values[-1]
    if len(values)==1 or not fits(values[0]):return None
    lower,upper=0,len(values) -1
    while upper -lower>1:
        middle=(lower +upper) //2
        if fits(values[middle]):lower=middle
        else:upper=middle
    return values[lower]


def vision_common_sequence(args, candidate, plan, budget, directory, probe_mode="preflight"):
    directory.mkdir(parents=True)
    config=argparse.Namespace(**vars(args))
    config.output=directory
    config.image=[]
    config.prompt_file=None
    config.copy_probe=args.copy_probe if probe_mode=="full" else False
    config.probe_mode=probe_mode
    config._vision_capacity_probe=True
    config.image_min_tokens=plan["min_image_tokens"]
    config.image_max_tokens=budget
    config._vision_bootstrap=[]
    for row in plan["ratios"]:
        shape=vision_shape_at_budget(plan,row,budget)
        image=directory /(row["ratio"].replace(":","x") +".png")
        image.write_bytes(vision_png(shape["input_width"],shape["input_height"]))
        config._vision_bootstrap.append({"ratio":row["ratio"],"shape":shape,"image":str(image)})
    receipt=_runtime_probe(config,candidate)
    receipt["record"]=str(directory /"record.json")
    Path(receipt["record"]).write_text(json.dumps(receipt,indent=2) +"\n")
    return receipt


def vision_capacity(args, candidate, plan):
    validate_vision_plan(plan,candidate)
    result = {"schema":"vision-capacity-v2", "controlled_workspace":plan["controlled_workspace"],
              "opaque_runtime_allocations_included":plan["opaque_runtime_allocations_included"],
              "arena_mib":candidate,"family":plan["family"],"alignment":plan["alignment"],"images_per_probe":1,
              "effective_image_max_tokens":plan["effective_image_max_tokens"],"microbatch":plan["microbatch"],"ratios":[],
              "planning_report":{"path":plan.get("artifact"),"sha256":plan.get("sha256")},
              "scope":"Per-source-aspect single-image capacity. The native planner measures every budget; runtime uses a boundary search ordered by processed token count. These are validated fits, not proof that every larger shape fails. Native alignment can change the processed ratio. Independent maxima have row-specific settings; the common command is validated for all listed ratios. Arbitrary ratios and image batches are not validated."}
    directory = args.output /"vision-capacity"
    directory.mkdir()
    cache = {}

    def check(ratio,shape,attempt):
        key=(ratio,shape["image_max_tokens"])
        if key in cache:return cache[key]
        attempt.mkdir(parents=True)
        image=attempt /"input.png"
        image.write_bytes(vision_png(shape["input_width"],shape["input_height"]))
        config=argparse.Namespace(**vars(args))
        config.output=attempt
        config.image=[image]
        config.prompt_file=None
        config.prompt_tokens=None
        config.copy_probe=False
        config.image_min_tokens=plan["min_image_tokens"]
        config.image_max_tokens=shape["image_max_tokens"]
        config._expected_vision_shape=shape
        config._vision_capacity_probe=True
        config._vision_validate_text_first=True
        try:receipt=probe(config,candidate)
        except SWEEP.ArenaTooSmallError as error:
            raise RuntimeError("runtime vision quota contradicts the shape admission plan") from error
        observation={"shape":shape,"receipt":receipt,"record":str(attempt /"record.json"),
                     "image_sha256":hashlib.sha256(image.read_bytes()).hexdigest()}
        (attempt /"record.json").write_text(json.dumps(observation,indent=2) +"\n")
        cache[key]=observation
        return observation
    for ratio in VISION_RATIOS:
        row=next(r for r in plan["ratios"] if r["ratio"]==ratio)
        observations=[]

        def fits(index):
            shape=row["candidates"][index]
            observation=check(ratio,shape,directory /(ratio.replace(":","x") +"-" +str(index)))
            observations.append(observation)
            return observation["receipt"]["fits"]
        selected=vision_runtime_boundary(list(reversed(range(len(row["candidates"])))),fits)
        chosen=None
        if selected is not None:
            shape=row["candidates"][selected]
            observation=cache[(ratio,shape["image_max_tokens"])]
            chosen=dict(shape,validated=True,record=observation["record"],limiting_factor="microbatch" if plan.get("batch_limited") and shape["image_max_tokens"]==plan["effective_image_max_tokens"] else "image_budget_ceiling" if shape["image_max_tokens"]==plan["effective_image_max_tokens"] else "arena_or_runtime_capacity")
        result["ratios"].append({"ratio":ratio,"maximum":chosen,"runtime_attempts":observations,"shapes_measured":row["shapes_measured"]})
        (directory /"results.json").write_text(json.dumps(result,indent=2) +"\n")
    common_results={}

    def common_fits(budget):
        common_rows=[]
        for ratio in VISION_RATIOS:
            row=next(r for r in plan["ratios"] if r["ratio"]==ratio)
            shape=vision_shape_at_budget(plan,row,budget)
            observation=check(ratio,shape,directory /("common-" +str(budget)) /(ratio.replace(":","x")))
            if not observation["receipt"]["fits"]:break
            common_rows.append({"ratio":ratio,"maximum":dict(shape,validated=True,record=observation["record"])})
        if len(common_rows)==len(VISION_RATIOS):
            sequence=vision_common_sequence(args,candidate,plan,budget,directory /("common-sequence-" +str(budget)))
            if not sequence["fits"]:return False
            common_results[budget]={"image_max_tokens":budget,"ratios":common_rows,"validated":True,"sequence":sequence}
            return True
        return False
    budgets=[plan["min_image_tokens"] +i for i in range(plan["effective_image_max_tokens"] -plan["min_image_tokens"] +1)
             if all(row["budget_shape_ids"][i]>0 for row in plan["ratios"])]
    selected=vision_runtime_boundary(budgets,common_fits)
    common=common_results.get(selected)
    result["common"]=common
    result["common_image_max_tokens"]=common["image_max_tokens"] if common else None
    result["all_ratios_validated"]=all(row["maximum"] is not None for row in result["ratios"])
    (directory /"results.json").write_text(json.dumps(result,indent=2) +"\n")
    return result


def format_vision_capacity(capacity):
    rows = {row["ratio"]:row for row in capacity["ratios"]}
    if set(rows)!=set(VISION_RATIOS):raise ValueError("incomplete vision ratio report")
    lines = ["Maximum validated processed resolution (one image):", "  Ratio     Resolution       Image budget setting           Limit"]
    for ratio in VISION_RATIOS:
        maximum = rows[ratio]["maximum"]
        if maximum is None:lines.append(f"  {ratio:<9} unavailable")
        else:lines.append(f"  {ratio:<9} {str(maximum['width']) +'x' +str(maximum['height']):<16} --image-max-tokens {maximum['image_max_tokens']:<6} {maximum.get('limiting_factor','validated')}")
    lines.append("Ratio labels describe the source image; native alignment can change the processed ratio slightly.")
    common=capacity.get("common")
    if common:
        lines.append("Selected command: shared --image-max-tokens " +str(common["image_max_tokens"]) +", validated for all listed ratios.")
        if any(next(row for row in common["ratios"] if row["ratio"]==ratio)["maximum"][key]!=rows[ratio]["maximum"][key] for ratio in VISION_RATIOS for key in ("width","height")):
            lines.append("Processed resolutions with that shared launch setting:")
            for row in common["ratios"]:lines.append("  " +row["ratio"] +": " +str(row["maximum"]["width"]) +"x" +str(row["maximum"]["height"]))
    else:lines.append("No common image setting was validated for all listed ratios; use the individual row settings.")
    return "\n".join(lines)


def probe(args, candidate):
    return _runtime_probe(args,candidate)


def fit(args, report_vision=False):
    report_vision=bool(report_vision and not args.image)
    operation=probe_with_vision if report_vision else probe
    records = []

    def checked(candidate):
        try:
            record = operation(args,candidate)
        except SWEEP.ArenaTooSmallError as error:
            record = {"arena_mib":candidate,"fits":False,"minimum_mib":error.minimum_mib}
            with (args.output /"probes.jsonl").open("a") as output:
                output.write(json.dumps(record) +"\n")
            raise
        records.append(record)
        logged=dict(record)
        if logged.get("vision_plan"):
            plan=logged["vision_plan"]
            logged["vision_plan"]={key:plan.get(key) for key in ("artifact","sha256","family","alignment","effective_image_max_tokens")}
        with (args.output /"probes.jsonl").open("a") as output:
            output.write(json.dumps(logged) +"\n")
        sys.stdout.write(str(json.dumps(logged)) + "\n")
        sys.stdout.flush()
        return record["fits"]
    maximum = SWEEP.find_max_arena_mib(args.initial_arena_mib, checked)
    next_rejected = any(row["arena_mib"] == maximum +1 and not row["fits"] for row in records)
    selected = next((row for row in records if row["arena_mib"] == maximum and row.get("fits")), None)
    runtime_validated = selected is not None and selected.get("timings") is not None
    selected_validation = None
    plan=(selected or {}).get("vision_plan")
    if report_vision and args.mmproj and not plan and (selected or {}).get("vision_capacity_requested"):
        plan=vision_plan_probe(args,maximum)
        if plan is None:raise RuntimeError("selected arena failed its vision planning initialization")
    capacity=vision_capacity(args,maximum,plan) if plan else None
    if capacity and (not capacity["all_ratios_validated"] or capacity["common_image_max_tokens"] is None):
        raise RuntimeError("no common image setting completed all listed ratios; no launch command published; see vision-capacity/results.json")
    command_args=argparse.Namespace(**vars(args))
    if capacity and capacity["common_image_max_tokens"] is not None:
        assert plan is not None
        command_args.image_min_tokens=plan["min_image_tokens"]
        command_args.image_max_tokens=capacity["common_image_max_tokens"]
    if args.validate_selected or (capacity and args.probe_mode == "full"):
        full = argparse.Namespace(**vars(command_args))
        full.probe_mode, full.validate_selected = "full", False
        full.output = args.output /"selected-validation"
        try:
            if capacity:
                selected_validation = vision_common_sequence(full,maximum,plan,capacity["common_image_max_tokens"],full.output,probe_mode="full")
            else:
                full.output.mkdir(exist_ok=False)
                selected_validation = operation(full,maximum)
        except Exception as error:
            full.output.mkdir(parents=True,exist_ok=True)
            (full.output /"record.json").write_text(json.dumps({"arena_mib":maximum,"fits":False,"error":str(error)},indent=2) +"\n")
            raise RuntimeError(f"selected arena full workload validation failed: {error}; see {full.output}") from error
        (full.output /"record.json").write_text(json.dumps(selected_validation,indent=2) +"\n")
        if not selected_validation["fits"]:
            raise RuntimeError(f"selected arena failed full workload validation; no fit published; see {full.output}")
        if capacity and selected_validation.get("command") != server_command(command_args,maximum):
            raise RuntimeError("full workload validation used a different vision launch configuration")
    full_validated = bool(selected_validation and selected_validation["fits"]) or runtime_validated and (args.probe_mode == "full" or bool(args.image))
    report = {"arena_mib":maximum, "initial_arena_mib":args.initial_arena_mib,
              "next_mib_rejected":maximum +1 if next_rejected else None,
              "command":server_command(command_args,maximum), "uvm":False, "reserve_mib":0,
              "context_size":args.context_size, "context_semantics":context_semantics(args),
              "selection_profiles":(selected or {}).get("selection_profiles"),
              "requested_probe_selected_slots":args.probe_selected,
              "max_observed_text_decode_selection":max((count for row in (selected or {}).get("selection_profiles") or []
                                                       for count in row.get("decode_selections", [])), default=None),
              "observed_context_limits":(selected or {}).get("observed_context_limits"),
              "loaded_runtime_libraries":(selected or {}).get("loaded_runtime_libraries"), "cache_type_k":args.cache_type_k, "cache_type_v":args.cache_type_v,
              "spec_mode":args.spec_mode, "draft_length":args.draft_length, "images":[str(path) for path in args.image],
              "ngram":ngram_settings(args), "mixed_routes_validated":args.spec_mode == "dflash-k4v" and bool(selected and selected.get("mixed_proof")),
              "copy_path_validated":bool(selected and selected.get("mixed_proof")), "headless":args.no_webui,
              "model_draft":str(args.model_draft) if args.model_draft else None,
              "prompt_file":str(args.prompt_file) if args.prompt_file else None,
              "projector_configured":bool(args.mmproj), "vision_capacity":capacity,
              "vision_execution_validated":runtime_validated and (bool(args.image) or bool((selected or {}).get("vision_bootstrap"))),
              "probe_mode":args.probe_mode, "runtime_probe_validated":runtime_validated,
              "startup_warmup":not args.no_warmup,
              "full_workload_validated":full_validated, "selected_validation":selected_validation,
              "prompt_tokens":None if args.image else probe_text_tokens(args),
              "requested_prompt_tokens":None if args.image else requested_text_tokens(args),
              "decode_tokens":args.decode_tokens,
              "scope":"Preflight executes bounded text history plus the requested decode, or the complete supplied image request. Full text history is validated only in full mode or by --validate-selected. Different image shapes, prompts, other processes or driver allocations can change fit. Configured slots are distinct from tested selections; --probe-selected does not certify larger selections."}
    removed = sorted(set(REMOVED_ENV) | {name for name in os.environ if name.startswith("LLAMA_ARG_")})
    report["environment"] = {name:None for name in removed}
    report["environment"]["LLAMA_MEDIA_MARKER"] = MARKER
    library_variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    report["environment"][library_variable] = environment(args.server)[library_variable]
    report["runtime_library_environment"] = {library_variable: report["environment"][library_variable]}
    commands = shell_commands(report["command"])
    report["shell"] = "powershell" if os.name == "nt" else "posix"
    report["shell_command"] = commands[report["shell"]]
    report["posix_shell_command"] = commands["posix"]
    report["powershell_command"] = commands["powershell"]
    report["windows_sysmem_fallback_verified"] = False if os.name == "nt" else None
    (args.output /"result.json").write_text(json.dumps(report,indent=2) +"\n", encoding="utf-8")
    (args.output /"command.txt").write_text(report["shell_command"] +"\n", encoding="utf-8")
    (args.output /"command.ps1").write_text(commands["powershell"] +"\n", encoding="utf-8-sig")
    return report


def main(argv=None):
    args = arguments(argv)
    for name in ("server", "model", "model_draft", "mmproj", "prompt_file", "prefill_text"):
        path = getattr(args,name)
        if path is not None:
            path = path.resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"not a file: {path}")
            setattr(args,name,path)
    args.image = [path.resolve(strict=True) for path in args.image]
    if any(not path.is_file() for path in args.image):
        raise ValueError("image paths must be files")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.stdout.write(str("Free the GPU before fitting. This tool does not stop containers. UVM is disabled for every probe.") + "\n")
    sys.stdout.flush()
    if os.name == "nt":
        sys.stdout.write(str("Windows: disabling UVM does not disable driver system-memory fallback. Configure this server's NVIDIA profile to Prefer No Sysmem Fallback before physical-VRAM fitting; allocation success alone does not verify residency.") + "\n")
        sys.stdout.flush()
    report = fit(args,report_vision=bool(args.mmproj and not args.image))
    if report.get("vision_capacity"):
        sys.stdout.write(str("\n" +format_vision_capacity(report["vision_capacity"])) + "\n")
        sys.stdout.flush()
    sys.stdout.write(str("\nSelected command (" +report["shell"] +"):\n" +report["shell_command"]) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
