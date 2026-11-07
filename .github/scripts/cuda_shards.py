"""Compile CUDA objects in bounded hosted jobs and validate them before final linking."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from fork_ci import VARIANTS, digest, read_configuration


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_identity(root):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if os.environ.get("GITHUB_SHA", sha) != sha:
        raise ValueError("checkout does not match the workflow SHA")
    checksum = hashlib.sha256()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for name in sorted(filter(None, names)):
        path = root / name
        value = os.readlink(path) if path.is_symlink() else digest(path)
        checksum.update(name.encode() + b"\0" + value.encode() + b"\0")
    return {"sha": sha, "tree_sha256": checksum.hexdigest()}


def profile(root, build, variant):
    receipt = Path(os.environ["CUDA_PATH"]) / "redistributables.json" if "CUDA_PATH" in os.environ else None
    return {"source": source_identity(root), "variant": variant,
            "workspace": str(root.resolve()), "build": str(build.resolve()),
            "system": platform.system(), "machine": platform.machine(),
            "image": os.environ.get("ImageVersion"), "configuration": read_configuration(build),
            "sdk": json.loads(receipt.read_text()) if receipt else None,
            "environment": {key: os.environ.get(key) for key in ("INCLUDE", "LIB", "LIBPATH", "CUDA_PATH")},
            "helper_sha256": digest(Path(__file__))}


def relative_output(build, output):
    path = Path(output.replace("\\", "/"))
    if path.is_absolute():
        path = path.resolve().relative_to(build.resolve())
    if not path.parts or any(part in ("..", ".") for part in path.parts) or ":" in str(path):
        raise ValueError("invalid object output path")
    if path.suffix not in (".o", ".obj"):
        raise ValueError("expected a compiler object output")
    return path.as_posix()


def compilation_units(build):
    raw = subprocess.check_output(["ninja", "-t", "compdb", "-x"], cwd=build, text=True)
    units = {}
    for entry in json.loads(raw):
        if not entry["file"].endswith(".cu"):
            continue
        output = relative_output(build, entry["output"])
        if output in units:
            raise ValueError("duplicate CUDA compilation output")
        source = Path(entry["file"])
        if not source.is_absolute():
            source = Path(entry["directory"]) / source
        units[output] = {"source": str(source.resolve()),
                         "command_sha256": hashlib.sha256(entry["command"].encode()).hexdigest()}
    if not units:
        raise ValueError("configuration has no CUDA compilation units")
    return units


def partition(units, workers):
    if workers < 1 or workers > 8:
        raise ValueError("expected one to eight concurrent workers per OS")
    heavy, regular = [], []
    for output, entry in sorted(units.items()):
        (heavy if Path(entry["source"]).name.startswith("kv-stream-") else regular).append(output)
    shards = [[output] for output in heavy]
    shards += [regular[i::workers] for i in range(min(workers, len(regular)))]
    if len(shards) > 96:
        raise ValueError("CUDA shard count exceeds the workflow matrix budget")
    return shards


def make_plan(root, build, variant, workers, output):
    units = compilation_units(build)
    plan = {"schema": 1, "deadline": int(time.time()) + 24 * 3600,
            "profile": profile(root, build, variant), "units": units,
            "shards": partition(units, workers)}
    json_write(output, plan)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("matrix=" + json.dumps({"include": [{"shard": i} for i in range(len(plan["shards"]))]}) + "\n")
    sys.stdout.write(f"Planned {len(units)} CUDA objects in {len(plan['shards'])} shards.\n")
    return plan


def remaining(plan):
    seconds = plan["deadline"] - time.time()
    if seconds <= 0:
        raise TimeoutError("24-hour CUDA build budget exhausted; start a new workflow run")
    return seconds


def validate_plan(plan, root, build):
    remaining(plan)
    if plan.get("schema") != 1 or plan["profile"] != profile(root, build, plan["profile"]["variant"]):
        raise ValueError("shard source, toolchain or build configuration changed")
    if plan["units"] != compilation_units(build):
        raise ValueError("CUDA source coverage or compiler commands changed")
    flattened = [name for shard in plan["shards"] for name in shard]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(plan["units"]):
        raise ValueError("invalid CUDA shard coverage")


def dependency_hashes(build, outputs, root):
    allowed = [root.resolve(), build.resolve()]
    if "CUDA_PATH" in os.environ:
        allowed.append(Path(os.environ["CUDA_PATH"]).resolve())
    if sys.platform == "win32":
        allowed += [Path(p).resolve() for p in os.environ.get("INCLUDE", "").split(os.pathsep) if p]
    else:
        allowed.append(Path("/usr"))
    result = {}
    for output in outputs:
        text = subprocess.check_output(["ninja", "-t", "deps", output], cwd=build, text=True)
        if "#deps" not in text or "(VALID)" not in text:
            raise ValueError("missing compiler dependency record: " + output)
        for line in text.splitlines():
            if not line.startswith("    "):
                continue
            path = Path(line[4:])
            if not path.is_absolute():
                path = build / path
            path = path.resolve()
            if not any(path.is_relative_to(parent) for parent in allowed):
                raise ValueError("unexpected compiler dependency root: " + str(path))
            result[str(path)] = digest(path)
    return result


def member_name(output):
    return "objects/" + hashlib.sha256(output.encode()).hexdigest() + Path(output).suffix


def build_shard(plan, index, root, build, output):
    validate_plan(plan, root, build)
    if index < 0 or index >= len(plan["shards"]):
        raise ValueError("invalid shard index")
    outputs = plan["shards"][index]
    subprocess.run(["cmake", "--build", str(build), "--parallel", "1", "--target", *outputs],
                   check=True, timeout=remaining(plan))
    dependencies = dependency_hashes(build, outputs, root)
    manifest = {"schema": 1, "shard": index, "profile": plan["profile"], "dependencies": dependencies, "objects": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in outputs:
            path = build / name
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise ValueError("missing compiled object: " + name)
            row = dict(plan["units"][name], member=member_name(name), sha256=digest(path), size=path.stat().st_size)
            manifest["objects"][name] = row
            archive.write(path, row["member"])
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
    sys.stdout.write(f"Shard {index}: {len(outputs)} objects, {output.stat().st_size} compressed bytes.\n")


def cmake_quote(value):
    if any(c in value for c in (";", "\n", "\r", "]==]")):
        raise ValueError("unsupported CMake object path")
    return "[==[" + value.replace("\\", "/") + "]==]"


def assemble(plan, root, build, archives, destination):
    validate_plan(plan, root, build)
    files = sorted(archives.rglob("objects.zip"))
    if len(files) != len(plan["shards"]):
        raise ValueError("missing CUDA shard archives")
    if destination.exists():
        raise ValueError("refusing to overwrite assembled objects")
    destination.parent.mkdir(parents=True, exist_ok=True)
    objects, seen, dependencies, provenance = {}, set(), {}, []
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        for path in files:
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("manifest.json")
                if info.file_size > 16 * 1024 * 1024:
                    raise ValueError("oversized shard manifest")
                manifest = json.loads(archive.read(info))
                index = manifest.get("shard")
                if type(index) is not int or index < 0 or index >= len(plan["shards"]) or index in seen:
                    raise ValueError("duplicate or invalid CUDA shard")
                if manifest.get("schema") != 1 or manifest.get("profile") != plan["profile"]:
                    raise ValueError("mixed CUDA build cohorts")
                seen.add(index)
                if set(manifest["objects"]) != set(plan["shards"][index]):
                    raise ValueError("shard object coverage changed")
                if not manifest.get("dependencies") or any(
                        plan["units"][n]["source"] not in manifest["dependencies"] for n in plan["shards"][index]):
                    raise ValueError("missing compiler source dependencies")
                expected_members = {"manifest.json"} | {member_name(n) for n in plan["shards"][index]}
                if len(archive.namelist()) != len(expected_members) or set(archive.namelist()) != expected_members:
                    raise ValueError("unexpected CUDA archive members")
                if sum(item.file_size for item in archive.infolist()) > shutil.disk_usage(stage).free:
                    raise ValueError("insufficient disk space for CUDA shard objects")
                for dependency, checksum in manifest["dependencies"].items():
                    if dependency in dependencies and dependencies[dependency] != checksum:
                        raise ValueError("mixed compiler header contents")
                    if not Path(dependency).is_file() or digest(Path(dependency)) != checksum:
                        raise ValueError("compiler dependency changed: " + dependency)
                    dependencies[dependency] = checksum
                for name, row in manifest["objects"].items():
                    if any(row.get(k) != v for k, v in plan["units"][name].items()) or row.get("member") != member_name(name):
                        raise ValueError("object compiler command changed")
                    entry = archive.getinfo(row["member"])
                    if type(row.get("size")) is not int or row["size"] <= 0 or entry.file_size != row["size"]:
                        raise ValueError("invalid CUDA object size")
                    target = stage / Path(row["member"]).name
                    with archive.open(entry) as source, target.open("xb") as out:
                        shutil.copyfileobj(source, out)
                    if digest(target) != row.get("sha256"):
                        raise ValueError("CUDA object checksum changed")
                    objects[name] = {"source": row["source"], "file": target.name, "sha256": row["sha256"]}
                provenance.append({"shard": index, "archive_sha256": digest(path), "objects": len(manifest["objects"])})
        if set(objects) != set(plan["units"]):
            raise ValueError("incomplete CUDA object coverage")
        stage.rename(destination)
    sources = "\n".join(cmake_quote(objects[n]["source"]) for n in sorted(objects))
    binaries = "\n".join(cmake_quote(str((destination / objects[n]["file"]).resolve())) for n in sorted(objects))
    manifest_path = destination / "objects.cmake"
    manifest_path.write_text("set(GGML_CUDA_CI_SOURCES\n" + sources + "\n)\nset(GGML_CUDA_CI_OBJECTS\n" + binaries + "\n)\n")
    compressed_bytes = sum(path.stat().st_size for path in files)
    json_write(build / "cuda-shards.json", {"profile": plan["profile"], "shards": provenance, "objects": objects,
                                            "dependencies": dependencies, "object_root": str(destination.resolve()),
                                            "compressed_object_bytes": compressed_bytes})
    subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                    "-DGGML_CUDA_CI_OBJECTS_FILE=" + str(manifest_path.resolve())], check=True, timeout=remaining(plan))
    sys.stdout.write(f"Validated all {len(objects)} CUDA objects ({compressed_bytes} archive bytes) for final linking.\n")


def finish(plan, root, build):
    remaining(plan)
    if plan["profile"] != profile(root, build, plan["profile"]["variant"]):
        raise ValueError("final link configuration changed")
    metadata = json.loads((build / "cuda-shards.json").read_text())
    if set(metadata["objects"]) != set(plan["units"]):
        raise ValueError("incomplete final object coverage")
    for name, row in metadata["objects"].items():
        expected = Path(member_name(name)).name
        path = Path(metadata["object_root"]) / expected
        if row["file"] != expected or path.is_symlink() or not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError("assembled CUDA object changed")
    for dependency, checksum in metadata["dependencies"].items():
        if not Path(dependency).is_file() or digest(Path(dependency)) != checksum:
            raise ValueError("compiler dependency changed before final link: " + dependency)
    commands = json.loads(subprocess.check_output(["ninja", "-t", "compdb", "-x"], cwd=build, text=True))
    if any(row["file"].endswith(".cu") for row in commands):
        raise ValueError("final job would recompile shader sources")
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True, timeout=remaining(plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "build", "assemble", "finish"))
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--plan", type=Path, default=Path(".ci-cuda/plan.json"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    build = args.build.resolve()
    if args.action == "plan":
        if args.variant is None:
            parser.error("plan requires --variant")
        make_plan(root, build, args.variant, args.workers, args.plan)
    else:
        plan = json.loads(args.plan.read_text())
        if args.action == "finish":
            finish(plan, root, build)
            return
        if args.output is None:
            parser.error("--output is required")
        if args.action == "build":
            if args.shard is None:
                parser.error("build requires --shard")
            build_shard(plan, args.shard, root, build, args.output)
        else:
            assemble(plan, root, build, Path(".ci-cuda/archives"), args.output)


if __name__ == "__main__":
    main()
