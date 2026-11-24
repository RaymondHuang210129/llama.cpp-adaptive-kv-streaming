"""Fork CI policy, binary packaging and immutable SHA release validation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


VARIANTS = ("linux-x64-cuda", "windows-x64-cuda")
BUILD_VARIANTS = ("linux-x64-cpu", "windows-x64-cpu", *VARIANTS)
RELEASE_BRANCH = "feature/adaptive-kv-stream-v3"
CUDA_PROFILES = {
    "rtx50": {"architectures": "120a", "toolkit": "13.0.1", "cccl": "v3.4.3", "day": None},
    "rtx40": {"architectures": "89-real", "toolkit": "13.0.1", "cccl": "v3.4.3", "day": 1},
    "rtx30": {"architectures": "86-real", "toolkit": "13.0.1", "cccl": "v3.4.3", "day": 2},
    "rtx20": {"architectures": "75-real", "toolkit": "13.0.1", "cccl": "v3.4.3", "day": 3},
    "v100": {"architectures": "70-real", "toolkit": "12.9.1", "cccl": "v2.8.2", "day": 4},
    "p100": {"architectures": "60-real", "toolkit": "12.9.1", "cccl": "v2.8.2", "day": 5},
    "pascal": {"architectures": "61-real", "toolkit": "12.9.1", "cccl": "v2.8.2", "day": 6},
}
BASE_JOBS = {"changes", "cpu", "cuda-linux", "cuda-windows", "python", "source", "cmake-package", "ui-build"}


def classify_changes(paths, full=False):
    flags = {key: full for key in ("server", "sanitize", "ui", "vendor", "tokenizer", "ops", "requirements")}
    for path in paths:
        if path.startswith(".github/") or path == "CMakeLists.txt" or path.startswith("cmake/"):
            return dict.fromkeys(flags, True)
        if path.startswith(("src/", "include/", "common/", "ggml/", "tests/", "tools/server/", "tools/mtmd/")):
            flags["server"] = flags["sanitize"] = True
        flags["ui"] |= path.startswith("tools/ui/")
        flags["vendor"] |= path.startswith("vendor/") or path == "scripts/sync_vendor.py"
        flags["tokenizer"] |= path in ("conversion/base.py", "convert_hf_to_gguf_update.py")
        flags["ops"] |= path == "docs/ops.md" or path.startswith("docs/ops/") or path == "scripts/create_ops_docs.py"
        flags["requirements"] |= bool(re.search(r"(^|/)requirements[^/]*\.txt$", path)) or bool(
            re.fullmatch(r"convert[^/]*\.py", path)) or path == "scripts/check-requirements.sh"
    return flags


def validate_results(results, expected):
    bad = {name: results.get(name, {}).get("result", "missing") for name in expected
           if results.get(name, {}).get("result") != "success"}
    bad.update({name: row.get("result") for name, row in results.items()
                if row.get("result") in ("failure", "cancelled")})
    if bad:
        raise ValueError("CI checks did not succeed: " + json.dumps(bad, sort_keys=True))


def cuda_profile(name):
    if name not in CUDA_PROFILES:
        raise ValueError("unknown CUDA architecture profile")
    return dict(CUDA_PROFILES[name])


def scheduled_profile(schedule):
    for name, profile in CUDA_PROFILES.items():
        if profile["day"] is not None and schedule == f"17 3 * * {profile['day']}":
            return name
    raise ValueError("schedule has no assigned architecture profile")


def can_publish(event, ref, result, profile="rtx50"):
    if ref != "refs/heads/" + RELEASE_BRANCH or result != "success" or profile not in CUDA_PROFILES:
        return False
    return event == "push" if profile == "rtx50" else event in ("schedule", "workflow_dispatch")


def validate_sha(sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("a full lowercase commit SHA is required")


def release_tag(sha, profile=None):
    validate_sha(sha)
    if profile is not None:
        cuda_profile(profile)
    return "sha-" + sha if profile is None else "v3-" + profile + "-" + sha


def archive_name(sha, variant, profile=None):
    validate_sha(sha)
    if variant not in BUILD_VARIANTS:
        raise ValueError("unsupported package variant")
    if profile is not None:
        cuda_profile(profile)
    prefix = f"llama-kv-stream-{sha}" if profile is None else f"oceanus-v3-{sha}-{profile}"
    return f"{prefix}-{variant}" + (".zip" if variant.startswith("windows") else ".tar.gz")


def digest(path):
    with path.open("rb") as source:
        checksum = hashlib.sha256()
        while chunk := source.read(1024 *1024):
            checksum.update(chunk)
        return checksum.hexdigest()


def read_configuration(build):
    keys = {"BUILD_SHARED_LIBS", "CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
            "CMAKE_CUDA_COMPILER", "CMAKE_LINKER", "CMAKE_CUDA_ARCHITECTURES", "GGML_CUDA", "GGML_BACKEND_DL",
            "GGML_NATIVE", "GGML_CUDA_FA_QUANTS", "GGML_CUDA_CCCL_VERSION", "LLAMA_FATAL_WARNINGS", "GGML_FATAL_WARNINGS"}
    result = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.fullmatch(r"([^:=]+):[^=]*=(.*)", line)
        if match and match[1] in keys:
            result[match[1]] = match[2]
    for path in (build / "CMakeFiles").glob("*/CMake*Compiler.cmake"):
        for key, value in re.findall(r'set\((CMAKE_\w+_COMPILER_(?:ID|VERSION)) "([^"]+)"\)', path.read_text()):
            result[key] = value
    return result


def validate_profile_build(build, sha, variant, profile):
    expected = cuda_profile(profile)
    configuration = build.get("configuration", {})
    sdk = build.get("cuda_redistributables", {})
    shards = build.get("cuda_shards", {}).get("profile", {})
    platform = "windows-x86_64" if variant.startswith("windows") else "linux-x86_64"
    if (build.get("sha") != sha or build.get("variant") != variant or build.get("profile") != profile
            or configuration.get("CMAKE_CUDA_ARCHITECTURES") != expected["architectures"]
            or configuration.get("GGML_CUDA_CCCL_VERSION") != expected["cccl"]
            or sdk.get("toolkit") != expected["toolkit"] or sdk.get("platform") != platform
            or shards.get("source", {}).get("sha") != sha or shards.get("variant") != variant
            or shards.get("sdk") != sdk or shards.get("configuration") != configuration):
        raise ValueError("CUDA package profile, toolchain or source provenance does not match")


def package(build, output, sha, variant, cuda_root=None, profile=None):
    filename = archive_name(sha, variant, profile)
    windows = variant.startswith("windows")
    binaries = [name + (".exe" if windows else "") for name in ("llama-server", "llama-cli", "llama-bench")]
    source_dir = build / "bin"
    if any(not (source_dir / name).is_file() for name in binaries):
        raise ValueError("package is missing a required executable")
    files = {path.name: path for path in source_dir.iterdir()
             if path.name in binaries or path.name.endswith(".dll") or ".so" in path.name}
    if cuda_root:
        patterns = (["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"] if windows else
                    ["libcudart.so*", "libcublas.so*", "libcublasLt.so*"])
        for pattern in patterns:
            for path in cuda_root.rglob(pattern):
                if path.is_file() and "stubs" not in path.parts:
                    files[path.name] = path
    if not any("llama" in name and name not in binaries for name in files):
        raise ValueError("package is missing shared libraries")
    if variant.endswith("cuda"):
        for prefix in (["ggml-cuda", "cudart64_", "cublas64_", "cublasLt64_"] if windows else
                       ["libggml-cuda", "libcudart.so", "libcublas.so", "libcublasLt.so"]):
            if not any(name.startswith(prefix) for name in files):
                raise ValueError("CUDA package is missing " + prefix)
    output.mkdir(parents=True, exist_ok=True)
    target = output / filename
    if target.exists() or (output / (variant + ".json")).exists():
        raise ValueError("refusing to overwrite an existing package")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary) / filename.removesuffix(".tar.gz").removesuffix(".zip")
        stage.mkdir()
        allowed = [source_dir.resolve()] + ([cuda_root.resolve()] if cuda_root else [])
        for name, source in files.items():
            if not any(source.resolve().is_relative_to(root) for root in allowed):
                raise ValueError("package source escapes the build/toolkit directory")
            if source.is_symlink():
                resolved = source.resolve(strict=True)
                shutil.copy2(resolved, stage / resolved.name)
                (stage / name).symlink_to(resolved.name)
            else:
                shutil.copy2(source, stage / name)
        if windows and sys.platform == "win32":
            redist = os.environ.get("VCToolsRedistDir")
            if not redist:
                raise ValueError("MSVC redistributable directory is unavailable")
            crt = list((Path(redist) / "x64").glob("Microsoft.VC*.CRT/*.dll"))
            if not crt:
                raise ValueError("MSVC redistributable runtime is missing")
            for source in crt + list((Path(redist) / "x64").glob("Microsoft.VC*.OpenMP/*.dll")):
                shutil.copy2(source, stage / source.name)
        repo = Path(__file__).resolve().parents[2]
        for name in ("LICENSE", "AUTHORS"):
            if (repo / name).is_file():
                shutil.copy2(repo / name, stage / name)
        for root in (repo / "vendor", build / "_deps"):
            for path in root.rglob("LICENSE*"):
                if path.is_file():
                    license_path = stage / "licenses" / root.name / path.relative_to(root)
                    license_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, license_path)
        if cuda_root and (cuda_root / "licenses").is_dir():
            shutil.copytree(cuda_root / "licenses", stage / "licenses", dirs_exist_ok=True)
        metadata = {"sha": sha, "variant": variant, "configuration": read_configuration(build),
                    "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                    "files": {name: digest(path) for name, path in sorted(files.items())}}
        if cuda_root:
            metadata["cuda_redistributables"] = json.loads((cuda_root / "redistributables.json").read_text())
        if (build / "cuda-shards.json").is_file():
            metadata["cuda_shards"] = json.loads((build / "cuda-shards.json").read_text())
        if profile is not None:
            metadata["profile"] = profile
            validate_profile_build(metadata, sha, variant, profile)
        (stage / "build-info.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if windows:
            with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(stage.parent))
        else:
            with tarfile.open(target, "x:gz") as archive:
                archive.add(stage, arcname=stage.name)
    row = {"sha": sha, "variant": variant, "file": filename, "size": target.stat().st_size,
           "sha256": digest(target), "build": metadata}
    if profile is not None:
        row["profile"] = profile
    (output / (variant + ".json")).write_text(json.dumps(row, indent=2) + "\n")
    return row


def verify_packages(directory, sha, profile=None):
    validate_sha(sha)
    packages = {}
    for variant in VARIANTS:
        path = directory / (variant + ".json")
        if not path.is_file():
            raise ValueError("missing package manifest: " + variant)
        row = json.loads(path.read_text())
        if row.get("sha") != sha or row.get("variant") != variant or row.get("file") != archive_name(sha, variant, profile):
            raise ValueError("package identity does not match the source commit")
        if profile is not None:
            if row.get("profile") != profile:
                raise ValueError("package architecture profile does not match")
            validate_profile_build(row.get("build", {}), sha, variant, profile)
        archive = directory / row["file"]
        if (not archive.is_file() or archive.is_symlink() or archive.stat().st_size <= 0
                or archive.stat().st_size != row.get("size") or digest(archive) != row.get("sha256")):
            raise ValueError("package checksum or size does not match: " + variant)
        packages[variant] = row
    result = {"sha": sha, "packages": packages}
    if profile is not None:
        result["profile"] = profile
    return result


def write_release_metadata(directory, sha, profile=None):
    manifest = verify_packages(directory, sha, profile)
    (directory / "build-info.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "SHA256SUMS").write_text("".join(
        row["sha256"] + "  " + row["file"] + "\n" for row in manifest["packages"].values()))
    return manifest


def release_assets(directory, sha, profile=None):
    return [directory / archive_name(sha, variant, profile) for variant in VARIANTS] + [
        directory / "build-info.json", directory / "SHA256SUMS"]


def validate_release(release, directory, sha, profile=None):
    if release.get("tag_name") != release_tag(sha, profile) or release.get("target_commitish") != sha:
        raise ValueError("existing release points at another commit")
    assets = {row["name"]: row for row in release.get("assets", [])}
    for path in release_assets(directory, sha, profile):
        asset = assets.get(path.name, {})
        if (asset.get("state") != "uploaded" or asset.get("size") != path.stat().st_size
                or asset.get("digest") != "sha256:" + digest(path)):
            raise ValueError("existing release is incomplete or has different bytes: " + path.name)


def validate_published_release(release, manifest_bytes, sha, profile=None):
    if release.get("tag_name") != release_tag(sha, profile) or release.get("target_commitish") != sha:
        raise ValueError("published release points at another commit")
    manifest = json.loads(manifest_bytes)
    packages = manifest.get("packages", {})
    if manifest.get("sha") != sha or set(packages) != set(VARIANTS):
        raise ValueError("published release has invalid provenance")
    if profile is not None and manifest.get("profile") != profile:
        raise ValueError("published release has a different architecture profile")
    expected = {}
    for variant, row in packages.items():
        if row.get("sha") != sha or row.get("variant") != variant or row.get("file") != archive_name(sha, variant, profile):
            raise ValueError("published package identity is invalid")
        if profile is not None:
            if row.get("profile") != profile:
                raise ValueError("published package has a different architecture profile")
            validate_profile_build(row.get("build", {}), sha, variant, profile)
        expected[row["file"]] = (row["size"], row["sha256"])
    sums = "".join(packages[v]["sha256"] + "  " + packages[v]["file"] + "\n" for v in VARIANTS).encode()
    expected["build-info.json"] = (len(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest())
    expected["SHA256SUMS"] = (len(sums), hashlib.sha256(sums).hexdigest())
    assets = {row["name"]: row for row in release.get("assets", [])}
    for name, (size, checksum) in expected.items():
        asset = assets.get(name, {})
        if asset.get("state") != "uploaded" or asset.get("size") != size or asset.get("digest") != "sha256:" + checksum:
            raise ValueError("published release is incomplete or inconsistent: " + name)


class AssetRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and urllib.parse.urlsplit(result.full_url).hostname != "api.github.com":
            result.remove_header("Authorization")
        return result


class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not token:
            raise ValueError("a GitHub repository and token are required")
        self.repository = repository
        self.token = token

    def request(self, method, path, payload=None, upload=None):
        base = "https://uploads.github.com" if upload else "https://api.github.com"
        data = upload.read_bytes() if upload else json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(base + "/repos/" + self.repository + path, data=data, method=method,
                                         headers={"Authorization": "Bearer " + self.token,
                                                  "Accept": "application/vnd.github+json",
                                                  "Content-Type": "application/octet-stream" if upload else "application/json",
                                                  "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as error:
            if error.code == 404 and method == "GET":
                return None
            raise RuntimeError(f"GitHub {method} failed with HTTP {error.code}") from error

    def tag_commit(self, sha, profile=None):
        ref = self.request("GET", "/git/ref/tags/" + release_tag(sha, profile))
        if ref is None:
            return None
        obj = ref["object"]
        for _ in range(5):
            if obj["type"] == "commit":
                return obj["sha"]
            if obj["type"] != "tag":
                break
            validate_sha(obj["sha"])
            obj = self.request("GET", "/git/tags/" + obj["sha"])["object"]
        raise ValueError("release tag does not resolve to a commit")

    def download_manifest(self, asset_id):
        request = urllib.request.Request("https://api.github.com/repos/" + self.repository
                                         + "/releases/assets/" + str(int(asset_id)),
                                         headers={"Authorization": "Bearer " + self.token,
                                                  "Accept": "application/octet-stream"})
        with urllib.request.build_opener(AssetRedirect()).open(request, timeout=120) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("release manifest exceeds the size limit")
        return data


def publish(directory, sha, github, run_id, profile=None):
    write_release_metadata(directory, sha, profile)
    tag = release_tag(sha, profile)
    release = github.request("GET", "/releases/tags/" + tag)
    owner = "<!-- ci-run:" + run_id + " -->"
    tag_commit = github.tag_commit(sha, profile)
    if tag_commit is not None and tag_commit != sha:
        raise ValueError("immutable release tag points at another commit")
    if release and not release["draft"]:
        if tag_commit != sha:
            raise ValueError("published release tag is missing")
        manifest = next((row for row in release.get("assets", []) if row["name"] == "build-info.json"), None)
        if manifest is None:
            raise ValueError("published release has no provenance manifest")
        validate_published_release(release, github.download_manifest(manifest["id"]), sha, profile)
        return release
    if release:
        if owner not in release.get("body", "") or release.get("target_commitish") != sha:
            raise ValueError("refusing to change a draft release owned by another run")
    if tag_commit is None:
        github.request("POST", "/git/refs", {"ref": "refs/tags/" + tag, "sha": sha})
    if not release:
        release = github.request("POST", "/releases", {"tag_name": tag, "target_commitish": sha,
                                 "name": "Oceanus V3 " + (profile or "legacy") + " " + sha[:12], "draft": True, "prerelease": True,
                                                       "body": f"Continuous Oceanus V3 build of `{sha}` ({profile or 'legacy'}). See build-info.json and SHA256SUMS.\n\n{owner}"})
    release_id = release["id"]
    current = {row["name"]: row for row in release.get("assets", [])}
    for path in release_assets(directory, sha, profile):
        asset = current.get(path.name)
        if asset:
            if asset.get("state") == "uploaded" and asset.get("size") == path.stat().st_size and asset.get("digest") == "sha256:" + digest(path):
                continue
            # Only this run's unpublished draft is mutable during recovery.
            github.request("DELETE", "/releases/assets/" + str(asset["id"]))
        github.request("POST", f"/releases/{release_id}/assets?name=" + urllib.parse.quote(path.name), upload=path)
    release = github.request("GET", "/releases/" + str(release_id))
    validate_release(release, directory, sha, profile)
    return github.request("PATCH", "/releases/" + str(release_id), {"draft": False, "prerelease": True, "make_latest": "false"})


def successful_source_run(run, sha):
    return (run.get("head_sha") == sha and run.get("head_branch") == RELEASE_BRANCH
            and run.get("event") == "push" and run.get("status") == "completed"
            and run.get("conclusion") == "success" and run.get("path") == ".github/workflows/ci.yml")


def resolve_compat_source(github, requested=None):
    if requested:
        validate_sha(requested)
        candidates = [(requested, None)]
    else:
        query = urllib.parse.urlencode({"branch": RELEASE_BRANCH, "event": "push", "status": "success", "per_page": 100})
        runs = github.request("GET", "/actions/workflows/ci.yml/runs?" + query)
        candidates = [(row.get("head_sha"), row) for row in runs.get("workflow_runs", [])
                      if successful_source_run(row, row.get("head_sha"))]
    for sha, run in candidates:
        validate_sha(sha)
        release = github.request("GET", "/releases/tags/" + release_tag(sha, "rtx50"))
        if release is None and not requested:
            continue
        if not release or release.get("draft") or github.tag_commit(sha, "rtx50") != sha:
            raise ValueError("compatibility source needs a published RTX 50 release")
        asset = next((a for a in release.get("assets", []) if a["name"] == "build-info.json"), None)
        if not asset:
            raise ValueError("source release has no provenance manifest")
        raw = github.download_manifest(asset["id"])
        validate_published_release(release, raw, sha, "rtx50")
        manifest = json.loads(raw)
        run_ids = {str(row["build"].get("run_id", "")) for row in manifest["packages"].values()}
        if len(run_ids) != 1 or not next(iter(run_ids)).isdigit():
            raise ValueError("source OS packages do not belong to one CI run")
        run_id = next(iter(run_ids))
        if run is None or str(run.get("id")) != run_id:
            run = github.request("GET", "/actions/runs/" + run_id)
        if not run or not successful_source_run(run, sha):
            raise ValueError("source release does not come from successful V3 push CI")
        return sha
    raise ValueError("no successful V3 push with a published RTX 50 release was found")


def write_outputs(values):
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")
    sys.stdout.write(json.dumps(values) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("changes", "required", "package", "publish", "profile", "resolve-compat"))
    parser.add_argument("--build", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=BUILD_VARIANTS)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--profile", choices=tuple(CUDA_PROFILES))
    parser.add_argument("--source-sha")
    args = parser.parse_args()
    sha = os.environ.get("OCEANUS_SOURCE_SHA") or os.environ.get("GITHUB_SHA", "")
    profile = args.profile or os.environ.get("OCEANUS_CUDA_PROFILE", "rtx50")
    if args.action == "profile":
        configuration = cuda_profile(profile)
        write_outputs(dict(profile=profile, toolkit=configuration["toolkit"],
                           architectures=configuration["architectures"], cccl=configuration["cccl"]))
    elif args.action == "resolve-compat":
        event = os.environ.get("GITHUB_EVENT_NAME")
        if event == "schedule":
            payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            profile = scheduled_profile(payload["schedule"])
        if not can_publish(event, os.environ.get("GITHUB_REF"), "success", profile) or profile == "rtx50":
            parser.error("compatibility builds require a scheduled/manual V3 architecture profile")
        github = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
        sha = resolve_compat_source(github, args.source_sha)
        write_outputs({"sha": sha, "profile": profile})
    elif args.action == "changes":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        base = event.get("pull_request", {}).get("base", {}).get("sha") or event.get("before")
        if base and re.fullmatch(r"[0-9a-f]{40}", base) and base != "0" * 40:
            paths = subprocess.check_output(["git", "diff", "--name-only", base, sha], text=True).splitlines()
        else:
            paths = []
        flags = classify_changes(paths, full=not base or base == "0" * 40 or os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch")
        flags["draft"] = bool(event.get("pull_request", {}).get("draft", False))
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for name, value in flags.items():
                output.write(f"{name}={str(value).lower()}\n")
        sys.stdout.write(json.dumps(flags) + "\n")
    elif args.action == "required":
        flags = json.loads(os.environ["CI_FLAGS"])
        expected = {"changes", "python", "source"} if flags.get("draft") == "true" else BASE_JOBS | {
            key for key in ("server", "sanitize", "ui") if flags.get(key) == "true"}
        validate_results(json.loads(os.environ["CI_RESULTS"]), expected)
    elif args.action == "package":
        if args.build is None or args.output is None or args.variant is None:
            parser.error("package requires --build, --output and --variant")
        validate_sha(sha)
        if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != sha:
            parser.error("package checkout does not match the selected source SHA")
        package(args.build, args.output, sha, args.variant, args.cuda_root, profile=profile if args.variant in VARIANTS else None)
    else:
        if args.output is None or not can_publish(os.environ.get("GITHUB_EVENT_NAME"), os.environ.get("GITHUB_REF"),
                                                  os.environ.get("CI_RESULT"), profile):
            parser.error("publication requires successful trusted V3 CI and --output")
        github = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
        if profile != "rtx50":
            resolve_compat_source(github, sha)
        publish(args.output, sha, github, os.environ["GITHUB_RUN_ID"], profile=profile)


if __name__ == "__main__":
    main()
