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


def can_publish(event, ref, result):
    return event == "push" and ref == "refs/heads/feature/adaptive-kv-stream-v2" and result == "success"


def validate_sha(sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("a full lowercase commit SHA is required")


def archive_name(sha, variant):
    validate_sha(sha)
    if variant not in BUILD_VARIANTS:
        raise ValueError("unsupported package variant")
    return f"llama-kv-stream-{sha}-{variant}" + (".zip" if variant.startswith("windows") else ".tar.gz")


def digest(path):
    with path.open("rb") as source:
        checksum = hashlib.sha256()
        while chunk := source.read(1024 *1024):
            checksum.update(chunk)
        return checksum.hexdigest()


def read_configuration(build):
    keys = {"BUILD_SHARED_LIBS", "CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
            "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_ARCHITECTURES", "GGML_CUDA", "GGML_BACKEND_DL",
            "GGML_NATIVE", "GGML_CUDA_FA_QUANTS", "LLAMA_FATAL_WARNINGS", "GGML_FATAL_WARNINGS"}
    result = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.fullmatch(r"([^:=]+):[^=]*=(.*)", line)
        if match and match[1] in keys:
            result[match[1]] = match[2]
    for path in (build / "CMakeFiles").glob("*/CMake*Compiler.cmake"):
        for key, value in re.findall(r'set\((CMAKE_\w+_COMPILER_(?:ID|VERSION)) "([^"]+)"\)', path.read_text()):
            result[key] = value
    return result


def package(build, output, sha, variant, cuda_root=None):
    filename = archive_name(sha, variant)
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
            shutil.copytree(cuda_root / "licenses", stage / "licenses")
        metadata = {"sha": sha, "variant": variant, "configuration": read_configuration(build),
                    "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                    "files": {name: digest(path) for name, path in sorted(files.items())}}
        if cuda_root:
            metadata["cuda_redistributables"] = json.loads((cuda_root / "redistributables.json").read_text())
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
    (output / (variant + ".json")).write_text(json.dumps(row, indent=2) + "\n")
    return row


def verify_packages(directory, sha):
    validate_sha(sha)
    packages = {}
    for variant in VARIANTS:
        path = directory / (variant + ".json")
        if not path.is_file():
            raise ValueError("missing package manifest: " + variant)
        row = json.loads(path.read_text())
        if row.get("sha") != sha or row.get("variant") != variant or row.get("file") != archive_name(sha, variant):
            raise ValueError("package identity does not match the source commit")
        archive = directory / row["file"]
        if (not archive.is_file() or archive.is_symlink() or archive.stat().st_size <= 0
                or archive.stat().st_size != row.get("size") or digest(archive) != row.get("sha256")):
            raise ValueError("package checksum or size does not match: " + variant)
        packages[variant] = row
    return {"sha": sha, "packages": packages}


def write_release_metadata(directory, sha):
    manifest = verify_packages(directory, sha)
    (directory / "build-info.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "SHA256SUMS").write_text("".join(
        row["sha256"] + "  " + row["file"] + "\n" for row in manifest["packages"].values()))
    return manifest


def release_assets(directory, sha):
    return [directory / archive_name(sha, variant) for variant in VARIANTS] + [
        directory / "build-info.json", directory / "SHA256SUMS"]


def validate_release(release, directory, sha):
    if release.get("tag_name") != "sha-" + sha or release.get("target_commitish") != sha:
        raise ValueError("existing release points at another commit")
    assets = {row["name"]: row for row in release.get("assets", [])}
    for path in release_assets(directory, sha):
        asset = assets.get(path.name, {})
        if (asset.get("state") != "uploaded" or asset.get("size") != path.stat().st_size
                or asset.get("digest") != "sha256:" + digest(path)):
            raise ValueError("existing release is incomplete or has different bytes: " + path.name)


def validate_published_release(release, manifest_bytes, sha):
    if release.get("tag_name") != "sha-" + sha or release.get("target_commitish") != sha:
        raise ValueError("published release points at another commit")
    manifest = json.loads(manifest_bytes)
    packages = manifest.get("packages", {})
    if manifest.get("sha") != sha or set(packages) != set(VARIANTS):
        raise ValueError("published release has invalid provenance")
    expected = {}
    for variant, row in packages.items():
        if row.get("sha") != sha or row.get("variant") != variant or row.get("file") != archive_name(sha, variant):
            raise ValueError("published package identity is invalid")
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

    def tag_commit(self, sha):
        ref = self.request("GET", "/git/ref/tags/sha-" + sha)
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


def publish(directory, sha, github, run_id):
    write_release_metadata(directory, sha)
    release = github.request("GET", "/releases/tags/sha-" + sha)
    owner = "<!-- ci-run:" + run_id + " -->"
    tag_commit = github.tag_commit(sha)
    if tag_commit is not None and tag_commit != sha:
        raise ValueError("immutable release tag points at another commit")
    if release and not release["draft"]:
        if tag_commit != sha:
            raise ValueError("published release tag is missing")
        manifest = next((row for row in release.get("assets", []) if row["name"] == "build-info.json"), None)
        if manifest is None:
            raise ValueError("published release has no provenance manifest")
        validate_published_release(release, github.download_manifest(manifest["id"]), sha)
        return release
    if release:
        if owner not in release.get("body", "") or release.get("target_commitish") != sha:
            raise ValueError("refusing to change a draft release owned by another run")
    if tag_commit is None:
        github.request("POST", "/git/refs", {"ref": "refs/tags/sha-" + sha, "sha": sha})
    if not release:
        release = github.request("POST", "/releases", {"tag_name": "sha-" + sha, "target_commitish": sha,
                                 "name": "Adaptive KV Streaming " + sha[:12], "draft": True, "prerelease": True,
                                                       "body": f"Continuous v2 build of `{sha}`. See build-info.json and SHA256SUMS.\n\n{owner}"})
    release_id = release["id"]
    current = {row["name"]: row for row in release.get("assets", [])}
    for path in release_assets(directory, sha):
        asset = current.get(path.name)
        if asset:
            if asset.get("state") == "uploaded" and asset.get("size") == path.stat().st_size and asset.get("digest") == "sha256:" + digest(path):
                continue
            # Only this run's unpublished draft is mutable during recovery.
            github.request("DELETE", "/releases/assets/" + str(asset["id"]))
        github.request("POST", f"/releases/{release_id}/assets?name=" + urllib.parse.quote(path.name), upload=path)
    release = github.request("GET", "/releases/" + str(release_id))
    validate_release(release, directory, sha)
    return github.request("PATCH", "/releases/" + str(release_id), {"draft": False, "prerelease": True, "make_latest": "false"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("changes", "required", "package", "publish"))
    parser.add_argument("--build", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=BUILD_VARIANTS)
    parser.add_argument("--cuda-root", type=Path)
    args = parser.parse_args()
    sha = os.environ.get("GITHUB_SHA", "")
    if args.action == "changes":
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
        package(args.build, args.output, sha, args.variant, args.cuda_root)
    else:
        if args.output is None or not can_publish(os.environ.get("GITHUB_EVENT_NAME"), os.environ.get("GITHUB_REF"),
                                                  os.environ.get("CI_RESULT")):
            parser.error("publication requires a successful trusted v2 push and --output")
        github = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
        publish(args.output, sha, github, os.environ["GITHUB_RUN_ID"])


if __name__ == "__main__":
    main()
