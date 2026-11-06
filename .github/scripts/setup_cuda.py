"""Install checksum-pinned NVIDIA redistributables without a GPU or driver install."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

from fork_ci import digest


VERSION = "13.0.1"
MANIFESTS = {
    "13.0.1": "9c494bc13b34e8fbcad083a6486d185b0906068b821722502edf9d0e3bd14096",
    "12.9.1": "8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630",
}
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ("cuda_cccl", "cuda_crt", "cuda_cudart", "cuda_nvcc", "libcublas", "libnvvm")


def install(destination, version=VERSION):
    manifest_sha256 = MANIFESTS[version]
    components = COMPONENTS if version == "13.0.1" else ("cuda_cccl", "cuda_cudart", "cuda_nvcc", "libcublas")
    platform = "windows-x86_64" if sys.platform == "win32" else "linux-x86_64"
    receipt = destination / "redistributables.json"
    if receipt.exists():
        data = json.loads(receipt.read_text())
        if data.get("manifest_sha256") != manifest_sha256 or data.get("platform") != platform:
            raise ValueError("cached CUDA toolkit has a different identity")
        if not (destination / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")).is_file():
            raise ValueError("cached CUDA toolkit is incomplete")
        return
    destination.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(BASE + "redistrib_" + version + ".json", timeout=120).read()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("NVIDIA redistributable manifest checksum changed")
    manifest = json.loads(raw)
    installed = {}
    for name in components:
        entry = manifest[name][platform]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / Path(entry["relative_path"]).name
            with urllib.request.urlopen(BASE + entry["relative_path"], timeout=600) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
            if digest(archive) != entry["sha256"]:
                raise ValueError("CUDA component checksum mismatch: " + name)
            unpacked = root / "unpacked"
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as payload:
                    for member in payload.namelist():
                        if not (unpacked / member).resolve().is_relative_to(unpacked.resolve()):
                            raise ValueError("CUDA archive path escapes the extraction directory")
                    payload.extractall(unpacked)
            else:
                with tarfile.open(archive) as payload:
                    payload.extractall(unpacked, filter="data")
            roots = list(unpacked.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("unexpected CUDA redistributable layout")
            for child in roots[0].iterdir():
                if child.is_dir():
                    shutil.copytree(child, destination / child.name, dirs_exist_ok=True, symlinks=True)
                elif "license" in child.name.lower():
                    (destination / "licenses" / name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, destination / "licenses" / name / child.name)
            installed[name] = {"version": manifest[name]["version"], "sha256": entry["sha256"]}
        sys.stdout.write("Installed " + name + "\n")
        sys.stdout.flush()
    receipt.write_text(json.dumps({"toolkit": version, "platform": platform,
                                  "manifest_sha256": manifest_sha256, "components": installed}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--version", choices=tuple(MANIFESTS), default=VERSION)
    args = parser.parse_args()
    destination = args.destination.resolve()
    install(destination, args.version)
    with open(os.environ["GITHUB_ENV"], "a") as output:
        output.write(f"CUDA_PATH={destination}\nCUDAToolkit_ROOT={destination}\n")
    with open(os.environ["GITHUB_PATH"], "a") as output:
        output.write(str(destination / "bin") + "\n")


if __name__ == "__main__":
    main()
