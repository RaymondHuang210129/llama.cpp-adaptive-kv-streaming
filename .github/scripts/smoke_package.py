"""Check packaged executables and dependencies without claiming CUDA execution."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from fork_ci import digest


def main():
    directory = Path(sys.argv[1])
    manifests = list(directory.glob("*-x64-*.json"))
    if len(manifests) != 1:
        raise ValueError("one package manifest is required")
    manifest = json.loads(manifests[0].read_text())
    archive = directory / manifest["file"]
    if digest(archive) != manifest["sha256"]:
        raise ValueError("package checksum changed before startup checks")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as payload:
                if any(not (root / name).resolve().is_relative_to(root) for name in payload.namelist()):
                    raise ValueError("invalid package path")
                payload.extractall(root)
        else:
            with tarfile.open(archive) as payload:
                payload.extractall(root, filter="data")
        package = root / manifest["file"].removesuffix(".tar.gz").removesuffix(".zip")
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("LD_PRELOAD", None)
        if sys.platform == "win32":
            env["PATH"] = str(package) + os.pathsep + str(Path(env["WINDIR"]) / "System32")
            for library in package.glob("*.dll"):
                output = subprocess.check_output(["dumpbin", "/dependents", str(library)], text=True)
                for name in re.findall(r"^\s+([\w.-]+\.dll)\s*$", output, re.M | re.I):
                    if name.lower() == "nvcuda.dll" or name.lower().startswith("api-ms-win-"):
                        continue
                    if not (package / name).exists() and not (Path(env["WINDIR"]) / "System32" / name).exists():
                        raise ValueError("missing packaged DLL: " + name)
        else:
            for library in package.glob("*.so*"):
                result = subprocess.run(["ldd", str(library)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if result.returncode:
                    raise RuntimeError(result.stdout)
                missing = re.findall(r"^\s*(\S+) => not found", result.stdout, re.M)
                if set(missing) - {"libcuda.so.1", "libnvidia-ml.so.1"}:
                    raise ValueError("missing packaged libraries: " + str(missing))
        for name in ("llama-server", "llama-cli", "llama-bench"):
            executable = package / (name + (".exe" if sys.platform == "win32" else ""))
            subprocess.run([str(executable), "--help"], cwd=package, env=env, timeout=60, check=True,
                           stdout=subprocess.DEVNULL)
    sys.stdout.write("Packaged startup and library checks passed; CUDA execution was not tested.\n")


if __name__ == "__main__":
    main()
