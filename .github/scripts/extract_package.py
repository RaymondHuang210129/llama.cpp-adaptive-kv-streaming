"""Extract a verified CI package for the existing server API suite."""

import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import zipfile

from fork_ci import digest


def main():
    directory, destination = map(Path, sys.argv[1:])
    manifests = list(directory.glob("*-x64-*.json"))
    if len(manifests) != 1:
        raise ValueError("exactly one tested package is required")
    row = json.loads(manifests[0].read_text())
    archive = directory / row["file"]
    if archive.parent != directory or digest(archive) != row["sha256"]:
        raise ValueError("invalid package identity/checksum")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as payload:
                if any(not (root / name).resolve().is_relative_to(root) for name in payload.namelist()):
                    raise ValueError("invalid package entry")
                payload.extractall(root)
        else:
            with tarfile.open(archive) as payload:
                payload.extractall(root, filter="data")
        content = list(root.iterdir())
        if len(content) != 1 or not content[0].is_dir():
            raise ValueError("unexpected package layout")
        shutil.copytree(content[0], destination, symlinks=True, dirs_exist_ok=True)
    if os.environ.get("GITHUB_ENV"):
        executable = destination.resolve() / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
        with open(os.environ["GITHUB_ENV"], "a") as output:
            output.write("LLAMA_SERVER_BIN_PATH=" + str(executable) + "\n")


if __name__ == "__main__":
    main()
