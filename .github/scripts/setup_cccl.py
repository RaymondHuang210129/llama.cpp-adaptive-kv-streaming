"""Prepare a checksum-pinned, portable CCCL source artifact for all CUDA jobs."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile


PINS = {
    "v3.4.3": {"commit": "5fb1013e3c6f72877a2ebd30f54fe5158d64eec4",
               "sha256": "3edd0c8bbabd6fbdfd7d64e488c51f323d74c193064160afb872a63ecaa2c123",
               "tree_sha256": "2764e9a8d8eb80e2320ded291e66edf3c1dd11117872ac6446f57da2a1c1fac6"},
    "v2.8.2": {"commit": "207d20fb6a4165fcd2551bcf2f04b8a5ea40829f",
               "sha256": "6af362d03ddb0377d5b28a6dc6a020b1cbc1a6cbfdbab02f266a27717f2f4f96",
               "tree_sha256": "cf0bdd2499c9cbeb2db85636fdfc3f3241557dc985089cffaa69b587b55d0350"},
}
MAX_BYTES = 256 * 1024 * 1024


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def tree_digest(root):
    checksum = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("CCCL contents must not contain filesystem symlinks")
        if path.is_file():
            checksum.update(path.relative_to(root).as_posix().encode() + b"\0" + digest(path).encode() + b"\0")
    return checksum.hexdigest()


def initializer(version):
    return ('if(DEFINED GGML_CUDA_CCCL_VERSION AND NOT GGML_CUDA_CCCL_VERSION STREQUAL "' + version + '")\n'
            '    message(FATAL_ERROR "Shared CCCL version does not match the selected CUDA profile")\n'
            'endif()\n'
            'set(FETCHCONTENT_SOURCE_DIR_CCCL "${CMAKE_CURRENT_LIST_DIR}/source" CACHE PATH "Verified shared CCCL source" FORCE)\n')


def extract(archive, destination, commit):
    destination = destination.resolve()
    links = []
    size = 0
    names = set()
    with zipfile.ZipFile(archive) as payload:
        if len(payload.infolist()) > 100000:
            raise ValueError("CCCL archive has too many entries")
        for entry in payload.infolist():
            path = PurePosixPath(entry.filename)
            if (not path.parts or path.parts[0] != "cccl-" + commit or path.is_absolute()
                    or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename):
                raise ValueError("unsafe CCCL archive path")
            relative = Path(*path.parts[1:])
            if not relative.parts:
                continue
            key = relative.as_posix().casefold()
            if key in names:
                raise ValueError("duplicate CCCL archive path")
            names.add(key)
            output = destination / relative
            if entry.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            size += entry.file_size
            if size > MAX_BYTES:
                raise ValueError("CCCL archive exceeds size limit")
            output.parent.mkdir(parents=True, exist_ok=True)
            if stat.S_ISLNK(entry.external_attr >> 16):
                links.append((output, payload.read(entry).decode()))
            else:
                with payload.open(entry) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
    # Materialize upstream documentation/container aliases so Windows needs no symlink privilege.
    while links:
        pending = []
        for output, value in links:
            target = (output.parent / value).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("unsafe CCCL link path")
            if target.is_file():
                size += target.stat().st_size
                if size > MAX_BYTES:
                    raise ValueError("CCCL links exceed size limit")
                shutil.copyfile(target, output)
            elif target.is_dir():
                if output.is_relative_to(target) or target.is_relative_to(output):
                    raise ValueError("cyclic CCCL link path")
                if any(other.is_relative_to(target) for other, _ in links):
                    pending.append((output, value))
                    continue
                size += sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
                if size > MAX_BYTES:
                    raise ValueError("CCCL links exceed size limit")
                shutil.copytree(target, output)
            else:
                # Older upstream releases contain dangling documentation aliases.
                if output.suffix != ".md":
                    pending.append((output, value))
        if len(pending) == len(links):
            raise ValueError("missing or cyclic CCCL link path")
        links = pending


def install(destination, version):
    pin = PINS[version]
    receipt_path = destination / "receipt.json"
    source = destination / "source"
    expected = dict(schema=1, version=version, **pin)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("cached CCCL identity changed")
        if not source.is_dir() or source.is_symlink() or receipt.get("tree_sha256") != tree_digest(source):
            raise ValueError("cached CCCL contents changed")
        (destination / "cccl.cmake").write_text(initializer(version), newline="\n")
        return receipt
    if source.exists():
        raise ValueError("CCCL source exists without a verified receipt")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        root = Path(temporary)
        archive = root / "source.zip"
        url = "https://codeload.github.com/NVIDIA/cccl/zip/" + pin["commit"]
        with urllib.request.urlopen(url, timeout=120) as incoming, archive.open("wb") as target:
            size = 0
            while chunk := incoming.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("CCCL download exceeds size limit")
                target.write(chunk)
        if digest(archive) != pin["sha256"]:
            raise ValueError("CCCL archive checksum mismatch")
        unpacked = root / "source"
        unpacked.mkdir()
        extract(archive, unpacked, pin["commit"])
        for name in ("CMakeLists.txt", "cub/cub/version.cuh", "thrust/thrust/version.h",
                     "libcudacxx/include/cuda/std/type_traits"):
            if not (unpacked / name).is_file():
                raise ValueError("CCCL source is incomplete: " + name)
        checksum = tree_digest(unpacked)
        if pin.get("tree_sha256", checksum) != checksum:
            raise ValueError("CCCL source contents do not match the pinned tree")
        receipt = dict(expected, tree_sha256=checksum)
        unpacked.rename(source)
        (destination / "cccl.cmake").write_text(initializer(version), newline="\n")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", newline="\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=tuple(PINS), required=True)
    parser.add_argument("--destination", type=Path, default=Path(".ci-cccl"))
    args = parser.parse_args()
    install(args.destination, args.version)
