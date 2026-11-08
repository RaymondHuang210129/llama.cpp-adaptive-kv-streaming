from contextlib import nullcontext
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPTS = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = load_module("fork_ci")
with patch.dict(sys.modules, {"fork_ci": ci}):
    smoke = load_module("smoke_package")
    extract = load_module("extract_package")


class PackageExtraction(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="package extraction ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.packages = self.root / "packages"
        self.packages.mkdir()
        self.destination = self.root / "installed"
        self.extension = ".exe" if sys.platform == "win32" else ""

    def fixture(self, kind="zip", extra=None):
        variant = "windows-x64-cpu" if kind == "zip" else "linux-x64-cpu"
        filename = ci.archive_name(SHA, variant)
        name = filename.removesuffix(".tar.gz").removesuffix(".zip")
        members = {f"{name}/{exe}{self.extension}": b"executable fixture"
                   for exe in ("llama-server", "llama-cli", "llama-bench")}
        members[f"{name}/licenses/runtime/LICENSE"] = b"license fixture"
        members.update(extra or {})
        archive = self.packages / filename
        if kind == "zip":
            with zipfile.ZipFile(archive, "w") as payload:
                for path, data in members.items():
                    payload.writestr(path, data)
        else:
            with tarfile.open(archive, "w:gz") as payload:
                for path, data in members.items():
                    info = tarfile.TarInfo(path)
                    info.size = len(data)
                    payload.addfile(info, io.BytesIO(data))
        (self.packages / (variant + ".json")).write_text(json.dumps({"file": filename, "sha256": ci.digest(archive)}))
        return archive

    def run_script(self, script):
        base = self.root / script.__name__
        (base / "alias").mkdir(parents=True)
        (base / "extracted").mkdir()
        # Model a noncanonical temp path without requiring symlink privileges on Windows.
        temporary = base / "alias" / ".." / "extracted"
        args = [script.__name__, str(self.packages)]
        if script is extract:
            args.append(str(self.destination))

        def startup(command, **kwargs):
            self.assertTrue(Path(command[0]).is_file())
            self.assertEqual(Path(kwargs["cwd"]).resolve(), temporary.resolve() / Path(command[0]).parent.name)
            self.assertEqual(command[1:], ["--help"])
            self.assertTrue(kwargs["check"])
            return subprocess.CompletedProcess(command, 0)

        with patch.object(sys, "argv", args), patch.dict(script.os.environ, {"GITHUB_ENV": ""}), \
                patch.object(script.tempfile, "TemporaryDirectory", return_value=nullcontext(str(temporary))):
            if script is smoke:
                with patch.object(smoke.subprocess, "run", side_effect=startup) as run:
                    script.main()
                self.assertEqual(run.call_count, 3)
            else:
                script.main()
        if script is extract:
            for name in ("llama-server", "llama-cli", "llama-bench"):
                self.assertEqual((self.destination / (name + self.extension)).read_bytes(), b"executable fixture")
            self.assertEqual((self.destination / "licenses/runtime/LICENSE").read_bytes(), b"license fixture")

    def test_valid_zip_accepts_noncanonical_temporary_directory(self):
        self.fixture()
        for script in (smoke, extract):
            with self.subTest(script=script.__name__):
                self.run_script(script)

    def test_valid_tar_still_extracts_and_starts(self):
        self.fixture("tar")
        for script in (smoke, extract):
            with self.subTest(script=script.__name__):
                self.run_script(script)

    def test_zip_paths_cannot_escape_resolved_temporary_directory(self):
        for index, name in enumerate(("../escaped", str(self.root / "escaped"))):
            for script in (smoke, extract):
                with self.subTest(script=script.__name__, path=name):
                    self.fixture(extra={name: b"escape"})
                    with self.assertRaisesRegex(ValueError, "invalid package (path|entry)"):
                        self.run_script(script)
                    self.assertFalse((self.root / "escaped").exists())
                    self.assertFalse(self.destination.exists())
                    # Each subcase gets a fresh extraction directory.
                    (self.root / script.__name__).rename(self.root / f"previous-{index}-{script.__name__}")

    def test_changed_archive_is_rejected_before_extraction(self):
        archive = self.fixture()
        archive.write_bytes(archive.read_bytes() + b"tampered")
        for script in (smoke, extract):
            with self.subTest(script=script.__name__), self.assertRaisesRegex(ValueError, "checksum"):
                self.run_script(script)


if __name__ == "__main__":
    unittest.main()
