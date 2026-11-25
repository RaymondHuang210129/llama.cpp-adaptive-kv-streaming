import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "setup_cccl.py"
SPEC = importlib.util.spec_from_file_location("setup_cccl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cccl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cccl)


class SharedCccl(unittest.TestCase):
    def payload(self, extra=None):
        data = io.BytesIO()
        files = {"CMakeLists.txt": "add_library(shared_cccl INTERFACE)\n",
                 "libcudacxx/include/cuda/std/type_traits": "fixture\n",
                 "cub/cub/version.cuh": "fixture\n", "thrust/thrust/version.h": "fixture\n"}
        files.update(extra or {})
        with zipfile.ZipFile(data, "w") as archive:
            for name, value in files.items():
                archive.writestr("cccl-" + "a" * 40 + "/" + name, value)
        return data.getvalue()

    def pin(self, payload):
        pin = {"commit": "a" * 40, "sha256": hashlib.sha256(payload).hexdigest()}
        return patch.dict(cccl.PINS, {"fixture": pin})

    def test_cached_source_is_verified_without_another_pull(self):
        payload = self.payload()
        with tempfile.TemporaryDirectory() as temporary, self.pin(payload):
            root = Path(temporary) / "shared"
            with patch.object(cccl.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as fetch:
                receipt = cccl.install(root, "fixture")
            self.assertEqual(fetch.call_count, 1)
            with patch.object(cccl.urllib.request, "urlopen", side_effect=AssertionError("unexpected network")):
                self.assertEqual(cccl.install(root, "fixture"), receipt)
            self.assertEqual(json.loads((root / "receipt.json").read_text()), receipt)
            (root / "source/cub/cub/version.cuh").write_text("changed")
            with self.assertRaisesRegex(ValueError, "contents"):
                cccl.install(root, "fixture")

    def test_cache_receipt_cannot_replace_the_pinned_source_identity(self):
        payload = self.payload()
        with tempfile.TemporaryDirectory() as temporary, self.pin(payload):
            root = Path(temporary) / "shared"
            with patch.object(cccl.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                receipt = cccl.install(root, "fixture")
            cccl.PINS["fixture"]["tree_sha256"] = receipt["tree_sha256"]
            (root / "source/cub/cub/version.cuh").write_text("changed")
            receipt["tree_sha256"] = cccl.tree_digest(root / "source")
            (root / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "identity"):
                cccl.install(root, "fixture")

    def test_wrong_checksum_and_escaping_archive_fail_without_publishing_source(self):
        for payload, failure in ((self.payload(), "checksum"), (self.payload({"../escape": "bad"}), "path")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary, self.pin(payload):
                root = Path(temporary) / "shared"
                received = b"corrupt" if failure == "checksum" else payload
                with patch.object(cccl.urllib.request, "urlopen", return_value=io.BytesIO(received)), \
                        self.assertRaisesRegex(ValueError, failure):
                    cccl.install(root, "fixture")
                self.assertFalse((root / "receipt.json").exists())
                self.assertFalse((root / "source").exists())

    def test_upstream_aliases_are_portable_and_escaping_links_are_rejected(self):
        for target, success in (("version.cuh", True), ("../../../../escape", False)):
            data = io.BytesIO(self.payload())
            with zipfile.ZipFile(data, "a") as archive:
                entry = zipfile.ZipInfo("cccl-" + "a" * 40 + "/cub/cub/alias.cuh")
                entry.external_attr = 0o120777 << 16
                archive.writestr(entry, target)
            payload = data.getvalue()
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary, self.pin(payload):
                root = Path(temporary) / "shared"
                with patch.object(cccl.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                    if success:
                        cccl.install(root, "fixture")
                        alias = root / "source/cub/cub/alias.cuh"
                        self.assertFalse(alias.is_symlink())
                        self.assertEqual(alias.read_text(), "fixture\n")
                    else:
                        with self.assertRaisesRegex(ValueError, "path"):
                            cccl.install(root, "fixture")

    def test_initializer_uses_shared_source_offline_and_rejects_wrong_version(self):
        payload = self.payload()
        with tempfile.TemporaryDirectory(prefix="shared cccl ") as temporary, self.pin(payload):
            root = Path(temporary)
            dependency = root / "dependency"
            with patch.object(cccl.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                cccl.install(dependency, "fixture")
            project = root / "project"
            project.mkdir()
            (project / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.20)\nproject(offline LANGUAGES NONE)\n'
                'include(FetchContent)\nFetchContent_Declare(CCCL GIT_REPOSITORY "missing://no-network")\n'
                'FetchContent_MakeAvailable(CCCL)\n'
                'if(NOT TARGET shared_cccl)\nmessage(FATAL_ERROR "shared source was not used")\nendif()\n')
            worker = root / "worker/build/_deps/cccl-src"
            shutil.copytree(dependency, worker)
            shutil.rmtree(dependency)
            environment = dict(os.environ, CMAKE_TOOLCHAIN_FILE=str(worker / "cccl.cmake"))
            for version, success in (("fixture", True), ("wrong", False)):
                result = subprocess.run(["cmake", "-S", str(project), "-B", str(root / version), "-G", "Ninja",
                                         "-DGGML_CUDA_CCCL_VERSION=" + version], env=environment,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.assertEqual(result.returncode == 0, success, result.stdout)


if __name__ == "__main__":
    unittest.main()
