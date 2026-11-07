import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Any
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "fork_ci.py"
SPEC = importlib.util.spec_from_file_location("fork_ci", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
SHA = "0123456789abcdef0123456789abcdef01234567"


class Policy(unittest.TestCase):
    def test_core_and_workflow_changes_request_safety_checks(self):
        for path in ["src/llama-context.cpp", "ggml/src/ggml-cuda/fattn.cu", ".github/workflows/ci.yml"]:
            flags = ci.classify_changes([path])
            self.assertTrue(flags["server"])
            self.assertTrue(flags["sanitize"])
        self.assertTrue(ci.classify_changes([".github/workflows/ci.yml"])["ui"])

    def test_python_harness_and_docs_do_not_request_ui_tests(self):
        for path in ["README.md", "tools/server/tests/test_adaptive_vision_harness.py"]:
            self.assertFalse(ci.classify_changes([path])["ui"])
        self.assertTrue(ci.classify_changes(["tools/ui/src/app.html"])["ui"])

    def test_generated_checks_keep_their_input_coverage(self):
        self.assertTrue(ci.classify_changes(["conversion/base.py"])["tokenizer"])
        self.assertTrue(ci.classify_changes(["vendor/json.hpp"])["vendor"])
        self.assertTrue(ci.classify_changes(["docs/ops/cuda.csv"])["ops"])
        self.assertTrue(ci.classify_changes(["requirements/requirements-all.txt"])["requirements"])
        self.assertTrue(all(ci.classify_changes([], full=True).values()))

    def test_required_checks_reject_missing_cancelled_and_skipped_jobs(self):
        expected = {"cpu", "cuda-linux", "cuda-windows", "python"}
        good = {name: {"result": "success"} for name in expected}
        ci.validate_results(good, expected)
        for value in ["failure", "cancelled", "skipped"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ci.validate_results(dict(good, python={"result": value}), expected)
        with self.assertRaises(ValueError):
            ci.validate_results({"cpu": {"result": "success"}}, expected)

    def test_only_unrequested_skips_are_allowed(self):
        ci.validate_results({"cpu": {"result": "success"}, "ui": {"result": "skipped"}}, {"cpu"})
        with self.assertRaises(ValueError):
            ci.validate_results({"cpu": {"result": "success"}, "ui": {"result": "failure"}}, {"cpu"})

    def test_publication_requires_a_successful_v2_push(self):
        self.assertTrue(ci.can_publish("push", "refs/heads/feature/adaptive-kv-stream-v2", "success"))
        for event, ref, result in [
            ("pull_request", "refs/pull/1/merge", "success"),
            ("workflow_dispatch", "refs/heads/feature/adaptive-kv-stream-v2", "success"),
            ("push", "refs/heads/feature/other", "success"),
            ("push", "refs/heads/feature/adaptive-kv-stream-v2", "failure"),
        ]:
            self.assertFalse(ci.can_publish(event, ref, result))


class Packages(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "packages"
        self.output.mkdir()

    def fixture(self, variant):
        build = self.root / variant
        (build / "bin").mkdir(parents=True)
        windows = variant.startswith("windows")
        for name in ["llama-server", "llama-cli", "llama-bench"]:
            path = build / "bin" / (name + (".exe" if windows else ""))
            path.write_bytes(b"binary fixture\x00" + name.encode())
        for name in (["llama.dll", "ggml.dll", "ggml-base.dll"] if windows else
                     ["libllama.so.0", "libggml.so.0", "libggml-base.so.0"]):
            (build / "bin" / name).write_bytes(b"library fixture")
        if variant.endswith("cuda"):
            names = (["ggml-cuda.dll", "cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"] if windows else
                     ["libggml-cuda.so", "libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"])
            for name in names:
                (build / "bin" / name).write_bytes(b"CUDA library fixture")
        (build / "CMakeCache.txt").write_text("BUILD_SHARED_LIBS:BOOL=ON\nGGML_NATIVE:BOOL=OFF\n")
        return build

    def all_packages(self):
        for variant in ci.VARIANTS:
            ci.package(self.fixture(variant), self.output, SHA, variant)

    def test_missing_executables_never_produce_a_package(self):
        build = self.fixture("linux-x64-cpu")
        (build / "bin/llama-server").unlink()
        with self.assertRaises(ValueError):
            ci.package(build, self.output, SHA, "linux-x64-cpu")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_both_cuda_operating_system_packages_are_required_before_publication(self):
        self.all_packages()
        manifest = ci.verify_packages(self.output, SHA)
        self.assertEqual(set(manifest["packages"]), set(ci.VARIANTS))
        self.assertEqual(set(ci.VARIANTS), {"linux-x64-cuda", "windows-x64-cuda"})
        self.assertEqual(manifest["sha"], SHA)
        for row in manifest["packages"].values():
            self.assertEqual(row["sha256"], hashlib.sha256((self.output / row["file"]).read_bytes()).hexdigest())
            self.assertIn(SHA, row["file"])
        (self.output / "windows-x64-cuda.json").unlink()
        with self.assertRaises(ValueError):
            ci.verify_packages(self.output, SHA)

    def test_checksum_or_commit_tampering_is_rejected(self):
        self.all_packages()
        archive = self.output / ci.archive_name(SHA, "linux-x64-cuda")
        original = archive.read_bytes()
        archive.write_bytes(original + b"tampered")
        with self.assertRaises(ValueError):
            ci.verify_packages(self.output, SHA)
        archive.write_bytes(original)
        with self.assertRaises(ValueError):
            ci.verify_packages(self.output, "a" * 40)

    def test_manifest_cannot_redirect_publication_outside_package_directory(self):
        self.all_packages()
        path = self.output / "linux-x64-cuda.json"
        data = json.loads(path.read_text())
        data["file"] = "../private.bin"
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            ci.verify_packages(self.output, SHA)

    def test_invalid_commit_and_variant_are_rejected(self):
        for sha in ["main", "../secret", "a" * 39, "g" * 40]:
            with self.assertRaises(ValueError):
                ci.archive_name(sha, "linux-x64-cpu")
        with self.assertRaises(ValueError):
            ci.archive_name(SHA, "linux-arm64-cuda")

    def test_cuda_package_requires_backend_and_redistributable_libraries(self):
        build = self.fixture("linux-x64-cuda")
        (build / "bin/libcublas.so.13").unlink()
        with self.assertRaises(ValueError):
            ci.package(build, self.output, SHA, "linux-x64-cuda")

    def test_published_release_is_immutable_and_must_be_complete(self):
        self.all_packages()
        ci.write_release_metadata(self.output, SHA)
        assets = [{"name": p.name, "state": "uploaded", "size": p.stat().st_size,
                   "digest": "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in ci.release_assets(self.output, SHA)]
        release = {"tag_name": "sha-" + SHA, "target_commitish": SHA, "draft": False, "assets": assets}
        ci.validate_release(release, self.output, SHA)
        release["assets"] = assets[:-1]
        with self.assertRaises(ValueError):
            ci.validate_release(release, self.output, SHA)
        release["assets"] = [dict(a) for a in assets]
        release["assets"][0]["digest"] = "sha256:" + "0" * 64
        with self.assertRaises(ValueError):
            ci.validate_release(release, self.output, SHA)

    def test_upload_failure_never_publishes_a_partial_release_and_retry_resumes(self):
        self.all_packages()
        github = FakeGitHub(fail_upload=3)
        with self.assertRaises(RuntimeError):
            ci.publish(self.output, SHA, github, "42")
        assert github.release is not None
        self.assertTrue(github.release["draft"])
        self.assertEqual(github.published, 0)
        self.assertEqual(len(github.release["assets"]), 2)
        ci.publish(self.output, SHA, github, "42")
        self.assertFalse(github.release["draft"])
        self.assertEqual(len(github.release["assets"]), 4)
        self.assertEqual(github.published, 1)
        uploads = github.uploads
        path = self.output / "linux-x64-cuda.json"
        changed = json.loads(path.read_text())
        changed["build"]["run_attempt"] = "2"
        path.write_text(json.dumps(changed))
        ci.publish(self.output, SHA, github, "43")
        self.assertEqual(github.uploads, uploads)
        self.assertEqual(github.published, 1)

    def test_draft_owned_by_another_run_is_preserved(self):
        self.all_packages()
        github = FakeGitHub(fail_upload=1)
        with self.assertRaises(RuntimeError):
            ci.publish(self.output, SHA, github, "42")
        with self.assertRaises(ValueError):
            ci.publish(self.output, SHA, github, "43")
        self.assertEqual(github.uploads, 1)
        self.assertEqual(github.published, 0)

    def test_release_tag_cannot_redirect_a_sha_version_to_another_commit(self):
        self.all_packages()
        github = FakeGitHub()
        github.tag = "a" * 40
        with self.assertRaises(ValueError):
            ci.publish(self.output, SHA, github, "42")
        self.assertIsNone(github.release)
        self.assertEqual(github.uploads, 0)

    def test_internal_cpu_test_artifacts_never_become_release_assets(self):
        self.all_packages()
        ci.package(self.fixture("linux-x64-cpu"), self.output, SHA, "linux-x64-cpu")
        ci.write_release_metadata(self.output, SHA)
        self.assertEqual(len(ci.release_assets(self.output, SHA)), 4)
        self.assertFalse(any("-cpu" in path.name for path in ci.release_assets(self.output, SHA)))


class FakeGitHub:
    def __init__(self, fail_upload=0):
        self.release: dict[str, Any] | None = None
        self.tag = None
        self.data = {}
        self.uploads = self.published = 0
        self.fail_upload = fail_upload

    def request(self, method, path, payload=None, upload=None):
        if method == "GET":
            return self.release
        if method == "POST" and path == "/git/refs":
            assert payload is not None
            self.tag = payload["sha"]
            return {}
        if method == "POST" and path == "/releases":
            assert payload is not None
            self.release = dict(payload, id=1, assets=[])
            return self.release
        if method == "POST" and upload is not None:
            assert self.release is not None
            self.uploads += 1
            if self.uploads == self.fail_upload:
                raise RuntimeError("simulated transport failure")
            row = {"id": self.uploads, "name": upload.name, "state": "uploaded", "size": upload.stat().st_size,
                   "digest": "sha256:" + ci.digest(upload)}
            self.data[row["id"]] = upload.read_bytes()
            self.release["assets"].append(row)
            return row
        if method == "PATCH":
            assert self.release is not None and payload is not None
            self.release.update(payload)
            self.published += 1
            return self.release
        raise AssertionError((method, path))

    def tag_commit(self, sha):
        return self.tag

    def download_manifest(self, asset_id):
        return self.data[asset_id]


class CudaToolkit(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("setup_cuda", SCRIPT.parent / "setup_cuda.py")
        assert spec is not None and spec.loader is not None
        self.sdk = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"fork_ci": ci}):
            spec.loader.exec_module(self.sdk)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("bin/nvcc", "include/cuda_runtime.h", "lib/libcudadevrt.a",
                     "lib/libcudart_static.a", "nvvm/libdevice/libdevice.10.bc"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"SDK fixture")
        (self.root / "redistributables.json").write_text(json.dumps({
            "manifest_sha256": self.sdk.MANIFESTS[self.sdk.VERSION], "platform": "linux-x86_64"}))

    def test_cached_linux_sdk_exposes_runtime_libraries_at_nvcc_search_path(self):
        with patch.object(self.sdk.sys, "platform", "linux"):
            self.sdk.install(self.root)
            self.sdk.install(self.root)
        self.assertTrue((self.root / "lib64/libcudadevrt.a").is_file())
        self.assertTrue((self.root / "lib64/libcudart_static.a").is_file())

    def test_incomplete_cached_sdk_fails_before_compilation(self):
        (self.root / "lib/libcudart_static.a").unlink()
        with patch.object(self.sdk.sys, "platform", "linux"), self.assertRaises(ValueError):
            self.sdk.install(self.root)


class Workflows(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / "workflows"
        self.workflows = {path.name: yaml.load(path.read_text(), Loader=yaml.BaseLoader)
                          for path in self.root.glob("*.yml")}

    def test_no_empty_jobs_special_runner_services_or_upstream_secrets(self):
        for name, data in self.workflows.items():
            with self.subTest(workflow=name):
                self.assertTrue(data.get("jobs"))
                text = (self.root / name).read_text()
                for forbidden in ("self-hosted", "hf-jobs-", "ah-ubuntu", "secrets.HF_", "secrets.DEPLOY_KEY"):
                    self.assertNotIn(forbidden, text)

    def test_every_reusable_reference_exists_and_accepts_calls(self):
        for data in self.workflows.values():
            for job in data["jobs"].values():
                uses = job.get("uses", "")
                if uses.startswith("./.github/workflows/"):
                    target = self.workflows[Path(uses).name]
                    self.assertIn("workflow_call", target["on"])

    def test_required_result_is_unconditional_and_covers_four_builds(self):
        data = self.workflows["ci.yml"]
        self.assertEqual(data["on"]["push"]["branches"], ["feature/adaptive-kv-stream-v2"])
        self.assertNotIn("paths", data["on"]["pull_request"])
        self.assertIn("ready_for_review", data["on"]["pull_request"]["types"])
        required = data["jobs"]["required"]
        self.assertEqual(required["if"], "always()")
        self.assertTrue({"cpu", "cuda-linux", "cuda-windows", "python", "source"} <= set(required["needs"]))
        self.assertEqual(data["jobs"]["publish"]["needs"], "required")
        self.assertIn("github.event_name == 'push'", data["jobs"]["publish"]["if"])
        self.assertIn("needs.required.result == 'success'", data["jobs"]["publish"]["if"])

    def test_optional_platforms_are_manual_and_cpu_cuda_components_have_no_duplicate_events(self):
        for name in ("build-cpu.yml", "build-cuda-ubuntu.yml", "build-cuda-windows.yml", "release.yml"):
            self.assertEqual(set(self.workflows[name]["on"]), {"workflow_call"})
        for name in ("build-apple.yml", "build-android.yml", "build-vulkan.yml", "build-sycl.yml"):
            self.assertEqual(set(self.workflows[name]["on"]), {"workflow_dispatch"})

    def test_publisher_downloads_only_cuda_artifacts_and_cpu_outputs_are_internal(self):
        release = self.workflows["release.yml"]["jobs"]["publish"]
        downloads = [step["with"] for step in release["steps"]
                     if step.get("uses", "").startswith("actions/download-artifact@")]
        self.assertEqual([step["pattern"] for step in downloads], ["package-*-cuda"])
        cpu = self.workflows["build-cpu.yml"]["jobs"]["build"]
        uploads = [step["with"] for step in cpu["steps"]
                   if step.get("uses", "").startswith("actions/upload-artifact@")]
        self.assertEqual([step["name"] for step in uploads], ["cpu-test-${{ matrix.variant }}"])

    def run_typing_install(self, failures):
        steps = self.workflows["python-type-check.yml"]["jobs"]["typing"]["steps"]
        script = next(step["run"] for step in steps if "requirements-all.txt" in step.get("run", ""))
        stub = ('calls=0\npython() { calls=$((calls + 1)); printf "pip-attempt\\n"; '
                f'[ "$calls" -gt {failures} ]; }}\nsleep() {{ :; }}\n')
        return subprocess.run(["bash", "-e", "-c", stub + script], capture_output=True, text=True, timeout=10)

    def test_typing_install_recovers_from_a_transient_download_failure(self):
        result = self.run_typing_install(failures=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("pip-attempt\n"), 2)

    def test_typing_install_stops_after_three_failed_attempts(self):
        result = self.run_typing_install(failures=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout.count("pip-attempt\n"), 3)


if __name__ == "__main__":
    unittest.main()
