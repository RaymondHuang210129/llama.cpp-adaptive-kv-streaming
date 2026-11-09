import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {"fork_ci": load("fork_ci")}):
    probe = load("profile_cuda_compile")


class Selection(unittest.TestCase):
    def test_windows_counter_status_accepts_valid_and_new_data_and_rejects_missing_data(self):
        with patch.object(sys, "platform", "win32"):
            self.assertTrue(probe.valid_telemetry({"counters": [{"Status": 0}, {"Status": 1}]}))
            self.assertFalse(probe.valid_telemetry({"counters": []}))
            self.assertFalse(probe.valid_telemetry({"counters": [{"Status": 0xC0000BBA}]}))

    def test_missing_duplicate_escaping_and_cached_objects_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve()
            source = probe.SOURCES[0]
            entry = {"file": str(build / source), "output": "kernel.obj", "command": "nvcc"}
            for entries in ([], [entry, entry], [dict(entry, output="../escape.obj")]):
                with patch.object(probe.subprocess, "check_output", return_value=json.dumps(entries)), \
                        self.assertRaises(ValueError):
                    probe.select_object(build, source)
            (build / "kernel.obj").write_bytes(b"cached")
            with patch.object(probe.subprocess, "check_output", return_value=json.dumps([entry])), \
                    self.assertRaisesRegex(ValueError, "existing object"):
                probe.select_object(build, source)

    def test_only_timing_and_verbose_flags_are_added(self):
        with patch.dict(os.environ, {}, clear=True):
            env = probe.diagnostic_environment(Path("output with spaces"))
        self.assertEqual(set(env), {"NVCC_APPEND_FLAGS"})
        self.assertEqual(env["NVCC_APPEND_FLAGS"], subprocess.list2cmdline([
            "--time", str(Path("output with spaces") / "phases.csv"), "--verbose"]))
        for flag in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
            with patch.dict(os.environ, {flag: "--Ofast-compile=max"}), self.assertRaises(ValueError):
                probe.diagnostic_environment(Path("output"))


class Controller(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="compile probe ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / probe.SOURCES[0]
        self.source.write_text("source fixture\n")
        self.binary = self.root / "kernel.obj"
        self.entry = {"file": str(self.source), "output": self.binary.name, "command": "nvcc fixture"}
        self.popen = subprocess.Popen

    def run_probe(self, failure=None, telemetry_error=False, timeout=False, preflight_error=False):
        def start(command, **kwargs):
            self.assertEqual(command, ["cmake", "--build", str(self.root), "--parallel", "1", "--target", "kernel.obj"])
            self.assertIn("--time", kwargs["env"]["NVCC_APPEND_FLAGS"])
            script = f"from pathlib import Path; Path({str(self.binary)!r}).write_bytes(b'object fixture')"
            if timeout:
                script = "import time; time.sleep(30)"
            if failure:
                script = f"raise SystemExit({failure})"
            return self.popen([sys.executable, "-c", script], **kwargs)

        snapshot = {"processes": []}
        if telemetry_error:
            snapshot["counter_errors"] = ["counter unavailable"]
        samples = [{"processes": []}, snapshot] if telemetry_error else [snapshot]
        if preflight_error:
            samples = [{"counter_errors": ["counter unavailable"]}]

        def sample(pid):
            return samples.pop(0) if len(samples) > 1 else samples[0]

        with patch.object(probe, "select_object", return_value=(self.entry, self.binary)), \
                patch.object(probe, "read_configuration", return_value={}), \
                patch.object(probe.subprocess, "check_output", return_value="a" * 40 + "\n"), \
                patch.object(probe.subprocess, "Popen", side_effect=start), \
                patch.object(probe, "snapshot", side_effect=sample), \
                patch.dict(os.environ, {"NVCC_APPEND_FLAGS": "", "NVCC_PREPEND_FLAGS": ""}), \
                patch.object(sys, "stdout", io.StringIO()):
            probe.profile(self.root, self.source.name, self.root / "diagnostics", 0.1 if timeout else 10)

    def result(self):
        return json.loads((self.root / "diagnostics/result.json").read_text())

    def test_fresh_object_success_is_reported_with_instrumentation_provenance(self):
        self.run_probe()
        self.assertTrue(self.result()["passed"])
        self.assertTrue(self.result()["compile_completed"])
        self.assertEqual(self.result()["object_sha256"], probe.digest(self.binary))
        self.assertTrue((self.root / "diagnostics/provenance.json").is_file())

    def test_compiler_failure_is_not_reported_as_success(self):
        with self.assertRaises(RuntimeError):
            self.run_probe(failure=7)
        self.assertFalse(self.result()["passed"])
        self.assertEqual(self.result()["exit_code"], 7)

    def test_missing_resource_counters_are_not_a_qualified_diagnostic(self):
        with self.assertRaisesRegex(RuntimeError, "sampling was incomplete"):
            self.run_probe(telemetry_error=True)
        self.assertTrue(self.result()["compile_completed"])
        self.assertFalse(self.result()["passed"])

    def test_resource_preflight_failure_prevents_compiler_start(self):
        with self.assertRaisesRegex(RuntimeError, "no compiler started"):
            self.run_probe(preflight_error=True)
        self.assertFalse(self.result()["compile_completed"])
        self.assertFalse(self.result()["passed"])
        self.assertFalse(self.binary.exists())

    def test_timeout_stops_the_owned_compiler_and_writes_failure(self):
        with self.assertRaises(TimeoutError):
            self.run_probe(timeout=True)
        self.assertFalse(self.result()["passed"])
        self.assertIn("deadline", self.result()["error"])


if __name__ == "__main__":
    unittest.main()
