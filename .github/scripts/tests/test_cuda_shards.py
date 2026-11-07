import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPTS = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {"fork_ci": load_module("fork_ci")}):
    shards = load_module("cuda_shards")


class Partition(unittest.TestCase):
    def test_24_hour_budget_is_enforced_without_a_wall_clock_wait(self):
        with patch.object(shards.time, "time", return_value=2000):
            self.assertEqual(shards.remaining({"deadline": 2001}), 1)
            with self.assertRaises(TimeoutError):
                shards.remaining({"deadline": 1999})

    def test_heavy_objects_are_isolated_and_every_object_has_one_owner(self):
        units = {f"{i}.o": {"source": f"/src/{'kv-stream-instance-' if i < 24 else 'normal-'}{i}.cu"}
                 for i in range(64)}
        result = shards.partition(units, 8)
        self.assertEqual(len(result), 32)
        self.assertTrue(all(len(group) == 1 for group in result[:24]))
        flattened = [name for group in result for name in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(units))

    def test_invalid_workers_and_escaping_paths_are_rejected(self):
        for value in [0, 9]:
            with self.assertRaises(ValueError):
                shards.partition({}, value)
        for path in ["../object.o", "../../secret.obj", "object.exe"]:
            with self.assertRaises(ValueError):
                shards.relative_output(Path.cwd(), path)
        with self.assertRaises(ValueError):
            shards.cmake_quote("unsafe;command")


@unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja") and shutil.which("git"), "native build tools unavailable")
class NativeHandoff(unittest.TestCase):
    """Use CPU C++ objects with .cu filenames to test transport without a GPU SDK."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cuda shard fixture ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / "build"
        self.archives = self.root / "archives"
        self.destination = self.root / "external"
        self.root.joinpath("value.h").write_text("constexpr int value = 7;\n")
        self.root.joinpath("kv-stream-instance-test.cu").write_text('#include "generated.h"\nextern "C" int a() { return value; }\n')
        self.root.joinpath("normal.cu").write_text('extern "C" int b() { return 8; }\n')
        self.root.joinpath("main.cpp").write_text('extern "C" int a(); extern "C" int b(); int main() { return a()+b()==15 ? 0 : 1; }\n')
        hook = (SCRIPTS.parents[1] / "ggml/src/ggml-cuda/ci-objects.cmake").as_posix()
        self.root.joinpath("CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\nproject(shard_fixture LANGUAGES CXX)\n'
            'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n'
            'configure_file(value.h generated.h COPYONLY)\n'
            'set_source_files_properties(kv-stream-instance-test.cu normal.cu PROPERTIES LANGUAGE CXX)\n'
            f'include("{hook}")\nggml_cuda_ci_objects(sources kv-stream-instance-test.cu normal.cu)\n'
            'add_library(ggml-cuda SHARED ${sources})\n'
            'target_include_directories(ggml-cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")\n'
            'set_target_properties(ggml-cuda PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON LINKER_LANGUAGE CXX)\n'
            'add_executable(probe main.cpp)\ntarget_link_libraries(probe PRIVATE ggml-cuda)\n')
        self.run_command(["git", "init", "-q"])
        self.run_command(["git", "add", "CMakeLists.txt", "main.cpp", "normal.cu", "kv-stream-instance-test.cu", "value.h"])
        self.run_command(["git", "-c", "user.name=CI fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"])
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        environment = patch.dict(os.environ, {"GITHUB_SHA": sha, "GITHUB_OUTPUT": str(self.root / "test-outputs")})
        environment.start()
        self.addCleanup(environment.stop)
        self.run_command(["cmake", "-S", str(self.root), "-B", str(self.build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"])
        self.plan = shards.make_plan(self.root, self.build, "fixture", 8, self.root / "plan.json")
        for index in range(len(self.plan["shards"])):
            shards.build_shard(self.plan, index, self.root, self.build, self.archives / str(index) / "objects.zip")

    def run_command(self, args):
        subprocess.run(args, cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    def rewrite_archive(self, change):
        path = self.archives / "0/objects.zip"
        with zipfile.ZipFile(path) as archive:
            payload = {name: archive.read(name) for name in archive.namelist()}
        change(payload)
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in payload.items():
                archive.writestr(name, data)

    def test_imported_objects_link_without_recompiling_shader_sources(self):
        shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)
        commands = json.loads(subprocess.check_output(["ninja", "-t", "compdb", "-x"], cwd=self.build, text=True))
        self.assertFalse(any(row["file"].endswith(".cu") for row in commands))
        shards.finish(self.plan, self.root, self.build)
        self.run_command([str(self.build / ("probe.exe" if os.name == "nt" else "probe"))])
        self.assertEqual(len(json.loads((self.build / "cuda-shards.json").read_text())["objects"]), 2)

    def test_missing_shard_and_duplicate_coverage_are_rejected(self):
        (self.archives / "1/objects.zip").unlink()
        with self.assertRaisesRegex(ValueError, "missing CUDA shard"):
            shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)
        plan = copy.deepcopy(self.plan)
        plan["shards"].append(plan["shards"][0])
        with self.assertRaisesRegex(ValueError, "coverage"):
            shards.validate_plan(plan, self.root, self.build)
        self.assertFalse(self.destination.exists())

    def test_tampered_objects_are_rejected_before_linking(self):
        def corrupt(payload):
            name = next(name for name in payload if name.startswith("objects/"))
            payload[name] += b"changed"
        self.rewrite_archive(corrupt)
        with self.assertRaises(ValueError):
            shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)
        self.assertFalse(self.destination.exists())

    def test_changed_headers_and_changed_commands_are_rejected(self):
        self.root.joinpath("value.h").write_text("constexpr int value = 9;\n")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)
        self.root.joinpath("value.h").write_text("constexpr int value = 7;\n")
        plan = copy.deepcopy(self.plan)
        next(iter(plan["units"].values()))["command_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "compiler commands"):
            shards.validate_plan(plan, self.root, self.build)

    def test_changed_generated_header_is_rejected(self):
        self.build.joinpath("generated.h").write_text("constexpr int value = 9;\n")
        with self.assertRaisesRegex(ValueError, "compiler dependency changed"):
            shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)
        self.assertFalse(self.destination.exists())

    def test_post_assembly_changes_are_rejected_at_final_link(self):
        shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)
        self.build.joinpath("generated.h").write_text("constexpr int value = 9;\n")
        with self.assertRaisesRegex(ValueError, "compiler dependency changed"):
            shards.finish(self.plan, self.root, self.build)

    def test_traversal_and_mixed_cohorts_are_rejected(self):
        self.rewrite_archive(lambda payload: payload.update({"../escape.obj": b"bad"}))
        with self.assertRaisesRegex(ValueError, "archive members"):
            shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)
        self.assertFalse(self.root.joinpath("escape.obj").exists())

        def change_profile(payload):
            payload.pop("../escape.obj")
            manifest = json.loads(payload["manifest.json"])
            manifest["profile"]["variant"] = "different"
            payload["manifest.json"] = json.dumps(manifest).encode()
        self.rewrite_archive(change_profile)
        with self.assertRaisesRegex(ValueError, "mixed CUDA"):
            shards.assemble(self.plan, self.root, self.build, self.archives, self.destination)


if __name__ == "__main__":
    unittest.main()
