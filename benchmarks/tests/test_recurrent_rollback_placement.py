"""CUDA integration: set LLAMA_ROLLBACK_TEST_SERVER and LLAMA_ROLLBACK_TEST_MODEL.

The target GGUF must contain embedded Qwen MTP weights. No models are downloaded.
"""
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rollback_arena_finder", ROOT / "benchmarks/find-max-arena.py")
assert SPEC is not None and SPEC.loader is not None
FINDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINDER)


@unittest.skipUnless(os.environ.get("LLAMA_ROLLBACK_TEST_SERVER") and os.environ.get("LLAMA_ROLLBACK_TEST_MODEL"),
                     "set rollback test server and embedded-MTP model paths for CUDA integration")
class RecurrentRollbackPlacementTests(unittest.TestCase):
    def test_selected_mtp_restores_prefixes_in_vram_and_ram(self):
        self.check_placement("mtp", 3, 2096)

    @unittest.skipUnless(os.environ.get("LLAMA_ROLLBACK_TEST_DRAFT"), "set the matching DFlash2 draft path")
    def test_selected_dflash_restores_prefixes_in_vram_and_ram(self):
        self.check_placement("dflash", 7, 4096)

    def check_placement(self, mode, depth, arena):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        outputs = []
        with tempfile.TemporaryDirectory() as temporary:
            for ram in (False, True):
                args = FINDER.arguments([
                    "--server", str(Path(os.environ["LLAMA_ROLLBACK_TEST_SERVER"]).resolve()),
                    "--model", str(Path(os.environ["LLAMA_ROLLBACK_TEST_MODEL"]).resolve()),
                    "--context-size", "8192", "--parallel", "2", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                    "--spec-mode", mode, "--draft-length", str(depth), "--decode-tokens", "64", "--batch-size", "256",
                    "--ubatch-size", "256", "--probe-mode", "preflight", "--no-webui", "--verbosity", "4",
                    "--port", str(port), *(["--recurrent-rollback-ram"] if ram else []),
                    *(["--model-draft", str(Path(os.environ["LLAMA_ROLLBACK_TEST_DRAFT"]).resolve())] if mode == "dflash" else []),
                ])
                args.output = Path(temporary) / ("ram" if ram else "vram")
                args.output.mkdir()
                original = FINDER.server_command
                with patch.object(FINDER, "server_command", side_effect=lambda config, arena: original(config, arena) + ["--tensor-split", "1"]):
                    record = FINDER.probe(args, arena)
                self.assertTrue(record["fits"], record.get("error"))
                self.assertEqual("--recurrent-rollback-ram" in record["command"], ram)
                profiles = []
                for artifact in sorted(args.output.glob(f"arena-{arena}-*-response.json")):
                    value = json.loads(artifact.read_text())
                    if "responses" in value:
                        profiles.append([(row["id_slot"], row["tokens"]) for row in value["responses"]])
                self.assertTrue(profiles, "the finder must exercise native multi-slot selections")
                outputs.append(profiles)
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
