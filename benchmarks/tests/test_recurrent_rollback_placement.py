"""CUDA integration: set LLAMA_ROLLBACK_TEST_SERVER and LLAMA_ROLLBACK_TEST_MODEL.

The target GGUF must contain embedded Qwen MTP weights. No models are downloaded.
"""
import contextlib
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
    @unittest.skipUnless(os.environ.get("LLAMA_ROLLBACK_TEST_DRAFT"), "set the matching DFlash2 draft path")
    def test_explicit_compact_mode_admits_four_slots(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1",0));port=listener.getsockname()[1]
        destination=os.environ.get("LLAMA_ROLLBACK_TEST_OUTPUT")
        manager=contextlib.nullcontext(destination) if destination else tempfile.TemporaryDirectory()
        with manager as temporary:
            args=FINDER.arguments([
                "--server",str(Path(os.environ["LLAMA_ROLLBACK_TEST_SERVER"]).resolve()),
                "--model",str(Path(os.environ["LLAMA_ROLLBACK_TEST_MODEL"]).resolve()),
                "--model-draft",str(Path(os.environ["LLAMA_ROLLBACK_TEST_DRAFT"]).resolve()),
                "--context-size","262144","--parallel","4","--cache-type-k","q8_0","--cache-type-v","q4_0",
                "--spec-mode","dflash","--draft-length","7","--prompt-tokens","64","--decode-tokens","16",
                "--probe-mode","preflight","--recurrent-rollback","compact","--no-warmup","--no-webui","--verbosity","4","--port",str(port)])
            args.output=Path(temporary);args.output.mkdir(exist_ok=True)
            record=FINDER.probe(args,3072)
            (args.output/"record.json").write_text(json.dumps(record,indent=2)+"\n")
            self.assertTrue(record["fits"],record.get("error"))
            joint=next(row for row in record["selection_profiles"] if row["profile"]["label"]=="joint")
            self.assertEqual(4,max(joint["decode_selections"],default=0))

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
                    "--port", str(port), *(["--recurrent-rollback", "ram" if ram else "vram"]),
                    *(["--model-draft", str(Path(os.environ["LLAMA_ROLLBACK_TEST_DRAFT"]).resolve())] if mode == "dflash" else []),
                ])
                args.output = Path(temporary) / ("ram" if ram else "vram")
                args.output.mkdir()
                original = FINDER.server_command
                with patch.object(FINDER, "server_command", side_effect=lambda config, arena: original(config, arena) + ["--tensor-split", "1"]):
                    record = FINDER.probe(args, arena)
                self.assertTrue(record["fits"], record.get("error"))
                self.assertEqual(record["command"][record["command"].index("--recurrent-rollback") + 1] == "ram", ram)
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
