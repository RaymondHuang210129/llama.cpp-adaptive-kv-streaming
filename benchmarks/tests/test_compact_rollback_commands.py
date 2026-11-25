import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


FINDER = module("compact_finder", ROOT / "find-max-arena.py")
SWEEP = module("compact_sweep", ROOT / "run-fixed-span-sweep.py")


class CompactCommandTests(unittest.TestCase):
    def test_finder_and_sweep_preserve_explicit_compact_policy(self):
        for enabled in (False, True):
            extra = ["--recurrent-rollback-compact"] if enabled else []
            args = FINDER.arguments(["--model", "target.gguf", "--spec-mode", "mtp", *extra])
            self.assertEqual(enabled, "--recurrent-rollback-compact" in FINDER.server_command(args, 2096))
            with patch.object(sys, "argv", [str(ROOT / "run-fixed-span-sweep.py"), "--model", "target.gguf", *extra]):
                config = SWEEP.arguments()
            for depth in (0, 3):
                self.assertEqual(enabled, "--recurrent-rollback-compact" in SWEEP.server_command(config, 8192, depth, 2096))

    def test_conflicting_policies_are_rejected_before_launch(self):
        with self.assertRaises(SystemExit):
            FINDER.arguments(["--model", "target.gguf", "--recurrent-rollback-compact", "--recurrent-rollback-ram"])

    def test_environment_cannot_silently_enable_compact_in_control_runs(self):
        with patch.dict(os.environ, {"LLAMA_ARG_RECURRENT_ROLLBACK_COMPACT": "1"}):
            self.assertNotIn("LLAMA_ARG_RECURRENT_ROLLBACK_COMPACT", SWEEP.server_environment(False))
            self.assertNotIn("LLAMA_ARG_RECURRENT_ROLLBACK_COMPACT", FINDER.environment())


if __name__ == "__main__":
    unittest.main()
