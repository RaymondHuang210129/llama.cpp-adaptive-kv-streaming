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

    def test_independent_serial_and_joint_histories_match_for_each_slot(self):
        args=FINDER.arguments(["--model","target.gguf","--parallel","2","--context-size","8192",
                               "--prompt-tokens","16","--probe-mode","full","--distinct-probe-histories"])
        profiles=FINDER.multi_slot_probe_profiles(args)
        tokens=list(range(100))
        singles=[FINDER.multi_slot_probe_prompts(args,tokens,p)[0] for p in profiles if p["label"].startswith("single-")]
        joint=FINDER.multi_slot_probe_prompts(args,tokens,next(p for p in profiles if p["label"]=="joint"))
        self.assertEqual(singles,joint)
        self.assertNotEqual(joint[0],joint[1])
        self.assertEqual(joint[0][0],joint[1][0])
        with self.assertRaises(ValueError):
            FINDER.multi_slot_probe_prompts(args,tokens[:20],next(p for p in profiles if p["label"]=="joint"))

    def test_auto_policy_roundtrips_and_conflicts_with_explicit_placements(self):
        args=FINDER.arguments(["--model","target.gguf","--parallel","4","--recurrent-rollback-auto"])
        self.assertIn("--recurrent-rollback-auto",FINDER.server_command(args,2096))
        for explicit in ("--recurrent-rollback-ram","--recurrent-rollback-compact"):
            with self.assertRaises(SystemExit):
                FINDER.arguments(["--model","target.gguf","--recurrent-rollback-auto",explicit])

    def test_mixed_mode_joint_capacity_uses_the_larger_verification_width(self):
        args=FINDER.arguments(["--model","target.gguf","--model-draft","draft.gguf","--parallel","64",
                               "--spec-mode","dflash-k4v","--draft-length","1","--ngram-m","7"])
        joint=next(p for p in FINDER.multi_slot_probe_profiles(args) if p["label"]=="joint")
        self.assertEqual(32,joint["selected_max"])


if __name__ == "__main__":
    unittest.main()
