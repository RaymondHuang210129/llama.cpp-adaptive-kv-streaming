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
    def test_serial_state_quota_is_a_lower_bound_and_allocation_failure_is_not(self):
        prefix='serial recurrent state does not fit the declared arena: '
        message=prefix+'insufficient arena quota; required=104857601, available=67108864, additional=37748737 bytes'
        minimum=SWEEP.arena_minimum_mib(message)
        self.assertEqual(101,minimum)
        self.assertFalse(SWEEP.is_arena_capacity_failure(message))
        self.assertIsNone(SWEEP.arena_minimum_mib(prefix+'backend or host allocation failed'))
        self.assertEqual(2219,SWEEP.arena_minimum_mib(
            'workspace admission: insufficient arena quota; required=2326112384, available=2197815296, additional=128297088 bytes'))
        def probe(candidate):
            if candidate<minimum:raise SWEEP.ArenaTooSmallError(minimum)
            return candidate<=132
        self.assertEqual(132,SWEEP.find_max_arena_mib(64,probe,upper_limit=200))

    def test_finder_and_sweep_preserve_each_explicit_mode(self):
        for mode in ("ram", "vram", "compact"):
            args = FINDER.arguments(["--model", "target.gguf", "--spec-mode", "mtp", "--recurrent-rollback", mode])
            command = FINDER.server_command(args, 2096)
            self.assertEqual(mode, command[command.index("--recurrent-rollback") + 1])
            with patch.object(sys, "argv", [str(ROOT / "run-fixed-span-sweep.py"), "--model", "target.gguf", "--recurrent-rollback", mode]):
                config = SWEEP.arguments()
            for depth in (0, 3):
                command = SWEEP.server_command(config, 8192, depth, 2096)
                self.assertEqual(mode, command[command.index("--recurrent-rollback") + 1])

    def test_invalid_modes_are_rejected_before_launch(self):
        for mode in ("auto", "native", "RAM", ""):
            with self.subTest(mode=mode), self.assertRaises(SystemExit):
                FINDER.arguments(["--model", "target.gguf", "--recurrent-rollback", mode])
            with patch.object(sys, "argv", [str(ROOT / "run-fixed-span-sweep.py"), "--model", "target.gguf", "--recurrent-rollback", mode]), self.assertRaises(SystemExit):
                SWEEP.arguments()

    def test_environment_cannot_silently_enable_compact_in_control_runs(self):
        with patch.dict(os.environ, {"LLAMA_ARG_RECURRENT_ROLLBACK": "compact"}):
            self.assertNotIn("LLAMA_ARG_RECURRENT_ROLLBACK", SWEEP.server_environment(False))
            self.assertNotIn("LLAMA_ARG_RECURRENT_ROLLBACK", FINDER.environment())

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

    def test_missing_mode_stays_unselected_and_old_flags_are_rejected(self):
        args=FINDER.arguments(["--model","target.gguf"])
        command=FINDER.server_command(args,2096)
        self.assertIsNone(args.recurrent_rollback)
        self.assertNotIn("--recurrent-rollback",command)
        for flag in ("--recurrent-rollback-ram","--recurrent-rollback-compact","--recurrent-rollback-auto"):
            with self.subTest(flag=flag),self.assertRaises(SystemExit):
                FINDER.arguments(["--model","target.gguf",flag])

    def test_mixed_mode_joint_capacity_uses_the_larger_verification_width(self):
        args=FINDER.arguments(["--model","target.gguf","--model-draft","draft.gguf","--parallel","64",
                               "--spec-mode","dflash-k4v","--draft-length","1","--ngram-m","7"])
        joint=next(p for p in FINDER.multi_slot_probe_profiles(args) if p["label"]=="joint")
        self.assertEqual(32,joint["selected_max"])


if __name__ == "__main__":
    unittest.main()
