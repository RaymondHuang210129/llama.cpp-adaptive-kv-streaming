"""Reference comparisons must reject incomplete or mismatched executions."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "compact_rollback_reference", Path(__file__).parents[1] / "compact-rollback-reference.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class ReferenceTests(unittest.TestCase):
    def record(self):
        return {"fits": True, "prompt_ids_sha256": "abc", "profiles": [
            {"label": "joint", "responses": [
                {"id_slot": 0, "tokens": [1, 2], "timings": {"predicted_n": 2}},
                {"id_slot": 1, "tokens": [3, 4], "timings": {"predicted_n": 2}},
            ]}]}

    def test_matching_runs(self):
        self.assertEqual(REFERENCE.compare(self.record(), self.record()), 4)

    def test_changed_token_is_a_failure(self):
        changed = self.record()
        changed["profiles"][0]["responses"][1]["tokens"][1] = 9
        with self.assertRaisesRegex(ValueError, "token"):
            REFERENCE.compare(self.record(), changed)

    def test_incomplete_run_is_a_failure(self):
        for key, value in (("fits", False), ("profiles", []), ("prompt_ids_sha256", "other")):
            changed = self.record()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                REFERENCE.compare(self.record(), changed)

    def test_missing_selection_is_a_failure(self):
        changed = self.record()
        changed["profiles"][0]["responses"].pop()
        with self.assertRaises(ValueError):
            REFERENCE.compare(self.record(), changed)


if __name__ == "__main__":
    unittest.main()
