import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("recurrent_measurement", Path(__file__).parents[1] / "recurrent_measurement.py")
MEASUREMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEASUREMENT)


class DecodeWindowTests(unittest.TestCase):
    def row(self, times):
        return {"tokens": list(range(len(times))), "token_times": times}

    def test_common_window_is_not_a_sum_of_request_rates(self):
        result = MEASUREMENT.decode_window([self.row([0, 1, 1, 4]), self.row([2, 3, 4])])
        self.assertEqual(result["tokens"], [1, 2])
        self.assertEqual(result["start"], 2)
        self.assertEqual(result["end"], 4)
        self.assertEqual(result["tokens_per_second"], 1.5)

    def test_speculative_bundles_are_counted_as_tokens(self):
        result = MEASUREMENT.decode_window([self.row([0, 1, 1, 2]), self.row([0.5, 1.5, 1.5, 2])])
        self.assertEqual(result["tokens"], [3, 3])
        self.assertEqual(result["tokens_per_second"], 4)

    def test_single_slot_omits_its_initial_prefill_sample(self):
        result = MEASUREMENT.decode_window([self.row([1, 2, 3, 4])])
        self.assertEqual(result["tokens"], [3])
        self.assertEqual(result["tokens_per_second"], 1)

    def test_incomplete_unsorted_and_nonoverlapping_streams_fail(self):
        cases = [[], [self.row([])], [self.row([1])], [self.row([1, 1])],
                 [self.row([1, float("nan")])], [self.row([2, 1])],
                 [self.row([1, 2]), self.row([3, 4])],
                 [{"tokens": [1, 2], "token_times": [1, 2, 3]}]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                MEASUREMENT.decode_window(rows)

    def test_vector_streams_keep_their_tokens_slots_and_final_timings(self):
        import json
        events = [{"index": i, "tokens": [100*i+t], "stop": False} for t in range(3) for i in range(2)]
        events += [{"index": i, "tokens": [], "stop": True, "id_slot": i+3,
                    "timings": {"predicted_n": 3}} for i in range(2)]
        class Response:
            def __enter__(self): return iter(("data: "+json.dumps(event)+"\n").encode() for event in events)
            def __exit__(self, *_): pass
        with patch.object(MEASUREMENT.urllib.request,"urlopen",return_value=Response()), \
                patch.object(MEASUREMENT.time,"monotonic",side_effect=range(10)):
            rows=MEASUREMENT.stream_completions("http://localhost/completion",{"prompt":[[1],[2]]},2,10)
        self.assertEqual([row["id_slot"] for row in rows],[3,4])
        self.assertEqual([row["tokens"] for row in rows],[[0,1,2],[100,101,102]])
        self.assertTrue(all(row["timings"]["predicted_n"]==3 for row in rows))
        self.assertEqual(rows[0]["common_decode_window"]["tokens"],[2,1])


if __name__ == "__main__":
    unittest.main()
