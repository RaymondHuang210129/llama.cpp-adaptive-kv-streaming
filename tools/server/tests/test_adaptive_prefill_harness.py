"""CPU contracts for logical slot IDs and isolated qualification cohorts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('adaptive_prefill', Path(__file__).with_name('test_adaptive_prefill.py'))
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

class CohortConfiguration(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(parallel=2, requests=None, slot_ids=None, swap_slots=False, prompt_tokens=None,
                      prompt_tokens_a=8192, prompt_tokens_b=2048, tokens=None, tokens_a=None, tokens_b=None)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_existing_serial_and_pair_controls_keep_two_distinct_requests(self):
        self.assertEqual(harness.request_configuration(self.args(parallel=1))[0], [0, 0])
        self.assertEqual(harness.request_configuration(self.args(swap_slots=True))[0], [1, 0])
        self.assertEqual(harness.request_configuration(self.args())[1], [8192, 2048])

    def test_larger_configured_capacity_does_not_require_all_slots(self):
        ids, counts, files = harness.request_configuration(self.args(parallel=8, requests=3, slot_ids='1,3,7',
            prompt_tokens=[1024,2048,4096], tokens=[Path('a'),Path('b'),Path('c')]))
        self.assertEqual(ids, [1,3,7])
        self.assertEqual(counts, [1024,2048,4096])
        self.assertEqual(files, [Path('a'),Path('b'),Path('c')])
        self.assertEqual(harness.request_configuration(self.args(parallel=4))[0], [0,1,2,3])

    def test_rejects_aliased_out_of_range_or_incomplete_cohorts(self):
        for changes in (dict(parallel=4,slot_ids='0,1,1,3'), dict(parallel=4,slot_ids='0,1,2,4'),
                        dict(parallel=4,requests=5), dict(parallel=4,prompt_tokens=[128]),
                        dict(parallel=257), dict(parallel=4,tokens=[Path('a')])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                harness.request_configuration(self.args(**changes))

    def test_checkpoint_coverage_rejects_missing_siblings(self):
        text='''id 0 | task 10 | new prompt, task.n_tokens = 1024
id 1 | task 11 | new prompt, task.n_tokens = 1024
id 0 | task 10 | cached n_tokens = 768
id 1 | task 11 | cached n_tokens = 768
id 0 | task 10 | created context checkpoint 1 of 32, n_tokens = 768
'''
        with self.assertRaisesRegex(AssertionError,'sibling'):
            harness.checkpoint_coverage(text,[(0,1024),(1,1024)])
        text+='id 1 | task 11 | created context checkpoint 1 of 32, n_tokens = 768\n'
        self.assertEqual([r['prefixes'] for r in harness.checkpoint_coverage(text,[(0,1024),(1,1024)])],[[768],[768]])
        with self.assertRaisesRegex(AssertionError,'unsubmitted'):
            harness.checkpoint_coverage(text.replace('n_tokens = 768\n','n_tokens = 900\n',1),[(0,1024),(1,1024)])

    def test_prefill_frame_counter_rejects_shape_rebuilds(self):
        first='stream group bind: selected=3 rows=255 phase=prefill revision=1\n'
        tail='stream group bind: selected=3 rows=93 phase=prefill revision=2\n'
        decode='stream group bind: selected=3 rows=3 phase=decode revision=3\n'
        with self.assertRaisesRegex(AssertionError,'rebuilt'):
            harness.prefill_bindings(first+tail+decode,1)
        self.assertEqual(len(harness.prefill_bindings(first+decode+tail,1)),1)

if __name__ == '__main__':
    unittest.main()
