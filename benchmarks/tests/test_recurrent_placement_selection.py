import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location("placement_selection",Path(__file__).parents[1]/"select-recurrent-placement.py")
assert SPEC is not None and SPEC.loader is not None
SELECTION=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)


class PlacementSelectionTests(unittest.TestCase):
    def test_latency_choices_have_hysteresis_and_never_include_a_failed_fit(self):
        rows=[{"provider":"native","fits":True,"tokens_per_second":100},
              {"provider":"ram","fits":False,"tokens_per_second":1000},
              {"provider":"compact","fits":True,"tokens_per_second":103}]
        self.assertEqual("native",SELECTION.choose(rows,"native")["provider"])
        rows[2]["tokens_per_second"]=110
        self.assertEqual("compact",SELECTION.choose(rows,"native")["provider"])
        rows[0]["fits"]=False
        self.assertEqual("compact",SELECTION.choose(rows,"native")["provider"])

    def test_unreserved_selection_validates_the_requested_workload_before_measurement(self):
        finder=SELECTION.load("find-max-arena")
        for reserve in (None,8):
            with self.subTest(reserve=reserve),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                for name in ('server','model','corpus'):(root/name).write_text('fixture')
                options=['--server',str(root/'server'),'--model',str(root/'model'),'--prefill-text',str(root/'corpus'),
                         '--output',str(root/'results'),'--context-size','8192','--decode-tokens','64']
                if reserve is not None:options+=['--placement-reserve-mib',str(reserve)]
                def fit(config):
                    self.assertEqual(config.validate_selected,reserve is None)
                    self.assertEqual(config.decode_tokens,64 if reserve is None else 16)
                    self.assertEqual(config.probe_mode,'preflight')
                    raise RuntimeError('stop after validating fit arguments')
                with patch.object(SELECTION,'load',side_effect=lambda name:finder if name=='find-max-arena' else object()), \
                     patch.object(SELECTION,'identity',return_value={}),patch.object(finder,'fit',side_effect=fit):
                    with self.assertRaisesRegex(RuntimeError,'stop after validating fit arguments'):
                        SELECTION.main(options)

    def test_nonfinite_missing_and_duplicate_measurements_are_rejected(self):
        for rows in [[],[{"provider":"native","fits":True,"tokens_per_second":float("nan")}],
                     [{"provider":"native","fits":True,"tokens_per_second":0}],
                     [{"provider":"native","fits":True,"tokens_per_second":1}]*2]:
            with self.assertRaises(ValueError):SELECTION.choose(rows)


if __name__=="__main__":unittest.main()
