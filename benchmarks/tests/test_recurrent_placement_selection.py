import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location("placement_selection",Path(__file__).parents[1]/"select-recurrent-placement.py")
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

    def test_nonfinite_missing_and_duplicate_measurements_are_rejected(self):
        for rows in [[],[{"provider":"native","fits":True,"tokens_per_second":float("nan")}],
                     [{"provider":"native","fits":True,"tokens_per_second":0}],
                     [{"provider":"native","fits":True,"tokens_per_second":1}]*2]:
            with self.assertRaises(ValueError):SELECTION.choose(rows)


if __name__=="__main__":unittest.main()
