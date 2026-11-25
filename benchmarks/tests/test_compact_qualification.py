import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location("compact_qualification",Path(__file__).parents[1]/"qualify-recurrent-compact.py")
QUALIFICATION=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFICATION)


class QualificationMatrixTests(unittest.TestCase):
    def test_every_supported_dimension_is_enumerated_without_model_name_rules(self):
        manifest={"models":[
            {"name":"A","model":"a.gguf","mmproj":"p.gguf","modes":{"none":[0],"mtp":[1,2,3],
                "dflash":list(range(1,8)),"dflash-k4v":list(range(1,8))}},
            {"name":"B","model":"b.gguf","mmproj":"p2.gguf","modes":{"none":[0],"mtp":[1,2,3]}}]}
        cases=QUALIFICATION.cases(manifest)
        self.assertEqual(len(cases),528)
        self.assertEqual({c["slots"] for c in cases},{1,2,3,4})
        self.assertEqual({c["vision"] for c in cases},{False,True})
        self.assertEqual({tuple(c["kv"]) for c in cases},{("q8_0","q8_0"),("q8_0","q4_0"),("q4_0","q4_0")})

    def test_invalid_depths_and_duplicate_labels_are_rejected(self):
        for manifest in [{"models":[{"name":"A","model":"a","modes":{"mtp":[5]}}]},
                         {"models":[{"name":"A","model":"a","modes":{"none":[0]}},
                                    {"name":"A","model":"b","modes":{"none":[0]}}]}]:
            with self.assertRaises(ValueError): QUALIFICATION.cases(manifest)


if __name__=="__main__": unittest.main()
