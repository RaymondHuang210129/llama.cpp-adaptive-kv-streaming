import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / "tests/test-tokenizers-repo.sh"


@unittest.skipIf(os.name == "nt", "the vocabulary shell driver is a POSIX test")
class VocabularyDriver(unittest.TestCase):
    def run_fixture(self, fixtures, spaces=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/git").write_text("#!/bin/sh\nexit 0\n")
            (root / "bin/git").chmod(0o755)
            folder = root / ("vocab cache" if spaces else "vocabs")
            (folder / ".git").mkdir(parents=True)
            (folder / "BPE").mkdir()
            for name in fixtures:
                for suffix in ("", ".inp", ".out"):
                    (folder / "BPE" / (name + ".gguf" + suffix)).write_text("fixture")
            tokenizer = root / "tokenizer"
            tokenizer.write_text('#!/bin/sh\necho "$1"\ncase "$1" in *bad*) exit 7;; esac\n')
            tokenizer.chmod(0o755)
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"])
            return subprocess.run(["bash", str(SCRIPT), "https://fixture", str(folder), str(tokenizer)],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def test_an_earlier_failure_cannot_be_hidden_by_a_later_passing_vocab(self):
        result = self.run_fixture(["bad", "good"])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("bad.gguf", result.stdout)

    def test_no_executed_fixtures_is_not_a_passing_test(self):
        result = self.run_fixture([])
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_all_valid_fixtures_are_executed(self):
        result = self.run_fixture(["good", "other"], spaces=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("good.gguf", result.stdout)
        self.assertIn("other.gguf", result.stdout)


if __name__ == "__main__":
    unittest.main()
