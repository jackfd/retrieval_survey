import tempfile
import unittest
from pathlib import Path

from embed_pipe.domain.errors import InputValidationError
from embed_pipe.infra.dataset_loader import DatasetLoader


class TestDatasetResolution(unittest.TestCase):
    def test_exact_match_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MSMARCO").mkdir()
            (root / "HotpotQA").mkdir()
            resolved = DatasetLoader().resolve_subdataset_dir(root, "MSMARCO")
            self.assertEqual(resolved.name, "MSMARCO")

    def test_case_insensitive_unique_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "HotpotQA").mkdir()
            resolved = DatasetLoader().resolve_subdataset_dir(root, "hotpotqa")
            self.assertEqual(resolved.name, "HotpotQA")

    def test_case_insensitive_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "SciFact").mkdir()
            try:
                (root / "scifact").mkdir()
            except FileExistsError:
                self.skipTest("Case-insensitive filesystem does not allow ambiguous case-only dirs")
            with self.assertRaises(InputValidationError):
                DatasetLoader().resolve_subdataset_dir(root, "SCIFACT")


if __name__ == "__main__":
    unittest.main()
