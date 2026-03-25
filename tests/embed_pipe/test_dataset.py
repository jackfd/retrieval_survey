import tempfile
import unittest
from pathlib import Path

from embed_pipe.infra.dataset_loader import DatasetLoader


class TestDatasetResolution(unittest.TestCase):
    def test_exact_match_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MSMARCO").mkdir()
            (root / "HotpotQA").mkdir()
            resolved = DatasetLoader().resolve_subdataset_dir(root, "MSMARCO")
            self.assertTrue(resolved.ok)
            self.assertEqual(resolved.value.name, "MSMARCO")

    def test_case_insensitive_unique_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "HotpotQA").mkdir()
            resolved = DatasetLoader().resolve_subdataset_dir(root, "hotpotqa")
            self.assertTrue(resolved.ok)
            self.assertEqual(resolved.value.name, "HotpotQA")

    def test_case_insensitive_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "SciFact").mkdir()
            try:
                (root / "scifact").mkdir()
            except FileExistsError:
                self.skipTest("Case-insensitive filesystem does not allow ambiguous case-only dirs")
            resolved = DatasetLoader().resolve_subdataset_dir(root, "SCIFACT")
            self.assertFalse(resolved.ok)
            self.assertIn("Ambiguous", resolved.error_message)


if __name__ == "__main__":
    unittest.main()
