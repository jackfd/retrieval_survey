import unittest
from unittest.mock import patch

import numpy as np

from embed_pipe.services.chunking.chunk_scorer import ChunkScorer
from embed_pipe.services.chunking.selector_config import SelectorConfig


class TestChunkScorer(unittest.TestCase):
    def setUp(self):
        config = SelectorConfig(top_keywords=5, cooccur_window=2)
        self.scorer = ChunkScorer(stop_words={"the", "and", "of"}, config=config)
        self.chunks = [
            "alpha beta gamma",
            "beta gamma delta",
            "中文 测试 文本",
        ]

    @patch("embed_pipe.services.chunking.chunk_scorer.SKLEARN_AVAILABLE", False)
    def test_compute_global_statistics_fallback(self):
        self.scorer.compute_global_statistics(self.chunks)

        self.assertGreater(self.scorer.tfidf_matrix.size, 0)
        self.assertGreaterEqual(len(self.scorer.keywords), 1)
        self.assertEqual(len(self.scorer.text_rank_scores), len(self.chunks))

    @patch("embed_pipe.services.chunking.chunk_scorer.SKLEARN_AVAILABLE", False)
    def test_compute_scores(self):
        self.scorer.compute_global_statistics(self.chunks)
        candidate_idxs = [0, 1, 2]

        scores = self.scorer.compute_scores(self.chunks, candidate_idxs, title="gamma 测试")

        self.assertEqual(scores.shape[0], len(candidate_idxs))
        self.assertTrue(np.all(np.isfinite(scores)))

    @patch("embed_pipe.services.chunking.chunk_scorer.SKLEARN_AVAILABLE", False)
    def test_empty_vocabulary_raises(self):
        with self.assertRaises(ValueError):
            self.scorer.compute_global_statistics(["the and of", "and of the"])


if __name__ == "__main__":
    unittest.main()
