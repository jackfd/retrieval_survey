import unittest
from unittest.mock import Mock, patch

import numpy as np

from index_builder.config import RuntimeConfig
from index_builder.embedding import HttpEmbeddingStrategy
from index_builder.errors import EmbeddingGenerationError


class TestHttpEmbeddingStrategy(unittest.TestCase):
    def setUp(self):
        self.runtime = RuntimeConfig(
            embedding_dim=3,
            normalize_embeddings=False,
            max_length=128,
            query_prefix="",
            doc_prefix="",
            instruction_template="",
            batch_size=2,
            device="cpu",
            embedding_mode="http",
            embedding_api_url="http://mock-api",
            http_timeout=5.0,
            http_max_retries=1,
        )
        self.strategy = HttpEmbeddingStrategy(runtime=self.runtime)

    @patch("index_builder.embedding.requests.post")
    def test_encode_success(self, mock_post):
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        mock_post.return_value = resp

        vecs = self.strategy.encode(["a", "b"], is_query=False)
        self.assertEqual(vecs.shape, (2, 3))
        self.assertTrue(np.allclose(vecs[0], np.array([0.1, 0.2, 0.3], dtype=np.float32)))

    @patch("index_builder.embedding.requests.post")
    def test_encode_invalid_payload_raises(self, mock_post):
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"invalid": []}
        mock_post.return_value = resp

        with self.assertRaises(EmbeddingGenerationError):
            self.strategy.encode(["a"], is_query=False)


if __name__ == "__main__":
    unittest.main()

