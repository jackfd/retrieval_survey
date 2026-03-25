import unittest
from unittest.mock import Mock, patch

import numpy as np

from embed_pipe.domain.models import ModelConfig, RuntimeConfig
from embed_pipe.infra.embedding_strategies import EmbeddingStrategyFactory, HttpEmbeddingStrategy


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
            embedding_api_url="http://mock-api",
            http_timeout=5.0,
            http_max_retries=1,
        )
        self.strategy = HttpEmbeddingStrategy(runtime=self.runtime)

    @patch("embed_pipe.infra.embedding_strategies.http.requests.post")
    def test_encode_success(self, mock_post):
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        mock_post.return_value = resp

        vecs = self.strategy.encode(["a", "b"], is_query=False)
        self.assertEqual(vecs.shape, (2, 3))
        self.assertTrue(np.allclose(vecs[0], np.array([0.1, 0.2, 0.3], dtype=np.float32)))

    @patch("embed_pipe.infra.embedding_strategies.http.requests.post")
    def test_encode_invalid_payload_raises(self, mock_post):
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"invalid": []}
        mock_post.return_value = resp

        with self.assertRaises(RuntimeError):
            self.strategy.encode(["a"], is_query=False)


class TestStrategyRouting(unittest.TestCase):
    def test_build_strategy_uses_http_when_url_present(self):
        runtime = RuntimeConfig(
            embedding_dim=3,
            normalize_embeddings=False,
            max_length=128,
            query_prefix="",
            doc_prefix="",
            instruction_template="",
            batch_size=2,
            device="cpu",
            embedding_api_url="http://x",
            http_timeout=5.0,
            http_max_retries=1,
        )
        model = ModelConfig(model_name="m", provider="sentence_transformers", model_id="id")
        with patch("embed_pipe.infra.embedding_strategies.factory.HttpEmbeddingStrategy", return_value="HTTP") as http_cls:
            with patch("embed_pipe.infra.embedding_strategies.factory.LocalEmbeddingStrategy", return_value="LOCAL") as local_cls:
                result = EmbeddingStrategyFactory().build(runtime, model)
                self.assertTrue(result.ok)
                self.assertEqual(result.value, "HTTP")
                http_cls.assert_called_once()
                local_cls.assert_not_called()

    def test_build_strategy_uses_local_when_url_empty(self):
        runtime = RuntimeConfig(
            embedding_dim=3,
            normalize_embeddings=False,
            max_length=128,
            query_prefix="",
            doc_prefix="",
            instruction_template="",
            batch_size=2,
            device="cpu",
            embedding_api_url="   ",
            http_timeout=5.0,
            http_max_retries=1,
        )
        model = ModelConfig(model_name="m", provider="sentence_transformers", model_id="id")
        with patch("embed_pipe.infra.embedding_strategies.factory.HttpEmbeddingStrategy", return_value="HTTP") as http_cls:
            with patch("embed_pipe.infra.embedding_strategies.factory.LocalEmbeddingStrategy", return_value="LOCAL") as local_cls:
                result = EmbeddingStrategyFactory().build(runtime, model)
                self.assertTrue(result.ok)
                self.assertEqual(result.value, "LOCAL")
                local_cls.assert_called_once()
                http_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
