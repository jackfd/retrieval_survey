import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from embedding_client import EmbeddingClient
from selector_config import SelectorConfig


class TestEmbeddingClient(unittest.TestCase):
    def setUp(self):
        config = SelectorConfig(batch_size=2, max_retries=1, request_timeout=0.1)
        self.client = EmbeddingClient("http://mock-api", config)

    @patch("embedding_client.requests.post")
    def test_get_embeddings_success(self, mock_post):
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"vectors": [[0.1, 0.2], [0.3, 0.4]]}
        mock_post.return_value = resp

        chunks = ["a", "b"]
        vectors = self.client.get_embeddings(chunks)

        self.assertEqual(vectors.shape, (2, 2))
        self.assertTrue(np.allclose(vectors[0], np.array([0.1, 0.2])))

    @patch("embedding_client.requests.post")
    def test_get_embeddings_retry_then_success(self, mock_post):
        fail_exc = requests.exceptions.Timeout("timeout")

        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"vectors": [[0.5, 0.6], [0.7, 0.8]]}

        mock_post.side_effect = [fail_exc, resp]

        vectors = self.client.get_embeddings(["a", "b"])

        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(mock_post.call_count, 2)

    @patch("embedding_client.requests.post")
    def test_get_embeddings_invalid_payload_raises(self, mock_post):
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"invalid": []}
        mock_post.return_value = resp

        with self.assertRaises(requests.exceptions.RequestException):
            self.client.get_embeddings(["a", "b"])

    @patch("embedding_client.requests.post")
    def test_get_embeddings_row_mismatch_raises(self, mock_post):
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"vectors": [[0.1, 0.2]]}
        mock_post.return_value = resp

        with self.assertRaises(requests.exceptions.RequestException):
            self.client.get_embeddings(["a", "b"])

    def test_get_embeddings_empty_input(self):
        vectors = self.client.get_embeddings([])
        self.assertEqual(vectors.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
