import unittest
from unittest.mock import patch

import numpy as np

from embed_pipe.domain.models import RuntimeConfig
from embed_pipe.infra.embedding_strategies import BaseEmbeddingStrategy
from embed_pipe.services.chunking import ChunkSelector, SelectorConfig


class _DummyEmbeddingStrategy(BaseEmbeddingStrategy):
    def __init__(self, dim: int = 8):
        runtime = RuntimeConfig(
            embedding_dim=dim,
            normalize_embeddings=False,
            max_length=128,
            query_prefix="",
            doc_prefix="",
            instruction_template="",
            batch_size=8,
            device="cpu",
            embedding_api_url="",
            http_timeout=1.0,
            http_max_retries=0,
        )
        super().__init__(runtime)
        self.dim = dim
        self.fail = False

    def encode(self, texts, is_query):
        if self.fail:
            raise RuntimeError("mock encode failed")
        return np.ones((len(texts), self.dim), dtype=np.float32)


class TestChunkSelector(unittest.TestCase):
    def setUp(self):
        config = SelectorConfig(cluster_ratio=1.0)
        self.embedding_strategy = _DummyEmbeddingStrategy(dim=8)
        self.selector = ChunkSelector(
            embedding_strategy=self.embedding_strategy,
            chunk_num=3,
            config=config,
        )
        self.text = (
            "第一段内容，这是一个比较长的段落，超过30个字符，用于测试分段逻辑。\n\n"
            "第二段内容，也是比较长的段落，同样超过30个字符。\n\n"
            "第三段内容，继续补充信息，用于测试排序稳定性。\n\n"
            "第四段内容，作为备选段落，以防前面的段落数量不够。"
        )

    @patch.object(ChunkSelector, "get_embeddings")
    def test_select_chunks_basic(self, mock_get_embeddings):
        chunks = self.selector.split_paragraphs(self.text)
        mock_get_embeddings.return_value = np.random.rand(len(chunks), 8)

        results = self.selector.select_chunks(self.text, "测试标题")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), min(self.selector.chunk_num, len(chunks)))
        for item in results:
            self.assertIn("chunk_text", item)
            self.assertIn("score", item)
            self.assertIn("embedding", item)
            self.assertIsInstance(item["chunk_text"], str)
            self.assertIsInstance(item["score"], float)

    @patch.object(ChunkSelector, "get_embeddings")
    def test_candidate_less_than_chunk_num(self, mock_get_embeddings):
        selector = ChunkSelector(
            embedding_strategy=_DummyEmbeddingStrategy(dim=6),
            chunk_num=5,
            config=SelectorConfig(cluster_ratio=1.0),
        )
        text = "只有一段。"
        chunks = selector.split_paragraphs(text)
        mock_get_embeddings.return_value = np.random.rand(len(chunks), 6)

        results = selector.select_chunks(text, "标题")
        self.assertEqual(len(results), 1)

    def test_empty_text(self):
        results = self.selector.select_chunks("", "标题")
        self.assertEqual(results, [])

    @patch.object(ChunkSelector, "get_embeddings")
    def test_embedding_failure_returns_empty(self, mock_get_embeddings):
        mock_get_embeddings.side_effect = RuntimeError("timeout")
        results = self.selector.select_chunks(self.text, "标题")
        self.assertEqual(results, [])

    @patch.object(ChunkSelector, "get_embeddings")
    def test_empty_vocabulary_returns_empty(self, mock_get_embeddings):
        mock_get_embeddings.return_value = np.random.rand(1, 4)
        text = "the and of\n\nand the of"
        results = self.selector.select_chunks(text, "")
        self.assertEqual(results, [])

    def test_embedding_strategy_injection(self):
        selector = ChunkSelector(
            embedding_strategy=_DummyEmbeddingStrategy(dim=8),
            chunk_num=1,
            config=SelectorConfig(cluster_ratio=1.0),
        )
        results = selector.select_chunks("第一段内容足够长。第二句。第三句。", "")
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]["embedding"]), 8)


if __name__ == "__main__":
    unittest.main()
