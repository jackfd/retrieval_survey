import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from embed_pipe.domain.models import RuntimeConfig
from embed_pipe.services.document_service import DocumentService
from embed_pipe.services.query_service import QueryService


class _DummyEmbeddingStrategy:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, is_query):
        vectors = []
        for text in texts:
            value = self.mapping.get(text)
            if isinstance(value, Exception):
                raise value
            vectors.append(value)
        return np.asarray(vectors, dtype=np.float32)


class TestProcessors(unittest.TestCase):
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
            embedding_api_url="",
            http_timeout=5.0,
            http_max_retries=1,
        )

    def _write_jsonl(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
        path = Path(tmp.name)
        with path.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(f"{row}\n")
        return path

    def test_process_docs_missing_required_fails_fast(self):
        docs_path = self._write_jsonl(
            [
                '{"doc_id": "", "doc_text": "abc"}',
                '{"doc_id": "d2", "doc_text": ""}',
            ]
        )
        selector = Mock()

        result = DocumentService(selector=selector, runtime=self.runtime).process(
            docs_path=docs_path,
            retry_mode=False,
            retry_id_set=set(),
        )
        self.assertFalse(result.ok)

    def test_process_docs_raises_when_selector_raises(self):
        docs_path = self._write_jsonl(['{"doc_id": "d1", "doc_text": "text"}'])
        selector = Mock()
        selector.select_chunks.side_effect = RuntimeError("No chunk selected")

        with self.assertRaises(RuntimeError):
            DocumentService(selector=selector, runtime=self.runtime).process(
                docs_path=docs_path,
                retry_mode=False,
                retry_id_set=set(),
            )

    def test_process_queries_records_embedding_failure(self):
        queries_path = self._write_jsonl(
            ['{"query_id": "q1", "query_text": "bad"}', '{"query_id": "q2", "query_text": "good"}']
        )
        embedding_strategy = _DummyEmbeddingStrategy(
            {
                "bad": RuntimeError("boom"),
                "good": [0.1, 0.2, 0.3],
            }
        )

        result = QueryService(embedding_strategy=embedding_strategy, runtime=self.runtime).process(
            queries_path=queries_path,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.value.attempted_count, 2)
        self.assertEqual(len(result.value.output_df), 1)
        self.assertEqual(len(result.value.failures), 1)
        self.assertEqual(result.value.failures[0]["record_type"], "query")
        self.assertEqual(result.value.failures[0]["stage"], "embedding")


if __name__ == "__main__":
    unittest.main()
