import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from index_builder.config import RuntimeConfig
from index_builder.errors import ChunkSelectionError, EmbeddingGenerationError
from index_builder.processors import process_docs, process_queries


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
        self.logger = Mock()

    def _write_jsonl(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
        path = Path(tmp.name)
        with path.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(f"{row}\n")
        return path

    def test_process_docs_records_missing_required_as_failure(self):
        docs_path = self._write_jsonl(
            [
                '{"doc_id": "", "doc_text": "abc"}',
                '{"doc_id": "d2", "doc_text": ""}',
            ]
        )
        selector = Mock()

        result = process_docs(
            docs_path=docs_path,
            selector=selector,
            runtime=self.runtime,
            retry_mode=False,
            retry_id_set=set(),
            logger=self.logger,
        )

        self.assertEqual(result.attempted_count, 0)
        self.assertEqual(result.skipped_missing_required_count, 2)
        self.assertEqual(len(result.output_df), 0)
        self.assertEqual(len(result.failures), 2)
        self.assertTrue(all(item["record_type"] == "doc" for item in result.failures))
        self.assertTrue(all(item["stage"] == "input_validation" for item in result.failures))

    def test_process_docs_chunk_selection_failure_stage(self):
        docs_path = self._write_jsonl(['{"doc_id": "d1", "doc_text": "text"}'])
        selector = Mock()
        selector.select_chunks.side_effect = ChunkSelectionError("No chunk selected")

        result = process_docs(
            docs_path=docs_path,
            selector=selector,
            runtime=self.runtime,
            retry_mode=False,
            retry_id_set=set(),
            logger=self.logger,
        )

        self.assertEqual(result.attempted_count, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0]["record_type"], "doc")
        self.assertEqual(result.failures[0]["stage"], "chunk_selection")

    def test_process_queries_records_embedding_failure(self):
        queries_path = self._write_jsonl(
            ['{"query_id": "q1", "query_text": "bad"}', '{"query_id": "q2", "query_text": "good"}']
        )
        embedding_strategy = _DummyEmbeddingStrategy(
            {
                "bad": EmbeddingGenerationError("boom"),
                "good": [0.1, 0.2, 0.3],
            }
        )

        result = process_queries(
            queries_path=queries_path,
            embedding_strategy=embedding_strategy,
            runtime=self.runtime,
            logger=self.logger,
        )

        self.assertEqual(result.attempted_count, 2)
        self.assertEqual(len(result.output_df), 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0]["record_type"], "query")
        self.assertEqual(result.failures[0]["stage"], "embedding")


if __name__ == "__main__":
    unittest.main()
