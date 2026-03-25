import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from index_builder.config import BuilderConfig, ModelConfig, RuntimeConfig
from index_builder.dataset import DatasetContext
from index_builder.pipeline import run_builder
from index_builder.processors import ProcessResult


class TestPipeline(unittest.TestCase):
    def test_run_builder_writes_merged_typed_failures_and_metadata(self):
        runtime = RuntimeConfig(
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
        builder_cfg = BuilderConfig(
            runtime=runtime,
            model=ModelConfig(model_name="m", provider="sentence_transformers", model_id="id"),
            raw_config={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_path = root / "docs.jsonl"
            queries_path = root / "queries.jsonl"
            docs_path.write_text("", encoding="utf-8")
            queries_path.write_text("", encoding="utf-8")

            dataset_ctx = DatasetContext(
                dataset_root=root,
                resolved_dataset_dir=root,
                dataset_meta={},
                docs_path=docs_path,
                queries_path=queries_path,
            )

            doc_result = ProcessResult(
                output_df=pd.DataFrame(
                    [{"doc_id": "d1", "chunk_text": "t", "chunk_embedding": [0.1, 0.2, 0.3]}]
                ),
                failures=[
                    {
                        "record_type": "doc",
                        "record_id": "d_fail",
                        "error_type": "ChunkSelectionError",
                        "error_message": "bad",
                        "timestamp_utc": "2026-01-01T00:00:00Z",
                        "stage": "chunk_selection",
                    }
                ],
                attempted_count=1,
            )
            query_result = ProcessResult(
                output_df=pd.DataFrame(
                    [{"query_id": "q1", "query_text": "qt", "query_embedding": [0.1, 0.2, 0.3]}]
                ),
                failures=[
                    {
                        "record_type": "query",
                        "record_id": "q_fail",
                        "error_type": "EmbeddingGenerationError",
                        "error_message": "bad",
                        "timestamp_utc": "2026-01-01T00:00:01Z",
                        "stage": "embedding",
                    }
                ],
                attempted_count=1,
            )

            with patch("index_builder.pipeline.ChunkSelector", return_value=Mock()):
                with patch("index_builder.pipeline.process_docs", return_value=doc_result):
                    with patch("index_builder.pipeline.process_queries", return_value=query_result):
                        with patch("pandas.DataFrame.to_parquet", return_value=None):
                            run_builder(
                                dataset_name="ds",
                                output_dir=root / "out" / "ds" / "m",
                                dataset_ctx=dataset_ctx,
                                builder_cfg=builder_cfg,
                                embedding_strategy=Mock(),
                                logger=Mock(),
                            )

            output_dir = root / "out" / "ds" / "m"
            failures_path = output_dir / "failures.jsonl"
            metadata_path = output_dir / "run_metadata.json"

            lines = [
                json.loads(line)
                for line in failures_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 2)
            self.assertEqual([item["record_type"] for item in lines], ["doc", "query"])

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stats = metadata["stats"]
            self.assertEqual(stats["attempted_doc_count"], 1)
            self.assertEqual(stats["attempted_query_count"], 1)
            self.assertEqual(stats["doc_failure_count"], 1)
            self.assertEqual(stats["query_failure_count"], 1)
            self.assertEqual(stats["failure_count"], 2)


if __name__ == "__main__":
    unittest.main()
