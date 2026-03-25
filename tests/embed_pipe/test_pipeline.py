import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from embed_pipe.app.runner import BuilderRunner
from embed_pipe.domain.models import BuilderConfig, DatasetContext, ModelConfig, ProcessResult, RuntimeConfig


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
                        "error_message": "bad",
                        "timestamp_utc": "2026-01-01T00:00:01Z",
                        "stage": "embedding",
                    }
                ],
                attempted_count=1,
            )

            with patch("embed_pipe.app.runner.ChunkSelector", return_value=Mock()):
                with patch("embed_pipe.app.runner.DocumentService.process") as mock_doc_process:
                    with patch("embed_pipe.app.runner.QueryService.process") as mock_query_process:
                        with patch(
                            "embed_pipe.app.runner.OutputWriter.write_all_outputs", return_value=None
                        ) as mock_write_all:
                            mock_doc_process.return_value = Mock(ok=True, value=doc_result, error_message="")
                            mock_query_process.return_value = Mock(ok=True, value=query_result, error_message="")
                            runner = BuilderRunner(
                                output_dir=root / "out" / "ds" / "m",
                                dataset_ctx=dataset_ctx,
                                builder_cfg=builder_cfg,
                                embedding_strategy=Mock(),
                            )
                            result = runner.run()
                            self.assertTrue(result.ok)
                            mock_write_all.assert_called_once()
                            kwargs = mock_write_all.call_args.kwargs
            lines = kwargs["failures"]
            self.assertEqual(len(lines), 2)
            self.assertEqual([item["record_type"] for item in lines], ["doc", "query"])

            metadata = kwargs["metadata"]
            stats = metadata["stats"]
            self.assertEqual(stats["attempted_doc_count"], 1)
            self.assertEqual(stats["attempted_query_count"], 1)
            self.assertEqual(stats["doc_failure_count"], 1)
            self.assertEqual(stats["query_failure_count"], 1)
            self.assertEqual(stats["failure_count"], 2)

    def test_run_builder_returns_failed_result_when_write_all_outputs_raises(self):
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
                failures=[],
                attempted_count=1,
            )
            query_result = ProcessResult(
                output_df=pd.DataFrame(
                    [{"query_id": "q1", "query_text": "qt", "query_embedding": [0.1, 0.2, 0.3]}]
                ),
                failures=[],
                attempted_count=1,
            )

            with patch("embed_pipe.app.runner.ChunkSelector", return_value=Mock()):
                with patch("embed_pipe.app.runner.DocumentService.process") as mock_doc_process:
                    with patch("embed_pipe.app.runner.QueryService.process") as mock_query_process:
                        with patch(
                            "embed_pipe.app.runner.OutputWriter.write_all_outputs",
                            side_effect=RuntimeError("disk full"),
                        ):
                            mock_doc_process.return_value = Mock(
                                ok=True, value=doc_result, error_message=""
                            )
                            mock_query_process.return_value = Mock(
                                ok=True, value=query_result, error_message=""
                            )
                            runner = BuilderRunner(
                                output_dir=root / "out" / "ds" / "m",
                                dataset_ctx=dataset_ctx,
                                builder_cfg=builder_cfg,
                                embedding_strategy=Mock(),
                            )
                            result = runner.run()
                            self.assertFalse(result.ok)
                            self.assertIn("disk full", result.error_message)


if __name__ == "__main__":
    unittest.main()
