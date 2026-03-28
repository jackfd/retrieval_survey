"""运行器流程单元测试，覆盖日志摘要和空输出统计。"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from embedding.tests.support import (
    make_builder_config,
    make_dataset_context,
    make_experiment_config,
    make_inference_config,
    make_model_config,
)


def _load_runner_module():
    base_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "base.py"
    )
    base_spec = importlib.util.spec_from_file_location(
        "embedding.infra.embedding_strategies.base", base_path
    )
    base_module = importlib.util.module_from_spec(base_spec)
    assert base_spec.loader is not None
    base_spec.loader.exec_module(base_module)

    fake_embedding_pkg = types.ModuleType("embedding.infra.embedding_strategies")
    fake_embedding_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "infra" / "embedding_strategies")
    ]
    fake_embedding_pkg.BaseEmbeddingStrategy = base_module.BaseEmbeddingStrategy
    fake_embedding_pkg.EmbeddingStrategy = base_module.EmbeddingStrategy

    for module_name in [
        "embedding.app.runner",
        "embedding.services.chunking",
        "embedding.services.chunking.chunk_selector",
        "embedding.services.document_service",
        "embedding.services.query_service",
        "embedding.infra.embedding_strategies",
    ]:
        sys.modules.pop(module_name, None)

    with patch.dict(
        sys.modules,
        {
            "embedding.infra.embedding_strategies": fake_embedding_pkg,
            "embedding.infra.embedding_strategies.base": base_module,
        },
    ):
        return importlib.import_module("embedding.app.runner")


def test_builder_runner_logs_counts_for_empty_outputs(tmp_path, caplog):
    runner_module = _load_runner_module()
    dataset_ctx = make_dataset_context(
        root=tmp_path / "dataset",
        docs_rows=[],
        queries_rows=[],
    )
    builder_cfg = make_builder_config(
        experiment=make_experiment_config(),
        inference=make_inference_config(),
        model=make_model_config(provider="test", model_id="test-model"),
    )

    empty_docs = pd.DataFrame(
        columns=[
            "doc_id",
            "chunk_id",
            "chunk_text",
            "chunk_vector",
            "chunk_score",
            "chunk_rank",
        ]
    )
    empty_queries = pd.DataFrame(columns=["query_id", "query_text", "query_embedding"])

    with patch.object(runner_module, "ChunkSelector", return_value=Mock()) as selector_cls, patch.object(
        runner_module, "DocumentService"
    ) as document_service_cls, patch.object(
        runner_module, "QueryService"
    ) as query_service_cls, patch.object(
        runner_module, "utc_now_iso", side_effect=["2026-03-27T00:00:00Z", "2026-03-27T00:00:01Z"]
    ):
        document_service_cls.return_value.process.return_value.output_df = empty_docs
        query_service_cls.return_value.process.return_value.output_df = empty_queries
        caplog.set_level("INFO", logger="embedding")

        runner = runner_module.BuilderRunner(
            output_dir=tmp_path / "output",
            dataset_ctx=dataset_ctx,
            builder_cfg=builder_cfg,
            embedding_strategy=Mock(),
        )
        runner.run()

    selector_cls.assert_called_once()
    assert "start: model_id=test-model" in caplog.text
    assert "end: model_id=test-model" in caplog.text
    assert "doc_count=0 chunk_count=0 query_count=0" in caplog.text
