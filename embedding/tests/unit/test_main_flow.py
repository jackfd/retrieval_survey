"""运行器元数据构建的单元测试，覆盖 run metadata 的时间、模型信息和统计字段生成。"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

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

    fake_chunking_pkg = types.ModuleType("embedding.services.chunking")
    fake_chunking_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "services" / "chunking")
    ]
    fake_chunking_pkg.ChunkSelector = object
    fake_chunking_pkg.SelectorConfig = object

    fake_document_module = types.ModuleType("embedding.services.document_service")
    fake_document_module.DocumentService = object

    fake_query_module = types.ModuleType("embedding.services.query_service")
    fake_query_module.QueryService = object

    for module_name in [
        "embedding.app.runner",
        "embedding.services.chunking",
        "embedding.services.chunking.chunk_selector",
        "embedding.services.chunking.stopwords_loader",
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
            "embedding.services.chunking": fake_chunking_pkg,
            "embedding.services.document_service": fake_document_module,
            "embedding.services.query_service": fake_query_module,
        },
    ):
        return importlib.import_module("embedding.app.runner")


def test_build_run_metadata_uses_dataset_and_model_context(tmp_path):
    runner_module = _load_runner_module()
    dataset_ctx = make_dataset_context(
        root=tmp_path / "dataset",
        docs_rows=[],
        queries_rows=[],
        dataset_meta={"name": "demo"},
    )
    builder_cfg = make_builder_config(
        experiment=make_experiment_config(),
        inference=make_inference_config(),
        model=make_model_config(provider="test", model_id="test-model"),
    )

    with patch.object(
        runner_module,
        "utc_now_iso",
        return_value="2026-03-27T00:00:01Z",
    ):
        metadata = runner_module.build_run_metadata(
            run_start="2026-03-27T00:00:00Z",
            dataset_ctx=dataset_ctx,
            builder_cfg=builder_cfg,
            merged_docs_df=pd.DataFrame([{"doc_id": "d1"}]),
            query_df=pd.DataFrame([{"query_id": "q1"}]),
        )

    assert metadata["run"]["start_time_utc"] == "2026-03-27T00:00:00Z"
    assert metadata["run"]["end_time_utc"] == "2026-03-27T00:00:01Z"
    assert metadata["run"]["resolved_dataset_dir"] == str(tmp_path / "dataset")
    assert metadata["model"] == {
        "provider": "test",
        "model_id": "test-model",
    }
    assert metadata["stats"] == {"doc_count": 1, "query_count": 1}


def test_build_run_metadata_handles_empty_inputs(tmp_path):
    runner_module = _load_runner_module()
    dataset_ctx = make_dataset_context(
        root=tmp_path / "dataset",
        docs_rows=[],
        queries_rows=[],
    )
    builder_cfg = make_builder_config()

    with patch.object(
        runner_module,
        "utc_now_iso",
        return_value="2026-03-27T00:00:01Z",
    ):
        metadata = runner_module.build_run_metadata(
            run_start="2026-03-27T00:00:00Z",
            dataset_ctx=dataset_ctx,
            builder_cfg=builder_cfg,
            merged_docs_df=pd.DataFrame(columns=["doc_id"]),
            query_df=pd.DataFrame(columns=["query_id"]),
        )

    assert metadata["stats"] == {"doc_count": 0, "query_count": 0}
