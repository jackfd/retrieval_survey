"""Integration tests for BuilderRunner and the main embedding flow."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from embedding.tests.support import (
    ScriptedEmbeddingStrategy,
    make_builder_config,
    make_dataset_context,
    make_experiment_config,
    make_inference_config,
    make_model_config,
)

pytestmark = pytest.mark.integration


def _load_base_module():
    base_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "base.py"
    )
    spec = importlib.util.spec_from_file_location(
        "embedding.infra.embedding_strategies.base",
        base_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_fake_embedding_package(base_module):
    fake_pkg = types.ModuleType("embedding.infra.embedding_strategies")
    fake_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "infra" / "embedding_strategies")
    ]
    fake_pkg.BaseEmbeddingStrategy = base_module.BaseEmbeddingStrategy
    fake_pkg.EmbeddingStrategy = base_module.EmbeddingStrategy
    return fake_pkg


def test_builder_runner_run_rewrites_outputs_without_metadata(tmp_path, caplog):
    first_chunk = "主题部分" * 260 + "。"
    second_chunk = "其他部分" * 260 + "。"
    dataset_ctx = make_dataset_context(
        root=tmp_path / "dataset",
        docs_rows=[
            {
                "doc_id": "d1",
                "doc_text": f"{first_chunk}\n\n{second_chunk}",
            }
        ],
        queries_rows=[
            {
                "query_id": "q1",
                "query_text": "主题查询",
            }
        ],
    )
    builder_cfg = make_builder_config(
        experiment=make_experiment_config(embedding_dim=4),
        inference=make_inference_config(batch_size=2),
        model=make_model_config(provider="test", model_id="test-model"),
    )
    embedding_strategy = ScriptedEmbeddingStrategy(
        doc_vectors=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "doc_id": "stale",
                "chunk_id": "stale#c001",
                "chunk_text": "old chunk",
                "chunk_vector": [9.0, 9.0, 9.0, 9.0],
                "chunk_score": 9.0,
                "chunk_rank": 1,
            }
        ]
    ).to_parquet(output_dir / "docs_dim4.parquet", index=False)

    base_module = _load_base_module()
    fake_embedding_pkg = _build_fake_embedding_package(base_module)

    for module_name in [
        "embedding.app.runner",
        "embedding.services",
        "embedding.services.chunk_selector",
        "embedding.services.chunk_splitter",
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
        runner_module = importlib.import_module("embedding.app.runner")

        with patch.object(
            runner_module,
            "utc_now_iso",
            side_effect=["2026-03-27T00:00:00Z", "2026-03-27T00:00:01Z"],
        ):
            caplog.set_level("INFO", logger="embedding.app.runner")
            runner = runner_module.BuilderRunner(
                output_dir=output_dir,
                dataset_ctx=dataset_ctx,
                builder_cfg=builder_cfg,
                embedding_strategy=embedding_strategy,
            )
            result = runner.run()

    assert result is None
    assert embedding_strategy.calls == [
        ([first_chunk, second_chunk], False),
        (["主题查询"], True),
    ]

    docs_df = pd.read_parquet(output_dir / "docs_dim4.parquet")
    queries_df = pd.read_parquet(output_dir / "queries_dim4.parquet")

    assert len(docs_df) == 2
    assert set(docs_df["doc_id"]) == {"d1"}
    assert set(docs_df["chunk_id"]) == {"d1#c001", "d1#c002"}
    assert "stale" not in set(docs_df["doc_id"])
    assert list(queries_df["query_id"]) == ["q1"]
    assert np.allclose(
        np.asarray(queries_df.iloc[0]["query_embedding"], dtype=float),
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
    )
    assert not (output_dir / "run_metadata.json").exists()
    assert "start: model_id=test-model" in caplog.text
    assert (
        "end: start_time_utc=2026-03-27T00:00:00Z end_time_utc=2026-03-27T00:00:01Z"
        in caplog.text
    )
    assert "doc_count=1 chunk_count=2 query_count=1" in caplog.text
