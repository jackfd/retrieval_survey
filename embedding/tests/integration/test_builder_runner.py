"""Integration tests for BuilderRunner and the main embedding flow."""

from __future__ import annotations

import importlib
import importlib.util
import json
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


def _build_fake_stopwords_module():
    fake_module = types.ModuleType("embedding.services.chunking.stopwords_loader")

    class _Loader:
        @staticmethod
        def load_stopwords():
            return set()

    fake_module.StopwordsLoader = _Loader
    return fake_module


def _build_fake_sklearn_cluster_module():
    fake_module = types.ModuleType("sklearn.cluster")

    class FakeKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=None, max_iter=None):
            self.n_clusters = n_clusters
            self.labels_ = None
            self.cluster_centers_ = None

        def fit(self, embeddings):
            self.labels_ = np.arange(len(embeddings)) % self.n_clusters
            self.cluster_centers_ = np.asarray(
                embeddings[: self.n_clusters], dtype=np.float32
            )
            return self

    fake_module.KMeans = FakeKMeans
    return fake_module


def test_builder_runner_run_writes_outputs_and_metadata(tmp_path):
    dataset_ctx = make_dataset_context(
        root=tmp_path / "dataset",
        docs_rows=[
            {
                "doc_id": "d1",
                "doc_text": "主题部分。主题部分。主题部分。\n\n其他部分。其他部分。其他部分。",
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
    embedding_strategy = ScriptedEmbeddingStrategy()

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "doc_id": "d1",
                "chunk_text": "old chunk",
                "chunk_embedding": [9.0, 9.0, 9.0, 9.0],
            }
        ]
    ).to_parquet(output_dir / "docs.parquet", index=False)

    base_module = _load_base_module()
    fake_embedding_pkg = _build_fake_embedding_package(base_module)
    fake_stopwords_module = _build_fake_stopwords_module()
    fake_sklearn_cluster = _build_fake_sklearn_cluster_module()

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
            "embedding.services.chunking.stopwords_loader": fake_stopwords_module,
            "sklearn.cluster": fake_sklearn_cluster,
        },
    ):
        runner_module = importlib.import_module("embedding.app.runner")

        with patch.object(
            runner_module,
            "utc_now_iso",
            side_effect=["2026-03-27T00:00:00Z", "2026-03-27T00:00:01Z"],
        ):
            runner = runner_module.BuilderRunner(
                output_dir=output_dir,
                dataset_ctx=dataset_ctx,
                builder_cfg=builder_cfg,
                embedding_strategy=embedding_strategy,
            )
            result = runner.run()

    assert result is None
    assert embedding_strategy.calls == [
        (["主题部分。主题部分。主题部分。", "其他部分。其他部分。其他部分。"], False),
        (["主题查询"], True),
    ]

    docs_df = pd.read_parquet(output_dir / "docs.parquet")
    queries_df = pd.read_parquet(output_dir / "queries.parquet")
    with (output_dir / "run_metadata.json").open("r", encoding="utf-8") as fin:
        metadata = json.load(fin)

    assert len(docs_df) == 1
    assert list(docs_df["doc_id"]) == ["d1"]
    assert docs_df.iloc[0]["chunk_text"] in {
        "主题部分。主题部分。主题部分。",
        "其他部分。其他部分。其他部分。",
    }
    assert list(queries_df["query_id"]) == ["q1"]
    assert np.allclose(
        np.asarray(queries_df.iloc[0]["query_embedding"], dtype=float),
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
    )
    assert metadata["run"]["start_time_utc"] == "2026-03-27T00:00:00Z"
    assert metadata["run"]["end_time_utc"] == "2026-03-27T00:00:01Z"
    assert metadata["run"]["resolved_dataset_dir"] == str(tmp_path / "dataset")
    assert metadata["model"] == {
        "provider": "test",
        "model_id": "test-model",
    }
    assert metadata["stats"] == {"doc_count": 1, "query_count": 1}
