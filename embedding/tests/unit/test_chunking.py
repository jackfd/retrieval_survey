"""chunk 分割与选择逻辑的单元测试，覆盖段落切分、chunk 选择、聚类边界情况和 embedding 异常传播。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_chunk_splitter_module():
    splitter_path = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "chunking"
        / "chunk_splitter.py"
    )
    return _load_module_from_file("embedding.services.chunking.chunk_splitter", splitter_path)


class SelectorConfig:
    def __init__(self, **kwargs):
        self.cluster_ratio = 2.0
        self.batch_size = 64
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0
        self.top_keywords = 10
        self.cooccur_window = 4
        self.__dict__.update(kwargs)


def _load_chunk_selector_module():
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

    fake_stopwords_module = types.ModuleType(
        "embedding.services.chunking.stopwords_loader"
    )
    fake_stopwords_module.StopwordsLoader = Mock()
    fake_stopwords_module.StopwordsLoader.load_stopwords.return_value = set()

    fake_sklearn_cluster = types.ModuleType("sklearn.cluster")

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

    fake_sklearn_cluster.KMeans = FakeKMeans

    selector_path = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "chunking"
        / "chunk_selector.py"
    )
    selector_spec = importlib.util.spec_from_file_location(
        "embedding.services.chunking.chunk_selector", selector_path
    )
    selector_module = importlib.util.module_from_spec(selector_spec)
    assert selector_spec.loader is not None

    with patch.dict(
        sys.modules,
        {
            "embedding.infra.embedding_strategies": fake_embedding_pkg,
            "embedding.infra.embedding_strategies.base": base_module,
            "embedding.services.chunking.stopwords_loader": fake_stopwords_module,
            "sklearn.cluster": fake_sklearn_cluster,
        },
    ):
        selector_spec.loader.exec_module(selector_module)

    return selector_module


def test_chunk_splitter_basic_paragraph_split():
    splitter_module = _load_chunk_splitter_module()
    splitter = splitter_module.ChunkSplitter(min_sentences=2, max_tokens=100)

    paragraphs = splitter.split_paragraphs(
        "这是第一句。这是第二句。\n\n这是第三句。这是第四句。"
    )

    assert len(paragraphs) == 2
    assert all(paragraph.strip() for paragraph in paragraphs)


def test_chunk_selector_select_chunks_smoke():
    config = SelectorConfig(alpha=0.2, beta=0.2, gamma=3.0, top_keywords=5)
    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    chunk_selector_module = _load_chunk_selector_module()

    with patch.object(
        chunk_selector_module.StopwordsLoader,
        "load_stopwords",
        return_value=set(),
    ):
        selector = chunk_selector_module.ChunkSelector(
            embedding_strategy=mock_strategy,
            chunk_num=1,
            config=config,
        )
        selected = selector.select_chunks(
            "主题部分。主题部分。主题部分。\n\n其他部分。其他部分。其他部分。",
            title="主题部分",
        )

    assert len(selected) == 1
    assert selected[0]["chunk_text"].startswith("主题部分")
    assert isinstance(selected[0]["score"], float)
    assert selected[0]["embedding"] == [1.0, 0.0, 0.0, 0.0]
    mock_strategy.encode.assert_called_once_with(
        ["主题部分。主题部分。主题部分。", "其他部分。其他部分。其他部分。"],
        is_query=False,
    )


def test_chunk_selector_cluster_chunks_single_embedding_boundary():
    chunk_selector_module = _load_chunk_selector_module()
    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch.object(
        chunk_selector_module.StopwordsLoader,
        "load_stopwords",
        return_value=set(),
    ):
        selector = chunk_selector_module.ChunkSelector(
            embedding_strategy=mock_strategy,
            chunk_num=5,
            config=SelectorConfig(),
        )

    candidate_idxs = selector.cluster_chunks(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    assert candidate_idxs == [0]


def test_chunk_selector_propagates_embedding_strategy_error():
    chunk_selector_module = _load_chunk_selector_module()
    mock_strategy = Mock()
    mock_strategy.encode.side_effect = RuntimeError("boom")

    with patch.object(
        chunk_selector_module.StopwordsLoader,
        "load_stopwords",
        return_value=set(),
    ):
        selector = chunk_selector_module.ChunkSelector(
            embedding_strategy=mock_strategy,
            chunk_num=1,
            config=SelectorConfig(),
        )

    with pytest.raises(RuntimeError):
        selector.select_chunks("主题部分。主题部分。主题部分。")
