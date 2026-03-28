"""chunk 选择逻辑单元测试，覆盖 MMR 排序、稳定合并和异常传播。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SelectorConfig:
    def __init__(self, **kwargs):
        self.batch_size = 64
        self.hard_max_tokens = 8092
        self.target_tokens = 3200
        self.min_independent_tokens = 500
        self.top_n = 3
        self.mmr_lambda = 0.7
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

    selector_path = Path(__file__).resolve().parents[2] / "services" / "chunk_selector.py"
    selector_spec = importlib.util.spec_from_file_location(
        "embedding.services.chunk_selector", selector_path
    )
    selector_module = importlib.util.module_from_spec(selector_spec)
    assert selector_spec.loader is not None

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "embedding.infra.embedding_strategies", fake_embedding_pkg)
        mp.setitem(sys.modules, "embedding.infra.embedding_strategies.base", base_module)
        selector_spec.loader.exec_module(selector_module)

    return selector_module


def test_chunk_selector_run_returns_ranked_top3_with_normalized_vectors():
    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(
        top_n=3,
        min_independent_tokens=1,
        target_tokens=100,
    )

    selected = selector.run(
        "第一段内容。\n\n第二段内容。\n\n第三段内容。\n\n第四段内容。",
        "doc123",
    )

    assert [item["chunk_rank"] for item in selected] == [1, 2, 3]
    assert [item["chunk_id"] for item in selected] == [
        "doc123#c002",
        "doc123#c003",
        "doc123#c001",
    ]
    assert all(
        np.isclose(np.linalg.norm(np.asarray(item["chunk_vector"], dtype=float)), 1.0)
        for item in selected
    )
    mock_strategy.encode.assert_called_once_with(
        ["第一段内容。", "第二段内容。", "第三段内容。", "第四段内容。"],
        is_query=False,
    )


def test_chunk_selector_merges_small_chunk_with_better_neighbor():
    mock_strategy = Mock()
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(target_tokens=15, min_independent_tokens=5)

    merged = selector._merge_small_chunks(
        [
            "甲甲甲甲甲甲甲甲甲甲",
            "乙乙",
            "丙丙丙丙丙丙丙丙丙丙丙丙丙丙",
        ]
    )

    assert merged == ["甲甲甲甲甲甲甲甲甲甲\n\n乙乙", "丙丙丙丙丙丙丙丙丙丙丙丙丙丙"]


def test_chunk_selector_returns_all_candidates_when_count_not_exceeding_top_n():
    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(
        top_n=3,
        min_independent_tokens=1,
        target_tokens=100,
    )

    selected = selector.run("第一段。\n\n第二段。", "doc1")

    assert len(selected) == 2
    assert [item["chunk_rank"] for item in selected] == [1, 2]
    assert {item["chunk_id"] for item in selected} == {"doc1#c001", "doc1#c002"}


def test_chunk_selector_propagates_embedding_strategy_error():
    chunk_selector_module = _load_chunk_selector_module()
    mock_strategy = Mock()
    mock_strategy.encode.side_effect = RuntimeError("boom")
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(min_independent_tokens=1, target_tokens=100)

    with pytest.raises(RuntimeError):
        selector.run("主题部分。主题部分。主题部分。", "doc1")
