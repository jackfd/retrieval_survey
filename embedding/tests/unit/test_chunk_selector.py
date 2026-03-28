"""ChunkSelector unit tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest


def _load_chunk_splitter_module(monkeypatch: pytest.MonkeyPatch, offsets_fn=None):
    fake_blingfire = types.ModuleType("blingfire")

    def _default_offsets(text: str):
        return "", []

    fake_blingfire.text_to_sentences_and_offsets = offsets_fn or _default_offsets
    monkeypatch.setitem(sys.modules, "blingfire", fake_blingfire)

    splitter_path = Path(__file__).resolve().parents[2] / "services" / "chunk_splitter.py"
    splitter_spec = importlib.util.spec_from_file_location("chunk_splitter", splitter_path)
    splitter_module = importlib.util.module_from_spec(splitter_spec)
    assert splitter_spec.loader is not None
    splitter_spec.loader.exec_module(splitter_module)
    return splitter_module


def _load_chunk_selector_module(monkeypatch: pytest.MonkeyPatch, splitter_module):
    fake_embedding_pkg = types.ModuleType("embedding.infra.embedding_strategies")
    fake_embedding_pkg.EmbeddingStrategy = object
    monkeypatch.setitem(
        sys.modules, "embedding.infra.embedding_strategies", fake_embedding_pkg
    )
    monkeypatch.setitem(sys.modules, "chunk_splitter", splitter_module)

    selector_path = Path(__file__).resolve().parents[2] / "services" / "chunk_selector.py"
    selector_spec = importlib.util.spec_from_file_location(
        "embedding.services.chunk_selector", selector_path
    )
    selector_module = importlib.util.module_from_spec(selector_spec)
    assert selector_spec.loader is not None
    selector_spec.loader.exec_module(selector_module)
    return selector_module


def test_chunk_selector_run_returns_ranked_top_n_records(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    selector_module = _load_chunk_selector_module(monkeypatch, splitter_module)

    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    selector = selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.splitter.split_to_candidates = Mock(
        return_value=[
            {"order": 1, "text": "第一段"},
            {"order": 2, "text": "第二段"},
            {"order": 3, "text": "第三段"},
            {"order": 4, "text": "第四段"},
        ]
    )

    records = selector.run("ignored", "doc123")

    assert [item["chunk_rank"] for item in records] == [1, 2, 3]
    assert [item["chunk_id"] for item in records] == ["doc123#c002", "doc123#c003", "doc123#c001"]
    assert all(
        np.isclose(np.linalg.norm(np.asarray(item["chunk_vector"], dtype=float)), 1.0)
        for item in records
    )
    mock_strategy.encode.assert_called_once_with(
        ["第一段", "第二段", "第三段", "第四段"],
        is_query=False,
    )


def test_chunk_selector_embed_chunks_requires_2d(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    selector_module = _load_chunk_selector_module(monkeypatch, splitter_module)

    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array([1.0, 2.0], dtype=np.float32)
    selector = selector_module.ChunkSelector(embedding_strategy=mock_strategy)

    with pytest.raises(ValueError, match="2D"):
        selector._embed_chunks([{"order": 1, "text": "x"}])


def test_chunk_selector_is_better_candidate_tie_breaks_by_lower_index(
    monkeypatch: pytest.MonkeyPatch,
):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    selector_module = _load_chunk_selector_module(monkeypatch, splitter_module)

    selector = selector_module.ChunkSelector(embedding_strategy=Mock())

    assert selector._is_better_candidate(
        score=0.5,
        index=1,
        best_score=0.5,
        best_index=3,
    )
    assert not selector._is_better_candidate(
        score=0.5,
        index=4,
        best_score=0.5,
        best_index=3,
    )
