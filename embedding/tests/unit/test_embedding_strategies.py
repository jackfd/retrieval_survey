"""Unit tests for base embedding text preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock


def _load_base_module():
    base_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "base.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_embedding_base_test", base_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_strategy(
    *,
    query_prefix: str = "Q: ",
    doc_prefix: str = "D: ",
    instruction_template: str = "",
):
    base_module = _load_base_module()
    BaseEmbeddingStrategy = base_module.BaseEmbeddingStrategy

    experiment_cfg = Mock()
    experiment_cfg.embedding_dim = 384
    experiment_cfg.normalize_embeddings = False
    experiment_cfg.max_length = 512
    experiment_cfg.query_prefix = query_prefix
    experiment_cfg.doc_prefix = doc_prefix
    experiment_cfg.instruction_template = instruction_template

    inference_cfg = Mock()
    inference_cfg.batch_size = 16
    inference_cfg.device = "cpu"
    inference_cfg.embedding_api_url = ""
    inference_cfg.http_timeout = 10
    inference_cfg.http_max_retries = 3

    return BaseEmbeddingStrategy(experiment=experiment_cfg, inference=inference_cfg)


def test_prepare_texts_applies_query_and_doc_prefixes():
    strategy = _make_strategy()

    assert strategy._prepare_texts(["查询"], is_query=True) == ["Q: 查询"]
    assert strategy._prepare_texts(["文档"], is_query=False) == ["D: 文档"]


def test_prepare_texts_supports_instruction_template_placeholder():
    strategy = _make_strategy(instruction_template="Encode for retrieval: {text}")

    assert strategy._prepare_texts(["查询"], is_query=True) == [
        "Encode for retrieval: Q: 查询"
    ]
    assert strategy._prepare_texts(["文档"], is_query=False) == [
        "Encode for retrieval: D: 文档"
    ]


def test_prepare_texts_supports_instruction_template_prefix_mode():
    strategy = _make_strategy(instruction_template="Encode for retrieval: ")

    assert strategy._prepare_texts(["查询"], is_query=True) == [
        "Encode for retrieval: Q: 查询"
    ]
    assert strategy._prepare_texts(["文档"], is_query=False) == [
        "Encode for retrieval: D: 文档"
    ]
