"""文档和查询服务的单元测试，覆盖正常处理、空字段校验和下游异常包装。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest

from embedding.domain.exceptions import ProcessingError


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_service_modules():
    base_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "base.py"
    )
    base_module = _load_module_from_file(
        "embedding.infra.embedding_strategies.base", base_path
    )

    fake_embedding_pkg = types.ModuleType("embedding.infra.embedding_strategies")
    fake_embedding_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "infra" / "embedding_strategies")
    ]
    fake_embedding_pkg.EmbeddingStrategy = base_module.EmbeddingStrategy
    fake_embedding_pkg.BaseEmbeddingStrategy = base_module.BaseEmbeddingStrategy

    fake_chunk_selector_module = types.ModuleType(
        "embedding.services.chunking.chunk_selector"
    )
    fake_chunk_selector_module.ChunkSelector = object

    fake_chunking_pkg = types.ModuleType("embedding.services.chunking")
    fake_chunking_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "services" / "chunking")
    ]

    document_path = Path(__file__).resolve().parents[2] / "services" / "document_service.py"
    query_path = Path(__file__).resolve().parents[2] / "services" / "query_service.py"

    with patch.dict(
        sys.modules,
        {
            "embedding.infra.embedding_strategies": fake_embedding_pkg,
            "embedding.infra.embedding_strategies.base": base_module,
            "embedding.services.chunking": fake_chunking_pkg,
            "embedding.services.chunking.chunk_selector": fake_chunk_selector_module,
        },
    ):
        document_module = _load_module_from_file(
            "embedding.services.document_service", document_path
        )
        query_module = _load_module_from_file(
            "embedding.services.query_service", query_path
        )

    return document_module, query_module


def test_document_service_process_returns_selected_chunk():
    document_module, _ = _load_service_modules()
    selector = Mock()
    selector.run.return_value = [
        {
            "doc_id": "d1",
            "chunk_id": "d1#c001",
            "chunk_text": "主题部分。主题部分。主题部分。",
            "chunk_vector": [1.0, 0.0, 0.0, 0.0],
            "chunk_score": 0.9,
            "chunk_rank": 1,
        }
    ]
    service = document_module.DocumentService(selector=selector)
    service.jsonl_reader = Mock(
        read_objects=Mock(
            return_value=[
                (
                    1,
                    {
                        "doc_id": "d1",
                        "doc_text": "主题部分。主题部分。主题部分。",
                    },
                )
            ]
        )
    )

    result = service.process(Path("docs.jsonl"))

    assert list(result.output_df["doc_id"]) == ["d1"]
    assert list(result.output_df["chunk_id"]) == ["d1#c001"]
    assert list(result.output_df["chunk_text"]) == ["主题部分。主题部分。主题部分。"]
    assert list(result.output_df["chunk_vector"]) == [[1.0, 0.0, 0.0, 0.0]]
    assert list(result.output_df["chunk_score"]) == [0.9]
    assert list(result.output_df["chunk_rank"]) == [1]
    selector.run.assert_called_once_with("主题部分。主题部分。主题部分。", "d1")


def test_document_service_rejects_empty_doc_fields():
    document_module, _ = _load_service_modules()
    selector = Mock()
    service = document_module.DocumentService(selector=selector)
    service.jsonl_reader = Mock(
        read_objects=Mock(
            return_value=[(1, {"doc_id": "", "doc_text": "some text"})]
        )
    )

    with pytest.raises(ProcessingError):
        service.process(Path("docs.jsonl"))


def test_document_service_wraps_selector_errors():
    document_module, _ = _load_service_modules()
    selector = Mock()
    selector.run.side_effect = RuntimeError("boom")
    service = document_module.DocumentService(selector=selector)
    service.jsonl_reader = Mock(
        read_objects=Mock(
            return_value=[
                (
                    1,
                    {
                        "doc_id": "d1",
                        "doc_text": "主题部分。主题部分。主题部分。",
                    },
                )
            ]
        )
    )

    with pytest.raises(ProcessingError):
        service.process(Path("docs.jsonl"))


def test_query_service_process_encodes_queries():
    _, query_module = _load_service_modules()
    embedding_strategy = Mock()
    embedding_strategy.encode.return_value = np.array(
        [[0.1, 0.2, 0.3, 0.4]], dtype=np.float32
    )
    service = query_module.QueryService(embedding_strategy=embedding_strategy)
    service.jsonl_reader = Mock(
        read_objects=Mock(
            return_value=[(1, {"query_id": "q1", "query_text": "主题查询"})]
        )
    )

    result = service.process(Path("queries.jsonl"))

    assert list(result.output_df["query_id"]) == ["q1"]
    assert list(result.output_df["query_text"]) == ["主题查询"]
    assert np.allclose(
        np.asarray(result.output_df["query_embedding"].iloc[0], dtype=float),
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
    )
    embedding_strategy.encode.assert_called_once_with(["主题查询"], is_query=True)


def test_query_service_rejects_empty_query_fields():
    _, query_module = _load_service_modules()
    embedding_strategy = Mock()
    service = query_module.QueryService(embedding_strategy=embedding_strategy)
    service.jsonl_reader = Mock(
        read_objects=Mock(
            return_value=[(1, {"query_id": "q1", "query_text": ""})]
        )
    )

    with pytest.raises(ProcessingError):
        service.process(Path("queries.jsonl"))


def test_query_service_wraps_encoder_errors():
    _, query_module = _load_service_modules()
    embedding_strategy = Mock()
    embedding_strategy.encode.side_effect = RuntimeError("boom")
    service = query_module.QueryService(embedding_strategy=embedding_strategy)
    service.jsonl_reader = Mock(
        read_objects=Mock(
            return_value=[(1, {"query_id": "q1", "query_text": "主题查询"})]
        )
    )

    with pytest.raises(ProcessingError):
        service.process(Path("queries.jsonl"))
