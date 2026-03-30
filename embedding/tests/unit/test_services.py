"""Service-layer unit tests for function-based document/query processing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from embedding.domain.exceptions import ProcessingError
from embedding.domain.models import DOC_COLUMNS
from embedding.services import document_service, query_service


def _chunk(doc_id: str, idx: int) -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#c{idx:03d}",
        "chunk_text": f"chunk-{doc_id}-{idx}",
        "chunk_vector": [float(idx), 0.0, 1.0],
        "chunk_score": 0.9,
        "chunk_rank": idx,
    }


def test_process_doc_flushes_multiple_batches(monkeypatch: pytest.MonkeyPatch):
    docs = [
        (1, {"doc_id": "d1", "doc_text": "text-1"}),
        (2, {"doc_id": "d2", "doc_text": "text-2"}),
        (3, {"doc_id": "d3", "doc_text": "text-3"}),
    ]
    selected_batches = [
        [_chunk("d1", 1), _chunk("d1", 2)],
        [_chunk("d2", 1)],
        [_chunk("d3", 1), _chunk("d3", 2)],
    ]

    selector = Mock()
    selector.run.side_effect = selected_batches

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "ChunkSelector", lambda _e: selector)
    monkeypatch.setattr(document_service, "DOC_FLUSH_CHUNK_THRESHOLD", 3)

    output = Mock()

    document_service.process_doc(Mock(), Path("docs.jsonl"), output)

    assert output.write_doc_chunks.call_count == 3
    first_chunks = output.write_doc_chunks.call_args_list[0].args[0]
    second_chunks = output.write_doc_chunks.call_args_list[1].args[0]
    third_chunks = output.write_doc_chunks.call_args_list[2].args[0]

    assert len(first_chunks) == 2
    assert len(second_chunks) == 1
    assert len(third_chunks) == 2
    assert set(first_chunks[0].keys()) == set(DOC_COLUMNS)


def test_process_doc_flushes_final_partial_batch(monkeypatch: pytest.MonkeyPatch):
    docs = [
        (1, {"doc_id": "d1", "doc_text": "text-1"}),
        (2, {"doc_id": "d2", "doc_text": "text-2"}),
    ]
    selected_batches = [
        [_chunk("d1", 1), _chunk("d1", 2)],
        [_chunk("d2", 1)],
    ]

    selector = Mock()
    selector.run.side_effect = selected_batches

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "ChunkSelector", lambda _e: selector)
    monkeypatch.setattr(document_service, "DOC_FLUSH_CHUNK_THRESHOLD", 10)

    output = Mock()

    document_service.process_doc(Mock(), Path("docs.jsonl"), output)

    assert output.write_doc_chunks.call_count == 2
    written_chunks = []
    for call in output.write_doc_chunks.call_args_list:
        written_chunks.extend(call.args[0])
    assert len(written_chunks) == 3
    assert set(written_chunks[0].keys()) == set(DOC_COLUMNS)


def test_process_doc_raises_when_no_chunks(monkeypatch: pytest.MonkeyPatch):
    docs = [
        (1, {"doc_id": "d1", "doc_text": "text-1"}),
        (2, {"doc_id": "d2", "doc_text": "text-2"}),
    ]

    selector = Mock()
    selector.run.side_effect = [[], []]

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "ChunkSelector", lambda _e: selector)

    output = Mock()

    with pytest.raises(ProcessingError, match="No documents found"):
        document_service.process_doc(Mock(), Path("docs.jsonl"), output)

    output.write_doc_chunks.assert_not_called()


def test_process_query_encodes_and_writes(monkeypatch: pytest.MonkeyPatch):
    queries = [(1, {"query_id": "q1", "query_text": "主题查询"})]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    embedding = Mock()
    embedding.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    output = Mock()

    query_service.process_query(embedding, Path("queries.jsonl"), output)

    embedding.encode.assert_called_once_with(["主题查询"], is_query=True)
    output.write_queries.assert_called_once()
    written_df = output.write_queries.call_args.args[0]
    assert list(written_df["query_id"]) == ["q1"]


def test_process_query_rejects_empty_query_fields(monkeypatch: pytest.MonkeyPatch):
    queries = [(1, {"query_id": "q1", "query_text": ""})]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    with pytest.raises(ProcessingError):
        query_service.process_query(Mock(), Path("queries.jsonl"), Mock())


def test_process_query_wraps_encoder_errors(monkeypatch: pytest.MonkeyPatch):
    queries = [(1, {"query_id": "q1", "query_text": "主题查询"})]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    embedding = Mock()
    embedding.encode.side_effect = RuntimeError("boom")

    with pytest.raises(ProcessingError):
        query_service.process_query(embedding, Path("queries.jsonl"), Mock())
