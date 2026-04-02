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
    splitter = Mock()
    monkeypatch.setattr(document_service, "ChunkSplitter", lambda: splitter)
    build_candidates_mock = Mock(
        side_effect=[
        [{"order": 1, "text": "d1-c1"}, {"order": 2, "text": "d1-c2"}],
        [{"order": 1, "text": "d2-c1"}],
        [{"order": 1, "text": "d3-c1"}, {"order": 2, "text": "d3-c2"}],
    ])
    select_from_embeddings_mock = Mock(side_effect=[
        [_chunk("d1", 1), _chunk("d1", 2)],
        [_chunk("d2", 1)],
        [_chunk("d3", 1), _chunk("d3", 2)],
    ])
    embedding = Mock()
    embedding.encode.return_value = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "build_candidates", build_candidates_mock)
    monkeypatch.setattr(
        document_service, "select_from_embeddings", select_from_embeddings_mock
    )

    output = Mock()

    document_service.process_doc(embedding, Path("docs.jsonl"), output)

    assert output.write_doc_chunks.call_count == 3
    assert embedding.encode.call_count == 1
    assert embedding.encode.call_args_list[0].args[0] == [
        "d1-c1",
        "d1-c2",
        "d2-c1",
        "d3-c1",
        "d3-c2",
    ]
    assert embedding.encode.call_args_list[0].kwargs == {"is_query": False}
    assert build_candidates_mock.call_count == 3
    assert build_candidates_mock.call_args_list[0].args[1] is splitter
    assert select_from_embeddings_mock.call_args_list[0].args[0] == "d1"
    assert select_from_embeddings_mock.call_args_list[1].args[0] == "d2"
    assert select_from_embeddings_mock.call_args_list[2].args[0] == "d3"
    assert [
        item["text"]
        for item in select_from_embeddings_mock.call_args_list[1].args[1]
    ] == ["d2-c1"]
    assert [
        item["text"]
        for item in select_from_embeddings_mock.call_args_list[2].args[1]
    ] == ["d3-c1", "d3-c2"]
    assert select_from_embeddings_mock.call_args_list[1].args[2].shape == (1, 2)
    assert select_from_embeddings_mock.call_args_list[2].args[2].shape == (2, 2)
    assert select_from_embeddings_mock.call_args_list[0].args[2].shape == (2, 2)
    np.testing.assert_array_equal(
        select_from_embeddings_mock.call_args_list[1].args[2],
        np.array([[0.1, 0.9]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        select_from_embeddings_mock.call_args_list[2].args[2],
        np.array([[0.9, 0.1], [0.5, 0.5]], dtype=np.float32),
    )
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
    splitter = Mock()
    monkeypatch.setattr(document_service, "ChunkSplitter", lambda: splitter)
    build_candidates_mock = Mock(side_effect=[
        [{"order": 1, "text": "d1-c1"}, {"order": 2, "text": "d1-c2"}],
        [{"order": 1, "text": "d2-c1"}],
    ])
    select_from_embeddings_mock = Mock(side_effect=[
        [_chunk("d1", 1), _chunk("d1", 2)],
        [_chunk("d2", 1)],
    ])
    embedding = Mock()
    embedding.encode.return_value = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32
    )

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "build_candidates", build_candidates_mock)
    monkeypatch.setattr(
        document_service, "select_from_embeddings", select_from_embeddings_mock
    )

    output = Mock()

    document_service.process_doc(embedding, Path("docs.jsonl"), output)

    assert output.write_doc_chunks.call_count == 2
    assert embedding.encode.call_count == 1
    assert embedding.encode.call_args_list[0].args[0] == ["d1-c1", "d1-c2", "d2-c1"]
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

    splitter = Mock()
    monkeypatch.setattr(document_service, "ChunkSplitter", lambda: splitter)
    build_candidates_mock = Mock(side_effect=[[], []])

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "build_candidates", build_candidates_mock)

    output = Mock()

    with pytest.raises(ProcessingError, match="No documents found"):
        document_service.process_doc(Mock(), Path("docs.jsonl"), output)

    output.write_doc_chunks.assert_not_called()


def test_process_doc_waits_for_full_doc_before_selecting(
    monkeypatch: pytest.MonkeyPatch,
):
    docs = [
        (1, {"doc_id": "d1", "doc_text": "text-1"}),
        (2, {"doc_id": "d2", "doc_text": "text-2"}),
    ]
    splitter = Mock()
    monkeypatch.setattr(document_service, "ChunkSplitter", lambda: splitter)
    build_candidates_mock = Mock(side_effect=[
        [
            {"order": 1, "text": "d1-c1"},
            {"order": 2, "text": "d1-c2"},
            {"order": 3, "text": "d1-c3"},
        ],
        [{"order": 1, "text": "d2-c1"}],
    ])
    select_from_embeddings_mock = Mock(side_effect=[
        [_chunk("d1", 1), _chunk("d1", 2)],
        [_chunk("d2", 1)],
    ])
    embedding = Mock()
    embedding.encode.return_value = np.array(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "build_candidates", build_candidates_mock)
    monkeypatch.setattr(
        document_service, "select_from_embeddings", select_from_embeddings_mock
    )

    output = Mock()

    document_service.process_doc(embedding, Path("docs.jsonl"), output)

    assert embedding.encode.call_count == 1
    assert embedding.encode.call_args_list[0].args[0] == [
        "d1-c1",
        "d1-c2",
        "d1-c3",
        "d2-c1",
    ]
    assert select_from_embeddings_mock.call_args_list[0].args[0] == "d1"
    assert select_from_embeddings_mock.call_args_list[0].args[2].shape == (3, 2)
    assert select_from_embeddings_mock.call_args_list[1].args[0] == "d2"
    assert select_from_embeddings_mock.call_args_list[1].args[2].shape == (1, 2)


def test_process_doc_wraps_embedding_errors(monkeypatch: pytest.MonkeyPatch):
    docs = [(1, {"doc_id": "d1", "doc_text": "text-1"})]
    splitter = Mock()
    monkeypatch.setattr(document_service, "ChunkSplitter", lambda: splitter)
    build_candidates_mock = Mock(return_value=[{"order": 1, "text": "d1-c1"}])
    embedding = Mock()
    embedding.encode.side_effect = RuntimeError("boom")

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "build_candidates", build_candidates_mock)

    with pytest.raises(
        ProcessingError,
        match="start_doc_id=d1 end_doc_id=d1 chunk_count=1",
    ):
        document_service.process_doc(embedding, Path("docs.jsonl"), Mock())


def test_process_doc_flushes_at_most_once_per_doc(monkeypatch: pytest.MonkeyPatch):
    docs = [
        (1, {"doc_id": "d1", "doc_text": "text-1"}),
        (2, {"doc_id": "d2", "doc_text": "text-2"}),
        (3, {"doc_id": "d3", "doc_text": "text-3"}),
    ]
    splitter = Mock()
    monkeypatch.setattr(document_service, "ChunkSplitter", lambda: splitter)
    build_candidates_mock = Mock(side_effect=[
        [{"order": 1, "text": "d1-c1"}],
        [],
        [{"order": 1, "text": "d3-c1"}],
    ])
    select_from_embeddings_mock = Mock(side_effect=[
        [_chunk("d1", 1)],
        [_chunk("d3", 1)],
    ])
    embedding = Mock()
    embedding.encode.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "build_candidates", build_candidates_mock)
    monkeypatch.setattr(
        document_service, "select_from_embeddings", select_from_embeddings_mock
    )

    output = Mock()

    document_service.process_doc(embedding, Path("docs.jsonl"), output)

    assert embedding.encode.call_count == 1
    assert embedding.encode.call_args_list[0].args[0] == ["d1-c1", "d3-c1"]


def test_process_doc_splits_work_across_windows(monkeypatch: pytest.MonkeyPatch):
    docs = [
        (1, {"doc_id": "d1", "doc_text": "text-1"}),
        (2, {"doc_id": "d2", "doc_text": "text-2"}),
        (3, {"doc_id": "d3", "doc_text": "text-3"}),
    ]
    splitter = Mock()
    monkeypatch.setattr(document_service, "ChunkSplitter", lambda: splitter)
    monkeypatch.setattr(document_service, "DOC_WINDOW_SIZE", 2)
    build_candidates_mock = Mock(
        side_effect=[
            [{"order": 1, "text": "d1-c1"}],
            [{"order": 1, "text": "d2-c1"}],
            [{"order": 1, "text": "d3-c1"}],
        ]
    )
    select_from_embeddings_mock = Mock(
        side_effect=[[_chunk("d1", 1)], [_chunk("d2", 1)], [_chunk("d3", 1)]]
    )
    embedding = Mock()
    embedding.encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[0.5, 0.5]], dtype=np.float32),
    ]

    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)
    monkeypatch.setattr(document_service, "build_candidates", build_candidates_mock)
    monkeypatch.setattr(
        document_service, "select_from_embeddings", select_from_embeddings_mock
    )

    output = Mock()

    document_service.process_doc(embedding, Path("docs.jsonl"), output)

    assert embedding.encode.call_count == 2
    assert embedding.encode.call_args_list[0].args[0] == ["d1-c1", "d2-c1"]
    assert embedding.encode.call_args_list[1].args[0] == ["d3-c1"]
    assert select_from_embeddings_mock.call_args_list[0].args[0] == "d1"
    assert select_from_embeddings_mock.call_args_list[1].args[0] == "d2"
    assert select_from_embeddings_mock.call_args_list[2].args[0] == "d3"


def test_process_query_encodes_and_writes(monkeypatch: pytest.MonkeyPatch):
    queries = [
        (1, {"query_id": "q1", "query_text": "主题查询1"}),
        (2, {"query_id": "q2", "query_text": "主题查询2"}),
        (3, {"query_id": "q3", "query_text": "主题查询3"}),
    ]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    embedding = Mock()
    embedding.inference = Mock(batch_size=2)
    embedding.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], dtype=np.float32
    )
    output = Mock()

    query_service.process_query(embedding, Path("queries.jsonl"), output)

    assert embedding.encode.call_count == 1
    assert embedding.encode.call_args_list[0].args[0] == ["主题查询1", "主题查询2", "主题查询3"]
    assert embedding.encode.call_args_list[0].kwargs == {"is_query": True}
    assert output.write_query_chunks.call_count == 1
    first_batch = output.write_query_chunks.call_args_list[0].args[0]
    assert [record["query_id"] for record in first_batch] == ["q1", "q2", "q3"]


def test_process_query_flushes_single_final_partial_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    queries = [
        (1, {"query_id": "q1", "query_text": "主题查询1"}),
        (2, {"query_id": "q2", "query_text": "主题查询2"}),
    ]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    embedding = Mock()
    embedding.inference = Mock(batch_size=8)
    embedding.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float32
    )
    output = Mock()

    query_service.process_query(embedding, Path("queries.jsonl"), output)

    embedding.encode.assert_called_once_with(["主题查询1", "主题查询2"], is_query=True)
    output.write_query_chunks.assert_called_once()
    written_batch = output.write_query_chunks.call_args.args[0]
    assert [record["query_id"] for record in written_batch] == ["q1", "q2"]


def test_process_query_rejects_empty_query_fields(monkeypatch: pytest.MonkeyPatch):
    queries = [(1, {"query_id": "q1", "query_text": ""})]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    with pytest.raises(ProcessingError):
        query_service.process_query(Mock(), Path("queries.jsonl"), Mock())


def test_process_query_wraps_encoder_errors(monkeypatch: pytest.MonkeyPatch):
    queries = [
        (10, {"query_id": "q1", "query_text": "主题查询1"}),
        (11, {"query_id": "q2", "query_text": "主题查询2"}),
    ]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    embedding = Mock()
    embedding.inference = Mock(batch_size=4)
    embedding.encode.side_effect = RuntimeError("boom")

    with pytest.raises(
        ProcessingError,
        match="start_line_num=10 end_line_num=11 query_count=2",
    ):
        query_service.process_query(embedding, Path("queries.jsonl"), Mock())


def test_process_query_raises_when_encoder_returns_wrong_batch_size(
    monkeypatch: pytest.MonkeyPatch,
):
    queries = [
        (7, {"query_id": "q1", "query_text": "主题查询1"}),
        (8, {"query_id": "q2", "query_text": "主题查询2"}),
    ]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)

    embedding = Mock()
    embedding.inference = Mock(batch_size=2)
    embedding.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    with pytest.raises(
        ProcessingError,
        match="expected=2 actual=1",
    ):
        query_service.process_query(embedding, Path("queries.jsonl"), Mock())
