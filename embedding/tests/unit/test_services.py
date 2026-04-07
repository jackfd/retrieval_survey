"""Service-layer unit tests for document/query processing."""

from __future__ import annotations

import json
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


def _candidate_doc(doc_id: str, *texts: str) -> dict:
    return {
        "doc_id": doc_id,
        "candidate_count": len(texts),
        "candidates": [
            {"order": index, "text": text} for index, text in enumerate(texts, start=1)
        ],
    }


def _make_splitter_mock(max_length: int = 256) -> Mock:
    splitter = Mock()
    splitter.hard_max_tokens = max_length
    splitter.estimate_tokens.side_effect = lambda text: len(str(text))
    return splitter


def test_build_doc_candidates_writes_normalized_shared_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    docs = [
        (1, {"doc_id": "d1", "doc_text": "  text-1  "}),
        (2, {"doc_id": "d2", "doc_text": [" 第一段 ", " ", "", "第二段"]}),
    ]
    splitter = _make_splitter_mock()
    splitter.split_to_candidates = Mock(
        side_effect=[
            [{"order": 1, "text": "d1-c1"}],
            [{"order": 1, "text": "d2-c1"}, {"order": 2, "text": "d2-c2"}],
        ]
    )

    monkeypatch.setattr(document_service, "ChunkSplitter", lambda **_kwargs: splitter)
    monkeypatch.setattr(document_service, "read_objects", lambda _p: docs)

    output_path = tmp_path / "output" / "scifact_v1_candidates.jsonl"
    document_service.build_doc_candidates(Path("docs.jsonl"), output_path, 256)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows == [
        _candidate_doc("d1", "d1-c1"),
        _candidate_doc("d2", "d2-c1", "d2-c2"),
    ]
    assert splitter.split_to_candidates.call_args_list[0].args[0] == ["text-1"]
    assert splitter.split_to_candidates.call_args_list[1].args[0] == [
        "第一段",
        "第二段",
    ]


def test_process_doc_consumes_shared_candidates(monkeypatch: pytest.MonkeyPatch):
    candidate_docs = [
        (1, _candidate_doc("d1", "d1-c1", "d1-c2")),
        (2, _candidate_doc("d2", "d2-c1")),
        (3, _candidate_doc("d3", "d3-c1", "d3-c2")),
    ]
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
    select_from_embeddings_mock = Mock(
        side_effect=[
            [_chunk("d1", 1), _chunk("d1", 2)],
            [_chunk("d2", 1)],
            [_chunk("d3", 1), _chunk("d3", 2)],
        ]
    )

    monkeypatch.setattr(document_service, "read_objects", lambda _p: candidate_docs)
    monkeypatch.setattr(
        document_service, "select_from_embeddings", select_from_embeddings_mock
    )

    output = Mock()
    document_service.process_doc(
        embedding, Path("output/scifact_v1_candidates.jsonl"), output
    )

    assert output.write_doc_chunks.call_count == 3
    assert embedding.encode.call_count == 1
    assert embedding.encode.call_args.args[0] == [
        "d1-c1",
        "d1-c2",
        "d2-c1",
        "d3-c1",
        "d3-c2",
    ]
    assert embedding.encode.call_args.kwargs == {"is_query": False}
    assert select_from_embeddings_mock.call_args_list[0].args[0] == "d1"
    assert select_from_embeddings_mock.call_args_list[1].args[0] == "d2"
    assert select_from_embeddings_mock.call_args_list[2].args[0] == "d3"
    first_chunks = output.write_doc_chunks.call_args_list[0].args[0]
    assert set(first_chunks[0].keys()) == set(DOC_COLUMNS)


def test_process_doc_raises_when_no_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        document_service,
        "read_objects",
        lambda _p: [(1, _candidate_doc("d1")), (2, _candidate_doc("d2"))],
    )

    with pytest.raises(ProcessingError, match="No documents found"):
        document_service.process_doc(
            Mock(), Path("output/scifact_v1_candidates.jsonl"), Mock()
        )


def test_process_doc_wraps_embedding_errors(monkeypatch: pytest.MonkeyPatch):
    logger = Mock()
    embedding = Mock()
    embedding.encode.side_effect = RuntimeError("boom")

    monkeypatch.setattr(
        document_service,
        "read_objects",
        lambda _p: [(1, _candidate_doc("d1", "abcde"))],
    )
    monkeypatch.setattr(document_service, "logger", logger)

    with pytest.raises(
        ProcessingError,
        match="start_doc_id=d1 end_doc_id=d1 chunk_count=1",
    ):
        document_service.process_doc(
            embedding, Path("output/scifact_v1_candidates.jsonl"), Mock()
        )

    assert logger.exception.call_count == 1
    message = logger.exception.call_args.args[0]
    assert "docs embedding batch failed, start_doc_id=%s" in message
    assert logger.exception.call_args.args[1:] == (
        "d1",
        "d1",
        1,
        "RuntimeError",
    )


def test_process_doc_splits_work_across_windows(monkeypatch: pytest.MonkeyPatch):
    candidate_docs = [
        (1, _candidate_doc("d1", "d1-c1")),
        (2, _candidate_doc("d2", "d2-c1")),
        (3, _candidate_doc("d3", "d3-c1")),
    ]
    embedding = Mock()
    embedding.encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[0.5, 0.5]], dtype=np.float32),
    ]

    monkeypatch.setattr(document_service, "read_objects", lambda _p: candidate_docs)
    monkeypatch.setattr(document_service, "DOC_WINDOW_SIZE", 2)
    monkeypatch.setattr(
        document_service,
        "select_from_embeddings",
        Mock(side_effect=[[_chunk("d1", 1)], [_chunk("d2", 1)], [_chunk("d3", 1)]]),
    )

    output = Mock()
    document_service.process_doc(
        embedding, Path("output/scifact_v1_candidates.jsonl"), output
    )

    assert embedding.encode.call_count == 2
    assert embedding.encode.call_args_list[0].args[0] == ["d1-c1", "d2-c1"]
    assert embedding.encode.call_args_list[1].args[0] == ["d3-c1"]


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
    assert embedding.encode.call_args_list[0].args[0] == [
        "主题查询1",
        "主题查询2",
        "主题查询3",
    ]
    assert embedding.encode.call_args_list[0].kwargs == {"is_query": True}
    assert output.write_query_chunks.call_count == 1
    first_batch = output.write_query_chunks.call_args_list[0].args[0]
    assert [record["query_id"] for record in first_batch] == ["q1", "q2", "q3"]


def test_process_query_splits_work_across_windows(
    monkeypatch: pytest.MonkeyPatch,
):
    queries = [
        (1, {"query_id": "q1", "query_text": "主题查询1"}),
        (2, {"query_id": "q2", "query_text": "主题查询2"}),
        (3, {"query_id": "q3", "query_text": "主题查询3"}),
    ]
    monkeypatch.setattr(query_service, "read_objects", lambda _p: queries)
    monkeypatch.setattr(query_service, "QUERY_WINDOW_SIZE", 2)

    embedding = Mock()
    embedding.inference = Mock(batch_size=64)
    embedding.encode.side_effect = [
        np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float32),
        np.array([[0.4, 0.5, 0.6]], dtype=np.float32),
    ]
    output = Mock()

    query_service.process_query(embedding, Path("queries.jsonl"), output)

    assert embedding.encode.call_count == 2
    assert embedding.encode.call_args_list[0].args[0] == ["主题查询1", "主题查询2"]
    assert embedding.encode.call_args_list[1].args[0] == ["主题查询3"]
    assert output.write_query_chunks.call_count == 2


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
