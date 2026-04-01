"""OutputWriter unit tests for sharded docs output and close behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from embedding.domain.models import DOC_COLUMNS, QUERY_COLUMNS
from embedding.infra.output_writer import OutputWriter


def _doc_row(doc_id: str, idx: int) -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#c{idx:03d}",
        "chunk_text": f"chunk-{doc_id}-{idx}",
        "chunk_vector": [float(idx), 0.0, 1.0],
        "chunk_score": 0.9,
        "chunk_rank": idx,
    }


def _doc_chunks(doc_id: str, chunk_count: int) -> list[dict]:
    return [_doc_row(doc_id, index) for index in range(1, chunk_count + 1)]


def _read_all_parts(output_dir: Path, embedding_dim: int) -> pd.DataFrame:
    parts = sorted((output_dir / f"docs_dim{embedding_dim}").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=DOC_COLUMNS)
    frames = [pd.read_parquet(part) for part in parts]
    return pd.concat(frames, ignore_index=True)


def _query_row(query_id: str, idx: int) -> dict:
    return {
        "query_id": query_id,
        "query_text": f"query-{idx}",
        "query_embedding": [float(idx), 0.0],
    }


def _read_all_query_parts(output_dir: Path, embedding_dim: int) -> pd.DataFrame:
    parts = sorted((output_dir / f"queries_dim{embedding_dim}").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=QUERY_COLUMNS)
    frames = [pd.read_parquet(part) for part in parts]
    return pd.concat(frames, ignore_index=True)


def test_write_doc_chunks_rolls_over_to_multiple_parts(tmp_path: Path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4, shard_size=2)

    writer.write_doc_chunks(_doc_chunks("d1", 2))
    writer.write_doc_chunks(_doc_chunks("d2", 1))
    writer.write_doc_chunks(_doc_chunks("d3", 2))
    writer.close()

    parts = sorted((tmp_path / "docs_dim4").glob("part-*.parquet"))
    assert [p.name for p in parts] == [
        "part-00000.parquet",
        "part-00001.parquet",
    ]

    first_part = pd.read_parquet(parts[0])
    second_part = pd.read_parquet(parts[1])

    assert list(first_part.columns) == DOC_COLUMNS
    assert list(second_part.columns) == DOC_COLUMNS
    assert len(first_part) == 3
    assert len(second_part) == 2
    assert list(first_part["doc_id"]) == ["d1", "d1", "d2"]
    assert list(second_part["doc_id"]) == ["d3", "d3"]

    written = _read_all_parts(tmp_path, embedding_dim=4)
    assert len(written) == 5
    assert list(written["doc_id"]) == ["d1", "d1", "d2", "d3", "d3"]


def test_close_is_idempotent(tmp_path: Path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4, shard_size=2)

    writer.write_doc_chunks(_doc_chunks("d1", 1))
    writer.close()
    writer.close()

    parts = sorted((tmp_path / "docs_dim4").glob("part-*.parquet"))
    assert len(parts) == 1
    written = pd.read_parquet(parts[0])
    assert len(written) == 1


def test_write_doc_chunks_empty_does_not_create_parts(tmp_path: Path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4, shard_size=2)

    writer.write_doc_chunks([])
    writer.close()

    parts = list((tmp_path / "docs_dim4").glob("part-*.parquet"))
    assert parts == []


def test_write_query_chunks_rolls_over_to_multiple_parts(tmp_path: Path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4, shard_size=2)

    writer.write_query_chunks([_query_row("q1", 1)])
    writer.write_query_chunks([_query_row("q2", 2), _query_row("q3", 3)])
    writer.write_query_chunks([_query_row("q4", 4)])
    writer.close()

    parts = sorted((tmp_path / "queries_dim4").glob("part-*.parquet"))
    assert [p.name for p in parts] == [
        "part-00000.parquet",
        "part-00001.parquet",
    ]
    first_part = pd.read_parquet(parts[0])
    second_part = pd.read_parquet(parts[1])

    assert list(first_part.columns) == QUERY_COLUMNS
    assert list(second_part.columns) == QUERY_COLUMNS
    assert list(first_part["query_id"]) == ["q1", "q2", "q3"]
    assert list(second_part["query_id"]) == ["q4"]

    written = _read_all_query_parts(tmp_path, embedding_dim=4)
    assert list(written["query_id"]) == ["q1", "q2", "q3", "q4"]


def test_close_flushes_final_query_part(tmp_path: Path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4, shard_size=3)

    writer.write_query_chunks([_query_row("q1", 1)])
    writer.close()
    writer.close()

    parts = sorted((tmp_path / "queries_dim4").glob("part-*.parquet"))
    assert len(parts) == 1
    written = pd.read_parquet(parts[0])
    assert list(written["query_id"]) == ["q1"]


def test_write_query_chunks_empty_does_not_create_parts(tmp_path: Path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4, shard_size=2)

    writer.write_query_chunks([])
    writer.close()

    parts = list((tmp_path / "queries_dim4").glob("part-*.parquet"))
    assert parts == []


def test_docs_shard_size_must_be_positive(tmp_path: Path):
    try:
        OutputWriter(output_dir=tmp_path, embedding_dim=4, shard_size=0)
    except ValueError as exc:
        assert "docs_shard_size" in str(exc) or "shard_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
