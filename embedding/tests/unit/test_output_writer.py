"""OutputWriter unit tests for append semantics and close behavior."""

from __future__ import annotations

import pandas as pd

from embedding.domain.models import DOC_COLUMNS
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


def test_write_docs_appends_across_multiple_calls(tmp_path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4)

    df1 = pd.DataFrame([_doc_row("d1", 1), _doc_row("d1", 2)], columns=DOC_COLUMNS)
    df2 = pd.DataFrame([_doc_row("d2", 1)], columns=DOC_COLUMNS)

    writer.write_docs(df1)
    writer.write_docs(df2)
    writer.close()

    written = pd.read_parquet(writer.docs_path)
    assert len(written) == 3
    assert list(written["doc_id"]) == ["d1", "d1", "d2"]


def test_write_docs_empty_batch_does_not_change_row_count(tmp_path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4)

    df = pd.DataFrame([_doc_row("d1", 1)], columns=DOC_COLUMNS)
    empty_df = pd.DataFrame(columns=DOC_COLUMNS)

    writer.write_docs(df)
    writer.write_docs(empty_df)
    writer.close()

    written = pd.read_parquet(writer.docs_path)
    assert len(written) == 1
    assert list(written["doc_id"]) == ["d1"]


def test_close_is_idempotent(tmp_path):
    writer = OutputWriter(output_dir=tmp_path, embedding_dim=4)

    writer.write_docs(pd.DataFrame([_doc_row("d1", 1)], columns=DOC_COLUMNS))
    writer.close()
    writer.close()
