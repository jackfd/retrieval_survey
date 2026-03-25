from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from index_builder.config import RuntimeConfig
from index_builder.embedding import EmbeddingStrategy, ensure_embedding_shape
from index_builder.errors import ChunkSelectionError, EmbeddingGenerationError
from index_builder.io_utils import read_jsonl, utc_now_iso


@dataclass
class DocProcessResult:
    docs_df: pd.DataFrame
    failures: List[Dict[str, str]]
    attempted_docs: int


def build_chunk_selector(runtime: RuntimeConfig, embedding_strategy: EmbeddingStrategy):
    chunk_selector_dir = Path(__file__).resolve().parent.parent / "chunk_selector"
    if str(chunk_selector_dir) not in sys.path:
        sys.path.insert(0, str(chunk_selector_dir))
    from chunk_selector import ChunkSelector, SelectorConfig

    return ChunkSelector(
        embedding_api_url=runtime.embedding_api_url or "http://unused.local",
        chunk_num=1,
        config=SelectorConfig(batch_size=runtime.batch_size),
        embedding_provider=lambda chunks: ensure_embedding_shape(
            embedding_strategy.encode(chunks, is_query=False),
            runtime.embedding_dim,
        ),
    )


def process_docs(
    *,
    docs_path: Path,
    selector,
    runtime: RuntimeConfig,
    retry_mode: bool,
    retry_id_set: set[str],
    logger,
) -> DocProcessResult:
    doc_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    attempted_docs = 0

    for _, obj in read_jsonl(docs_path):
        doc_id = str(obj.get("doc_id", "")).strip()
        doc_text = str(obj.get("doc_text", "")).strip()
        if not doc_id or not doc_text:
            continue
        if retry_mode and doc_id not in retry_id_set:
            continue

        attempted_docs += 1
        try:
            selected = selector.select_chunks(doc_text, "")
            if not selected:
                raise ChunkSelectionError("No chunk selected")
            first = selected[0]
            chunk_text = str(first.get("chunk_text") or first.get("chunk") or "").strip()
            embedding = np.asarray(first.get("embedding", []), dtype=np.float32)
            if not chunk_text:
                raise ChunkSelectionError("Selected chunk text is empty")
            if embedding.ndim != 1 or embedding.shape[0] != runtime.embedding_dim:
                raise EmbeddingGenerationError(
                    f"Selected chunk embedding dim mismatch expected={runtime.embedding_dim}, "
                    f"got={embedding.shape}"
                )
            doc_records.append(
                {
                    "doc_id": doc_id,
                    "chunk_text": chunk_text,
                    "chunk_embedding": embedding.tolist(),
                }
            )
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error("doc_failed doc_id=%s error_type=%s error=%s", doc_id, error_type, exc, exc_info=True)
            failures.append(
                {
                    "doc_id": doc_id,
                    "error_type": error_type,
                    "error_message": str(exc),
                    "timestamp_utc": utc_now_iso(),
                }
            )

    docs_df = pd.DataFrame(doc_records, columns=["doc_id", "chunk_text", "chunk_embedding"])
    return DocProcessResult(docs_df=docs_df, failures=failures, attempted_docs=attempted_docs)


def process_queries(
    *,
    queries_path: Path,
    embedding_strategy: EmbeddingStrategy,
    runtime: RuntimeConfig,
) -> pd.DataFrame:
    query_ids: List[str] = []
    query_texts: List[str] = []
    for _, obj in read_jsonl(queries_path):
        qid = str(obj.get("query_id", "")).strip()
        qtext = str(obj.get("query_text", "")).strip()
        if qid and qtext:
            query_ids.append(qid)
            query_texts.append(qtext)

    query_vecs = (
        embedding_strategy.encode(query_texts, is_query=True)
        if query_texts
        else np.empty((0, runtime.embedding_dim))
    )
    query_vecs = ensure_embedding_shape(query_vecs, runtime.embedding_dim) if query_texts else query_vecs
    return pd.DataFrame(
        {
            "query_id": query_ids,
            "query_text": query_texts,
            "query_embedding": [vec.tolist() for vec in query_vecs] if len(query_ids) else [],
        }
    )

