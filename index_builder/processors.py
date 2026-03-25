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
class ProcessResult:
    output_df: pd.DataFrame
    failures: List[Dict[str, str]]
    attempted_count: int
    skipped_missing_required_count: int


def _build_failure_record(
    *,
    record_type: str,
    record_id: str,
    error_type: str,
    error_message: str,
    stage: str,
) -> Dict[str, str]:
    return {
        "record_type": record_type,
        "record_id": record_id,
        "error_type": error_type,
        "error_message": error_message,
        "timestamp_utc": utc_now_iso(),
        "stage": stage,
    }


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
) -> ProcessResult:
    """处理文档并将它们转换为嵌入向量格式

    该函数读取JSONL文件中的文档，对每个文档进行分块选择，并生成相应的嵌入向量。
    在重试模式下，只处理retry_id_set中指定的文档ID。

    Args:
        docs_path: 包含文档的JSONL文件路径
        selector: 用于选择文档块的对象，具有select_chunks方法
        runtime: 运行时配置对象，包含嵌入维度等信息
        retry_mode: 是否为重试模式，如果是则只处理retry_id_set中的文档
        retry_id_set: 重试模式下需要处理的文档ID集合
        logger: 用于记录错误和日志消息的日志记录器

    Returns:
        ProcessResult: 包含处理后的文档DataFrame、失败记录列表和统计信息
    """
    doc_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    attempted_count = 0
    skipped_missing_required_count = 0
    missing_warning_limit = 5
    missing_warning_count = 0

    for _, obj in read_jsonl(docs_path):
        doc_id = str(obj.get("doc_id", "")).strip()
        doc_text = str(obj.get("doc_text", "")).strip()
        if not doc_id or not doc_text:
            skipped_missing_required_count += 1
            failures.append(
                _build_failure_record(
                    record_type="doc",
                    record_id=doc_id,
                    error_type="MissingRequiredField",
                    error_message="doc_id or doc_text is empty",
                    stage="input_validation",
                )
            )
            if missing_warning_count < missing_warning_limit:
                logger.warning(
                    "doc_missing_required_fields doc_id=%s has_doc_text=%s",
                    doc_id,
                    bool(doc_text),
                )
                missing_warning_count += 1
            continue
        if retry_mode and doc_id not in retry_id_set:
            continue

        attempted_count += 1
        try:
            selected = selector.select_chunks(doc_text, "")
            if not selected:
                raise ChunkSelectionError("No chunk selected")
            first = selected[0]
            chunk_text = str(first.get("chunk_text") or "").strip()
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
            stage = "chunk_selection" if isinstance(exc, ChunkSelectionError) else "embedding"
            logger.error(
                "doc_failed doc_id=%s error_type=%s error=%s",
                doc_id,
                error_type,
                exc,
                exc_info=True,
            )
            failures.append(
                _build_failure_record(
                    record_type="doc",
                    record_id=doc_id,
                    error_type=error_type,
                    error_message=str(exc),
                    stage=stage,
                )
            )

    if skipped_missing_required_count > missing_warning_limit:
        logger.warning(
            "doc_missing_required_fields_suppressed suppressed_count=%s total_missing=%s",
            skipped_missing_required_count - missing_warning_limit,
            skipped_missing_required_count,
        )

    docs_df = pd.DataFrame(
        doc_records, columns=["doc_id", "chunk_text", "chunk_embedding"]
    )
    return ProcessResult(
        output_df=docs_df,
        failures=failures,
        attempted_count=attempted_count,
        skipped_missing_required_count=skipped_missing_required_count,
    )


def process_queries(
    *,
    queries_path: Path,
    embedding_strategy: EmbeddingStrategy,
    runtime: RuntimeConfig,
    logger,
) -> ProcessResult:
    query_ids: List[str] = []
    query_texts: List[str] = []
    query_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    attempted_count = 0
    skipped_missing_required_count = 0
    missing_warning_limit = 5
    missing_warning_count = 0
    for _, obj in read_jsonl(queries_path):
        qid = str(obj.get("query_id", "")).strip()
        qtext = str(obj.get("query_text", "")).strip()
        if not qid or not qtext:
            skipped_missing_required_count += 1
            failures.append(
                _build_failure_record(
                    record_type="query",
                    record_id=qid,
                    error_type="MissingRequiredField",
                    error_message="query_id or query_text is empty",
                    stage="input_validation",
                )
            )
            if missing_warning_count < missing_warning_limit:
                logger.warning(
                    "query_missing_required_fields query_id=%s has_query_text=%s",
                    qid,
                    bool(qtext),
                )
                missing_warning_count += 1
            continue

        query_ids.append(qid)
        query_texts.append(qtext)

    for qid, qtext in zip(query_ids, query_texts):
        attempted_count += 1
        try:
            vecs = embedding_strategy.encode([qtext], is_query=True)
            vecs = ensure_embedding_shape(vecs, runtime.embedding_dim)
            query_records.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "query_embedding": vecs[0].tolist(),
                }
            )
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error(
                "query_failed query_id=%s error_type=%s error=%s",
                qid,
                error_type,
                exc,
                exc_info=True,
            )
            failures.append(
                _build_failure_record(
                    record_type="query",
                    record_id=qid,
                    error_type=error_type,
                    error_message=str(exc),
                    stage="embedding",
                )
            )

    if skipped_missing_required_count > missing_warning_limit:
        logger.warning(
            "query_missing_required_fields_suppressed suppressed_count=%s total_missing=%s",
            skipped_missing_required_count - missing_warning_limit,
            skipped_missing_required_count,
        )

    query_df = pd.DataFrame(
        query_records, columns=["query_id", "query_text", "query_embedding"]
    )
    return ProcessResult(
        output_df=query_df,
        failures=failures,
        attempted_count=attempted_count,
        skipped_missing_required_count=skipped_missing_required_count,
    )
