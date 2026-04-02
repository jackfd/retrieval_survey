import logging
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any, Deque, Dict, List

import numpy as np
from embedding.domain.exceptions import ProcessingError

from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.infra.jsonl_reader import read_objects
from embedding.infra.output_writer import OutputWriter
from embedding.services.chunk_selector import build_candidates, select_from_embeddings
from embedding.services.chunk_splitter import ChunkSplitter

logger = logging.getLogger(__name__)
ENABLE_DOC_PERF_LOG = True


def _log_doc_batch_perf(
    batch_stats: Dict[str, Any],
    pending_docs: Deque[Dict[str, Any]],
    pending_chunk_items: List[Dict[str, Any]],
) -> None:
    if not ENABLE_DOC_PERF_LOG:
        return

    embedded_chunk_count = int(batch_stats["embedded_chunk_count"])
    embedding_elapsed_ms = float(batch_stats["embedding_elapsed_sec"]) * 1000.0
    avg_ms_per_chunk = (
        embedding_elapsed_ms / embedded_chunk_count if embedded_chunk_count > 0 else 0.0
    )
    logger.info(
        "doc_batch_perf chunk_count=%s elapsed_ms=%.3f avg_ms=%.3f selected_chunk_count=%s completed_doc_count=%s pending_doc_count_after=%s pending_chunk_count_after=%s start_doc_id=%s end_doc_id=%s",
        embedded_chunk_count,
        embedding_elapsed_ms,
        avg_ms_per_chunk,
        int(batch_stats["selected_chunk_count"]),
        int(batch_stats["completed_doc_count"]),
        len(pending_docs),
        len(pending_chunk_items),
        batch_stats["start_doc_id"],
        batch_stats["end_doc_id"],
    )


def _log_doc_periodic_perf(
    *,
    total_doc_count: int,
    candidate_chunk_count_total: int,
    total_chunk_count: int,
    embedding_batch_count_total: int,
    candidate_build_sec_total: float,
    embedding_sec_total: float,
    selection_sec_total: float,
    write_sec_total: float,
    pending_docs: Deque[Dict[str, Any]],
    pending_chunk_items: List[Dict[str, Any]],
) -> None:
    if not ENABLE_DOC_PERF_LOG:
        return

    logger.info(
        "doc_periodic_perf doc_count=%s candidate_chunk_count=%s selected_chunk_count=%s embedding_batch_count=%s candidate_build_ms_total=%.3f embedding_ms_total=%.3f selection_ms_total=%.3f write_ms_total=%.3f pending_doc_count=%s pending_chunk_count=%s",
        total_doc_count,
        candidate_chunk_count_total,
        total_chunk_count,
        embedding_batch_count_total,
        candidate_build_sec_total * 1000.0,
        embedding_sec_total * 1000.0,
        selection_sec_total * 1000.0,
        write_sec_total * 1000.0,
        len(pending_docs),
        len(pending_chunk_items),
    )


def _log_doc_summary_perf(
    *,
    total_doc_count: int,
    candidate_chunk_count_total: int,
    total_chunk_count: int,
    embedding_batch_count_total: int,
    candidate_build_sec_total: float,
    embedding_sec_total: float,
    selection_sec_total: float,
    write_sec_total: float,
    wall_sec_total: float,
) -> None:
    if not ENABLE_DOC_PERF_LOG:
        return

    logger.info(
        "doc_summary_perf doc_count=%s candidate_chunk_count=%s selected_chunk_count=%s embedding_batch_count=%s candidate_build_ms_total=%.3f embedding_ms_total=%.3f selection_ms_total=%.3f write_ms_total=%.3f docs_per_sec=%.3f selected_chunks_per_sec=%.3f",
        total_doc_count,
        candidate_chunk_count_total,
        total_chunk_count,
        embedding_batch_count_total,
        candidate_build_sec_total * 1000.0,
        embedding_sec_total * 1000.0,
        selection_sec_total * 1000.0,
        write_sec_total * 1000.0,
        (total_doc_count / wall_sec_total) if wall_sec_total > 0 else 0.0,
        (total_chunk_count / wall_sec_total) if wall_sec_total > 0 else 0.0,
    )


def _drain_completed_docs(
    pending_docs: Deque[Dict[str, Any]],
    output: OutputWriter,
) -> Dict[str, Any]:
    """
    处理已完成嵌入的文档并将其写入输出

    该函数会从待处理文档队列中取出已完成嵌入的文档（remaining_count为0），
    验证嵌入向量完整性，使用选择器选择合适的文档块，并将其写入输出。
    """
    selected_chunk_count = 0
    completed_doc_count = 0
    selection_elapsed_sec = 0.0
    write_elapsed_sec = 0.0
    # 检查是否有待处理文档且第一个文档的剩余计数为0（表示嵌入完成）
    while pending_docs and pending_docs[0]["remaining_count"] == 0:
        context = pending_docs.popleft()
        vectors = context["embedded_vectors"]
        # 检查嵌入向量是否完整
        if any(vector is None for vector in vectors):
            raise ProcessingError(
                "Document embeddings incomplete doc_id=%s line_num=%s"
                % (context["doc_id"], context["line_num"])
            )
        # 将向量转换为numpy数组格式
        embeddings = np.asarray(vectors, dtype=np.float32)
        try:
            # 使用选择函数从嵌入向量中选择文档块
            select_start = perf_counter()
            selected = select_from_embeddings(
                context["doc_id"],
                context["candidates"],
                embeddings,
            )
            selection_elapsed_sec += perf_counter() - select_start
        except Exception as exc:
            logger.exception(
                "Document chunk selection failed doc_id=%s line_num=%s error_type=%s",
                context["doc_id"],
                context["line_num"],
                type(exc).__name__,
            )
            raise ProcessingError(
                "Document chunk selection failed doc_id=%s line_num=%s"
                % (context["doc_id"], context["line_num"])
            ) from exc
        # 如果有选中的文档块，则写入输出并更新计数
        if selected:
            write_start = perf_counter()
            output.write_doc_chunks(selected)
            write_elapsed_sec += perf_counter() - write_start
            selected_chunk_count += len(selected)
        completed_doc_count += 1
    return {
        "selected_chunk_count": selected_chunk_count,
        "completed_doc_count": completed_doc_count,
        "selection_elapsed_sec": selection_elapsed_sec,
        "write_elapsed_sec": write_elapsed_sec,
    }


def _flush_batch(
    embedding: EmbeddingStrategy,
    pending_docs: Deque[Dict[str, Any]],
    pending_chunk_items: List[Dict[str, Any]],
    output: OutputWriter,
) -> Dict[str, Any]:
    batch_items = list(pending_chunk_items)
    pending_chunk_items.clear()

    batch_texts = [str(item["text"]) for item in batch_items]
    start_doc_id = str(batch_items[0]["doc_id"])
    end_doc_id = str(batch_items[-1]["doc_id"])

    try:
        embed_start = perf_counter()
        vectors = embedding.encode(batch_texts, is_query=False)
        elapsed_sec = perf_counter() - embed_start
    except Exception as exc:
        logger.exception(
            "Document embedding batch failed start_doc_id=%s end_doc_id=%s chunk_count=%s error_type=%s",
            start_doc_id,
            end_doc_id,
            len(batch_items),
            type(exc).__name__,
        )
        raise ProcessingError(
            "Document embedding batch failed start_doc_id=%s end_doc_id=%s chunk_count=%s"
            % (start_doc_id, end_doc_id, len(batch_items))
        ) from exc

    if len(vectors) != len(batch_items):
        logger.error(
            "Document embedding batch size mismatch start_doc_id=%s end_doc_id=%s expected=%s actual=%s",
            start_doc_id,
            end_doc_id,
            len(batch_items),
            len(vectors),
        )
        raise ProcessingError(
            "Document embedding batch size mismatch start_doc_id=%s end_doc_id=%s expected=%s actual=%s"
            % (start_doc_id, end_doc_id, len(batch_items), len(vectors))
        )

    for item, vector in zip(batch_items, vectors):
        context = item["context"]
        candidate_index = int(item["candidate_index"])
        context["embedded_vectors"][candidate_index] = np.asarray(
            vector, dtype=np.float32
        )
        context["remaining_count"] -= 1

    drain_stats = _drain_completed_docs(pending_docs, output)
    return {
        "embedding_elapsed_sec": elapsed_sec,
        "selection_elapsed_sec": float(drain_stats["selection_elapsed_sec"]),
        "write_elapsed_sec": float(drain_stats["write_elapsed_sec"]),
        "embedded_chunk_count": len(batch_items),
        "selected_chunk_count": int(drain_stats["selected_chunk_count"]),
        "completed_doc_count": int(drain_stats["completed_doc_count"]),
        "start_doc_id": start_doc_id,
        "end_doc_id": end_doc_id,
    }


def _accumulate_doc_batch_stats(
    *,
    batch_stats: Dict[str, Any],
    embedding_batch_count_total: int,
    embedding_sec_total: float,
    selection_sec_total: float,
    write_sec_total: float,
    total_chunk_count: int,
) -> Dict[str, Any]:
    return {
        "embedding_batch_count_total": embedding_batch_count_total + 1,
        "embedding_sec_total": embedding_sec_total
        + float(batch_stats["embedding_elapsed_sec"]),
        "selection_sec_total": selection_sec_total
        + float(batch_stats["selection_elapsed_sec"]),
        "write_sec_total": write_sec_total + float(batch_stats["write_elapsed_sec"]),
        "total_chunk_count": total_chunk_count
        + int(batch_stats["selected_chunk_count"]),
    }


def process_doc(
    embedding: EmbeddingStrategy, doc_path: Path, output: OutputWriter
) -> None:
    splitter = ChunkSplitter()
    total_doc_count = 0
    total_chunk_count = 0
    candidate_chunk_count_total = 0
    embedding_batch_count_total = 0
    candidate_build_sec_total = 0.0
    embedding_sec_total = 0.0
    selection_sec_total = 0.0
    write_sec_total = 0.0
    task_start = perf_counter()
    # 存储待处理文档上下文的双端队列
    pending_docs: Deque[Dict[str, Any]] = deque()
    # 存储待嵌入的文本块列表
    pending_chunk_items: List[Dict[str, Any]] = []

    # 遍历输入文档中的每行数据
    for line_num, obj in read_objects(doc_path):
        doc_id = str(obj.get("doc_id", "")).strip()
        doc_text = _normalize_doc_text(obj.get("doc_text"))
        if not doc_id or not doc_text:
            logger.error(
                "docs_path=%s line_num=%s id=%r, id/text must be non-empty",
                doc_path,
                line_num,
                doc_id,
            )
            raise ProcessingError(
                "Invalid docs input docs_path=%s line_num=%s doc_id=%r"
                % (doc_path, line_num, doc_id)
            )
        try:
            build_start = perf_counter()
            candidates = build_candidates(doc_text, splitter)
            candidate_build_sec_total += perf_counter() - build_start
        except Exception as exc:
            logger.exception(
                "Document candidate preparation failed docs_path=%s line_num=%s doc_id=%s error_type=%s",
                doc_path,
                line_num,
                doc_id,
                type(exc).__name__,
            )
            raise ProcessingError(
                "Document candidate preparation failed docs_path=%s line_num=%s doc_id=%s"
                % (doc_path, line_num, doc_id)
            ) from exc

        total_doc_count += 1
        if candidates:
            candidate_chunk_count_total += len(candidates)
            # 创建文档上下文信息
            context: Dict[str, Any] = {
                "doc_id": doc_id,
                "line_num": line_num,
                "candidates": candidates,
                "remaining_count": len(candidates),
                "embedded_vectors": [None] * len(candidates),
            }
            pending_docs.append(context)
            # 添加候选块到待处理项列表
            for candidate_index, candidate in enumerate(candidates):
                pending_chunk_items.append(
                    {
                        "doc_id": doc_id,
                        "candidate_index": candidate_index,
                        "text": str(candidate["text"]),
                        "context": context,
                    }
                )

        # 服务层不做 batch_size 切分，策略层负责分批。
        if pending_chunk_items:
            batch_stats = _flush_batch(
                embedding=embedding,
                pending_docs=pending_docs,
                pending_chunk_items=pending_chunk_items,
                output=output,
            )
            totals = _accumulate_doc_batch_stats(
                batch_stats=batch_stats,
                embedding_batch_count_total=embedding_batch_count_total,
                embedding_sec_total=embedding_sec_total,
                selection_sec_total=selection_sec_total,
                write_sec_total=write_sec_total,
                total_chunk_count=total_chunk_count,
            )
            embedding_batch_count_total = int(totals["embedding_batch_count_total"])
            embedding_sec_total = float(totals["embedding_sec_total"])
            selection_sec_total = float(totals["selection_sec_total"])
            write_sec_total = float(totals["write_sec_total"])
            total_chunk_count = int(totals["total_chunk_count"])
            _log_doc_batch_perf(
                batch_stats=batch_stats,
                pending_docs=pending_docs,
                pending_chunk_items=pending_chunk_items,
            )

        if total_doc_count % 1000 == 0:
            _log_doc_periodic_perf(
                total_doc_count=total_doc_count,
                candidate_chunk_count_total=candidate_chunk_count_total,
                total_chunk_count=total_chunk_count,
                embedding_batch_count_total=embedding_batch_count_total,
                candidate_build_sec_total=candidate_build_sec_total,
                embedding_sec_total=embedding_sec_total,
                selection_sec_total=selection_sec_total,
                write_sec_total=write_sec_total,
                pending_docs=pending_docs,
                pending_chunk_items=pending_chunk_items,
            )

    # 检查是否还有未完成嵌入的文档
    if pending_docs:
        first_pending = pending_docs[0]
        raise ProcessingError(
            "Document embeddings incomplete doc_id=%s line_num=%s"
            % (first_pending["doc_id"], first_pending["line_num"])
        )

    # 关闭输出流并记录最终统计信息
    output.close()
    if total_chunk_count == 0:
        logger.error("No document chunks found in %s", doc_path)
        raise ProcessingError(f"No documents found in {doc_path}")

    wall_sec_total = perf_counter() - task_start
    _log_doc_summary_perf(
        total_doc_count=total_doc_count,
        candidate_chunk_count_total=candidate_chunk_count_total,
        total_chunk_count=total_chunk_count,
        embedding_batch_count_total=embedding_batch_count_total,
        candidate_build_sec_total=candidate_build_sec_total,
        embedding_sec_total=embedding_sec_total,
        selection_sec_total=selection_sec_total,
        write_sec_total=write_sec_total,
        wall_sec_total=wall_sec_total,
    )


def _normalize_doc_text(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, list):
        normalized: List[str] = []
        for item in value:
            sentence = str(item).strip()
            if sentence:
                normalized.append(sentence)
        return normalized

    return []
