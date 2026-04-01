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
from embedding.services.chunk_selector import ChunkSelector

logger = logging.getLogger(__name__)


def _drain_completed_docs(
    selector: ChunkSelector,
    pending_docs: Deque[Dict[str, Any]],
    output: OutputWriter,
) -> int:
    """
    处理已完成嵌入的文档并将其写入输出

    该函数会从待处理文档队列中取出已完成嵌入的文档（remaining_count为0），
    验证嵌入向量完整性，使用选择器选择合适的文档块，并将其写入输出。
    """
    selected_chunk_count = 0
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
            # 使用selector从嵌入向量中选择文档块
            selected = selector.select_from_embeddings(
                context["doc_id"], context["candidates"], embeddings
            )
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
            output.write_doc_chunks(selected)
            selected_chunk_count += len(selected)
    return selected_chunk_count


def _flush_batch(
    embedding: EmbeddingStrategy,
    selector: ChunkSelector,
    pending_docs: Deque[Dict[str, Any]],
    pending_chunk_items: List[Dict[str, Any]],
    batch_size: int,
    output: OutputWriter,
) -> tuple[float, int]:
    batch_items = pending_chunk_items[:batch_size]
    del pending_chunk_items[:batch_size]

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

    selected_chunk_count = _drain_completed_docs(selector, pending_docs, output)
    return elapsed_sec, selected_chunk_count


def process_doc(
    embedding: EmbeddingStrategy, batch_size: int, doc_path: Path, output: OutputWriter
) -> None:
    # 初始化分块选择器
    selector = ChunkSelector()
    total_doc_count = 0
    total_chunk_count = 0
    total_elapsed_sec = 0.0
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
            # 构建文档文本的候选块
            candidates = selector.build_candidates(doc_text)
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

        # 当待处理项目数达到批处理大小时，执行批量嵌入
        while len(pending_chunk_items) >= batch_size:
            elapsed_sec, selected_chunks = _flush_batch(
                embedding=embedding,
                selector=selector,
                pending_docs=pending_docs,
                pending_chunk_items=pending_chunk_items,
                batch_size=batch_size,
                output=output,
            )
            total_elapsed_sec += elapsed_sec
            total_chunk_count += selected_chunks

        if line_num % 10000 == 0:
            logger.info(
                "chunk_count=%s elapsed_ms=%.3f", selected_chunks, elapsed_sec * 1000.0
            )

    # 处理剩余的待处理项目（最后不足一个批次的数据）
    while pending_chunk_items:
        elapsed_sec, selected_chunks = _flush_batch(
            embedding=embedding,
            selector=selector,
            pending_docs=pending_docs,
            pending_chunk_items=pending_chunk_items,
            batch_size=min(batch_size, len(pending_chunk_items)),
            output=output,
        )
        total_elapsed_sec += elapsed_sec
        total_chunk_count += selected_chunks

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

    avg_chunk_ms = total_elapsed_sec * 1000.0 / total_chunk_count
    logger.info(
        "doc processing completed: total doc:%s chunk:%s avg_chunk_ms=%.3f",
        total_doc_count,
        total_chunk_count,
        avg_chunk_ms,
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
