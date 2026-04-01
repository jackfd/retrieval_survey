import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from embedding.domain.exceptions import ProcessingError
from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.infra.jsonl_reader import read_objects
from embedding.infra.output_writer import OutputWriter

logger = logging.getLogger(__name__)


def _flush_batch(
    embedding: EmbeddingStrategy,
    batch: List[Dict[str, Any]],
    output: OutputWriter,
) -> None:
    start_line_num = int(batch[0]["line_num"])
    end_line_num = int(batch[-1]["line_num"])
    batch_texts = [str(item["query_text"]) for item in batch]

    try:
        embed_start = perf_counter()
        vectors = embedding.encode(batch_texts, is_query=True)
        elapsed_sec = perf_counter() - embed_start
    except Exception as exc:
        logger.exception(
            "Query embedding batch failed start_line_num=%s end_line_num=%s query_count=%s error_type=%s",
            start_line_num,
            end_line_num,
            len(batch),
            type(exc).__name__,
        )
        raise ProcessingError(
            "Query embedding batch failed start_line_num=%s end_line_num=%s query_count=%s"
            % (start_line_num, end_line_num, len(batch))
        ) from exc

    if len(vectors) != len(batch):
        logger.error(
            "Query embedding batch size mismatch, start_line_num=%s end_line_num=%s expected=%s actual=%s",
            start_line_num,
            end_line_num,
            len(batch),
            len(vectors),
        )
        raise ProcessingError(
            "Query embedding batch size mismatch start_line_num=%s end_line_num=%s expected=%s actual=%s"
            % (start_line_num, end_line_num, len(batch), len(vectors))
        )

    actual_batch_size = len(batch)
    avg_ms = elapsed_sec * 1000.0 / actual_batch_size
    logger.info(
        "query batch encoded batch_size=%s avg_ms_per_query=%.3f start_line_num=%s",
        actual_batch_size,
        avg_ms,
        start_line_num,
    )

    batch_records = []
    for item, vector in zip(batch, vectors):
        batch_records.append(
            {
                "query_id": item["query_id"],
                "query_text": item["query_text"],
                "query_embedding": vector.tolist(),
            }
        )
    output.write_query_chunks(batch_records)


def process_query(
    embedding: EmbeddingStrategy,
    batch_size: int,
    queries_path: Path,
    output: OutputWriter,
) -> None:
    pending_batch: List[Dict[str, Any]] = []
    total_query_count = 0

    for line_num, obj in read_objects(queries_path):
        query_id = str(obj.get("query_id", "")).strip()
        query_text = str(obj.get("query_text", "")).strip()
        if not query_id or not query_text:
            logger.error(
                "Invalid query input queries_path=%s line_num=%s query_id=%r: query_id/query_text must be non-empty",
                queries_path,
                line_num,
                query_id,
            )
            raise ProcessingError(
                "Invalid query input queries_path=%s line_num=%s query_id=%r"
                % (queries_path, line_num, query_id)
            )

        pending_batch.append(
            {"line_num": line_num, "query_id": query_id, "query_text": query_text}
        )
        if len(pending_batch) < batch_size:
            continue

        _flush_batch(embedding, pending_batch, output)
        total_query_count += len(pending_batch)
        pending_batch = []

    if pending_batch:
        _flush_batch(embedding, pending_batch, output)
        total_query_count += len(pending_batch)

    logger.info("query summary: query_count=%s", total_query_count)
