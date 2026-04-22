import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Tuple

from embedding.domain.exceptions import ProcessingError
from embedding.adapters.backends import EmbeddingStrategy
from embedding.adapters.jsonl_reader import read_objects
from embedding.adapters.output_writer import OutputWriter

logger = logging.getLogger(__name__)
QUERY_WINDOW_SIZE = 5000


def process_query(
    embedding: EmbeddingStrategy,
    queries_path: Path,
    output: OutputWriter,
) -> None:
    read_iter = iter(read_objects(queries_path))
    total_query_count = 0

    while True:
        query_window = _collect_query_window(read_iter, queries_path, QUERY_WINDOW_SIZE)
        if not query_window:
            break

        encoded_records = _encode_query_window(embedding, query_window)
        output.write_query_chunks(encoded_records)
        total_query_count += len(query_window)

    logger.info("query summary, path:%s, count=%s", queries_path, total_query_count)


def _collect_query_window(
    read_iter: Iterator[Tuple[int, Dict[str, Any]]],
    queries_path: Path,
    window_size: int,
) -> List[Dict[str, Any]]:
    query_window: List[Dict[str, Any]] = []

    while len(query_window) < window_size:
        try:
            line_num, obj = next(read_iter)
        except StopIteration:
            break

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

        query_window.append(
            {"line_num": line_num, "query_id": query_id, "query_text": query_text}
        )

    return query_window


def _encode_query_window(
    embedding: EmbeddingStrategy, query_window: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    start_line_num = int(query_window[0]["line_num"])
    end_line_num = int(query_window[-1]["line_num"])
    windows_texts = [str(item["query_text"]) for item in query_window]

    try:
        embed_start = perf_counter()
        vectors = embedding.encode(windows_texts, is_query=True)
        elapsed_sec = perf_counter() - embed_start
    except Exception as exc:
        logger.exception(
            "Query embedding batch failed start_line_num=%s end_line_num=%s query_count=%s error_type=%s",
            start_line_num,
            end_line_num,
            len(query_window),
            type(exc).__name__,
        )
        raise ProcessingError(
            "Query embedding batch failed start_line_num=%s end_line_num=%s query_count=%s"
            % (start_line_num, end_line_num, len(query_window))
        ) from exc

    if len(vectors) != len(query_window):
        logger.error(
            "Query embedding batch size mismatch, start_line_num=%s end_line_num=%s expected=%s actual=%s",
            start_line_num,
            end_line_num,
            len(query_window),
            len(vectors),
        )
        raise ProcessingError(
            "Query embedding batch size mismatch start_line_num=%s end_line_num=%s expected=%s actual=%s"
            % (start_line_num, end_line_num, len(query_window), len(vectors))
        )

    actual_batch_size = len(query_window)
    avg_ms = elapsed_sec * 1000.0 / actual_batch_size
    logger.info(
        "   query embedding size=%s avg_ms=%.3f start_line_num=%s",
        actual_batch_size,
        avg_ms,
        start_line_num,
    )

    encoded_records: List[Dict[str, Any]] = []
    for item, vector in zip(query_window, vectors):
        encoded_records.append(
            {
                "query_id": item["query_id"],
                "query_text": item["query_text"],
                "query_embedding": vector.tolist(),
            }
        )
    return encoded_records
