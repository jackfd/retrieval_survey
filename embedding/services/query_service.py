import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
import pandas as pd

from embedding.domain.exceptions import ProcessingError
from embedding.domain.models import QUERY_COLUMNS, LOG_EVERY_N
from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.infra.jsonl_reader import read_objects
from embedding.infra.output_writer import OutputWriter

logger = logging.getLogger(__name__)


def process_query(
    embedding: EmbeddingStrategy, queries_path: Path, output: OutputWriter
) -> None:

    records: List[Dict[str, Any]] = []
    query_counts = 0
    elapsed_sec = 0.0

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
        try:
            embed_start = perf_counter()
            vectors = embedding.encode([query_text], is_query=True)
            embed_elapsed_sec = perf_counter() - embed_start
            records.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_embedding": vectors[0].tolist(),
                }
            )
            query_counts += 1
            elapsed_sec += embed_elapsed_sec
            if query_counts >= LOG_EVERY_N:
                log_query_embedding_timing(queries_path, query_counts, elapsed_sec)
                query_counts = 0
                elapsed_sec = 0.0
        except Exception as exc:
            logger.exception(
                "Query embedding failed queries_path=%s line_num=%s query_id=%s error_type=%s",
                queries_path,
                line_num,
                query_id,
                type(exc).__name__,
            )
            raise ProcessingError(
                "Query embedding failed queries_path=%s line_num=%s query_id=%s"
                % (queries_path, line_num, query_id)
            ) from exc

    if query_counts > 0:
        log_query_embedding_timing(queries_path, query_counts, elapsed_sec)

    df = pd.DataFrame(records, QUERY_COLUMNS)
    output.write_queries(df)
    logger.info("query summary: query_count=%s", len(df))


def log_query_embedding_timing(
    queries_path: Path, queries: int, elapsed_sec: float
) -> None:
    avg_ms = (elapsed_sec * 1000.0 / queries) if queries > 0 else 0.0
    logger.info(
        "query counts=%s  avg_ms=%.3f queries=%s", queries, avg_ms, queries_path
    )
