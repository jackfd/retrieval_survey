import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
import pandas as pd

from embedding.domain.exceptions import ProcessingError
from embedding.domain.models import DOC_COLUMNS, DOC_FLUSH_CHUNK_THRESHOLD

from embedding.infra.jsonl_reader import read_objects
from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.services.chunk_selector import ChunkSelector
from embedding.infra.output_writer import OutputWriter

logger = logging.getLogger(__name__)


def process_doc(
    embedding: EmbeddingStrategy, doc_path: Path, output: OutputWriter
) -> None:
    selector = ChunkSelector(embedding)
    doc_records: List[Dict[str, Any]] = []
    batch_doc_count = 0
    batch_elapsed_sec = 0.0
    total_chunk_count = 0

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
            doc_start = perf_counter()
            selected = selector.run(doc_text, doc_id)
            elapsed_sec = perf_counter() - doc_start
        except Exception as exc:
            logger.exception(
                "Document chunk selection failed docs_path=%s line_num=%s doc_id=%s error_type=%s",
                doc_path,
                line_num,
                doc_id,
                type(exc).__name__,
            )
            raise ProcessingError(
                "Document chunk selection failed docs_path=%s line_num=%s doc_id=%s"
                % (doc_path, line_num, doc_id)
            ) from exc

        doc_records.extend(selected)
        batch_doc_count += 1
        batch_elapsed_sec += elapsed_sec
        total_chunk_count += len(selected)

        if len(doc_records) >= DOC_FLUSH_CHUNK_THRESHOLD:
            batch_flush(output, doc_records, batch_doc_count, batch_elapsed_sec)
            doc_records = []
            batch_doc_count = 0
            batch_elapsed_sec = 0.0

    if len(doc_records) > 0:
        batch_flush(output, doc_records, batch_doc_count, batch_elapsed_sec)

    if total_chunk_count == 0:
        logger.error("No document chunks found in %s", doc_path)
        raise ProcessingError(f"No documents found in {doc_path}")


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


def batch_flush(
    output: OutputWriter,
    doc_records: List[Dict[str, Any]],
    batch_doc_count: int,
    batch_elapsed_sec: float,
) -> None:
    batch_df = pd.DataFrame(doc_records, columns=DOC_COLUMNS)
    output.write_docs(batch_df)
    batch_chunk_count = len(doc_records)
    avg_chunk_ms = batch_elapsed_sec * 1000.0 / batch_chunk_count
    logger.info(
        "doc batch: doc_count=%s chunk_count=%s avg_chunk_ms=%.3f",
        batch_doc_count,
        batch_chunk_count,
        avg_chunk_ms,
    )
