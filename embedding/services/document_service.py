import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
import pandas as pd

from embedding.domain.exceptions import ProcessingError
from embedding.domain.models import ProcessResult
from embedding.infra.jsonl_reader import JsonlReader
from embedding.services.chunk_selector import ChunkSelector

logger = logging.getLogger(__name__)


class DocumentService:
    LOG_EVERY_N = 1000

    def __init__(self, selector: ChunkSelector):
        self.selector = selector
        self.jsonl_reader = JsonlReader()

    def process(self, docs_path: Path) -> ProcessResult:
        doc_records: List[Dict[str, Any]] = []
        doc_counts = 0
        elapsed_sec = 0.0

        for line_num, obj in self.jsonl_reader.read_objects(docs_path):
            doc_id = str(obj.get("doc_id", "")).strip()
            doc_text = self._normalize_doc_text(obj.get("doc_text"))
            if not doc_id or not doc_text:
                logger.error(
                    "docs_path=%s line_num=%s doc_id=%r: doc_id/doc_text must be non-empty",
                    docs_path,
                    line_num,
                    doc_id,
                )
                raise ProcessingError(
                    "Invalid docs input docs_path=%s line_num=%s doc_id=%r"
                    % (docs_path, line_num, doc_id)
                )
            try:
                doc_start = perf_counter()
                selected = self.selector.run(doc_text, doc_id)
                doc_elapsed_sec = perf_counter() - doc_start
            except Exception as exc:
                self.logger.exception(
                    "Document chunk selection failed docs_path=%s line_num=%s doc_id=%s error_type=%s",
                    docs_path,
                    line_num,
                    doc_id,
                    type(exc).__name__,
                )
                raise ProcessingError(
                    "Document chunk selection failed docs_path=%s line_num=%s doc_id=%s"
                    % (docs_path, line_num, doc_id)
                ) from exc

            doc_records.extend(selected)
            doc_counts += 1
            elapsed_sec += doc_elapsed_sec
            if doc_counts >= self.LOG_EVERY_N:
                log_doc_process_timing(docs_path, doc_counts, elapsed_sec)
                doc_counts = 0
                elapsed_sec = 0.0

        if doc_counts > 0:
            log_doc_process_timing(docs_path, doc_counts, elapsed_sec)

        docs_df = pd.DataFrame(
            doc_records,
            columns=[
                "doc_id",
                "chunk_id",
                "chunk_text",
                "chunk_vector",
                "chunk_score",
                "chunk_rank",
            ],
        )
        self.selector.flush_embedding_timing()
        return ProcessResult(output_df=docs_df)

    def _normalize_doc_text(self, value: Any) -> List[str]:
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


def log_doc_process_timing(docs_path: Path, docs: int, elapsed_sec: float) -> None:
    avg_ms = (elapsed_sec * 1000.0 / docs) if docs > 0 else 0.0
    logger.info("docs=%s avg_doc_process_ms=%.3f docs_path=%s", docs, avg_ms, docs_path)
