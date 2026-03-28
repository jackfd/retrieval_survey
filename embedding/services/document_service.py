import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from embedding.domain.exceptions import ProcessingError
from embedding.domain.models import ProcessResult
from embedding.infra.jsonl_reader import JsonlReader
from embedding.services.chunk_selector import ChunkSelector


class DocumentService:
    def __init__(self, selector: ChunkSelector):
        self.selector = selector
        self.jsonl_reader = JsonlReader()
        self.logger = logging.getLogger(__name__)

    def process(self, docs_path: Path) -> ProcessResult:
        doc_records: List[Dict[str, Any]] = []

        for line_num, obj in self.jsonl_reader.read_objects(docs_path):
            doc_id = str(obj.get("doc_id", "")).strip()
            doc_text = self._normalize_doc_text(obj.get("doc_text"))
            if not doc_id or not doc_text:
                self.logger.error(
                    "Invalid docs input docs_path=%s line_num=%s doc_id=%r: doc_id/doc_text must be non-empty",
                    docs_path,
                    line_num,
                    doc_id,
                )
                raise ProcessingError(
                    "Invalid docs input docs_path=%s line_num=%s doc_id=%r"
                    % (docs_path, line_num, doc_id)
                )
            try:
                selected = self.selector.run(doc_text, doc_id)
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
