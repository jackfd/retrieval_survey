import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from embedding.domain.exceptions import ProcessingError
from embedding.domain.models import ProcessResult
from embedding.infra.jsonl_reader import JsonlReader
from embedding.services.chunking.chunk_selector import ChunkSelector


class DocumentService:
    def __init__(self, selector: ChunkSelector):
        self.selector = selector
        self.jsonl_reader = JsonlReader()
        self.logger = logging.getLogger(__name__)

    def process(self, docs_path: Path) -> ProcessResult:
        doc_records: List[Dict[str, Any]] = []

        for line_num, obj in self.jsonl_reader.read_objects(docs_path):
            doc_id = str(obj.get("doc_id", "")).strip()
            doc_text = str(obj.get("doc_text", "")).strip()
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
                selected = self.selector.select_chunks(doc_text)
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

            first = selected[0]
            doc_records.append(
                {
                    "doc_id": doc_id,
                    "chunk_text": first.get("chunk_text"),
                    "chunk_embedding": first.get("embedding", []),
                }
            )

        docs_df = pd.DataFrame(
            doc_records, columns=["doc_id", "chunk_text", "chunk_embedding"]
        )
        return ProcessResult(output_df=docs_df)
