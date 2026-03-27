import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from embed_pipe.domain.exceptions import ProcessingError
from embed_pipe.domain.models import ProcessResult, RuntimeConfig
from embed_pipe.infra.jsonl_reader import JsonlReader
from embed_pipe.services.chunking.chunk_selector import ChunkSelector


class DocumentService:
    def __init__(self, selector: ChunkSelector, runtime: RuntimeConfig):
        self.selector = selector
        self.runtime = runtime
        self.jsonl_reader = JsonlReader()
        self.logger = logging.getLogger(__name__)

    def process(
        self, docs_path: Path, retry_mode: bool, retry_id_set: set[str]
    ) -> ProcessResult:
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
            if retry_mode and doc_id not in retry_id_set:
                continue
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
        return ProcessResult(output_df=docs_df, failures=[])
