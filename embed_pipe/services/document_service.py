from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from embed_pipe.domain.errors import ChunkSelectionError, EmbeddingGenerationError, InputValidationError
from embed_pipe.domain.models import ProcessResult, RuntimeConfig
from embed_pipe.domain.records import FailureRecord
from embed_pipe.infra.jsonl_reader import JsonlReader


class DocumentService:
    def __init__(self, selector, runtime: RuntimeConfig, jsonl_reader: JsonlReader | None = None):
        self.selector = selector
        self.runtime = runtime
        self.jsonl_reader = jsonl_reader or JsonlReader()

    def process(self, docs_path: Path, retry_mode: bool, retry_id_set: set[str]) -> ProcessResult:
        doc_records: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []
        attempted_count = 0

        for line_num, obj in self.jsonl_reader.read_objects(docs_path):
            doc_id = str(obj.get("doc_id", "")).strip()
            doc_text = str(obj.get("doc_text", "")).strip()
            if not doc_id or not doc_text:
                raise InputValidationError(
                    "Invalid docs input at line %s in %s: doc_id/doc_text must be non-empty (doc_id=%r)"
                    % (line_num, docs_path, doc_id)
                )
            if retry_mode and doc_id not in retry_id_set:
                continue

            attempted_count += 1
            try:
                selected = self.selector.select_chunks(doc_text, "")
                if not selected:
                    raise ChunkSelectionError("No chunk selected")

                first = selected[0]
                chunk_text = str(first.get("chunk_text") or "").strip()
                embedding = np.asarray(first.get("embedding", []), dtype=np.float32)
                if not chunk_text:
                    raise ChunkSelectionError("Selected chunk text is empty")
                if embedding.ndim != 1 or embedding.shape[0] != self.runtime.embedding_dim:
                    raise EmbeddingGenerationError(
                        "Selected chunk embedding dim mismatch expected=%s, got=%s"
                        % (self.runtime.embedding_dim, embedding.shape)
                    )

                doc_records.append(
                    {
                        "doc_id": doc_id,
                        "chunk_text": chunk_text,
                        "chunk_embedding": embedding.tolist(),
                    }
                )
            except Exception as exc:
                stage = "chunk_selection" if isinstance(exc, ChunkSelectionError) else "embedding"
                failure = FailureRecord(
                    record_type="doc",
                    record_id=doc_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    stage=stage,
                )
                failures.append(failure.to_dict())

        docs_df = pd.DataFrame(doc_records, columns=["doc_id", "chunk_text", "chunk_embedding"])
        return ProcessResult(output_df=docs_df, failures=failures, attempted_count=attempted_count)
