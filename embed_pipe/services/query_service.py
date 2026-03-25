import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from embed_pipe.domain.models import ProcessResult, RuntimeConfig
from embed_pipe.domain.records import FailureRecord
from embed_pipe.domain.result import Result
from embed_pipe.infra.embedding_strategies import EmbeddingStrategy, ensure_embedding_shape
from embed_pipe.infra.jsonl_reader import JsonlReader


class QueryService:
    def __init__(
        self,
        embedding_strategy: EmbeddingStrategy,
        runtime: RuntimeConfig,
        jsonl_reader: JsonlReader | None = None,
    ):
        self.embedding_strategy = embedding_strategy
        self.runtime = runtime
        self.jsonl_reader = jsonl_reader or JsonlReader()
        self.logger = logging.getLogger("embed_pipe")

    def process(self, queries_path: Path) -> Result[ProcessResult]:
        query_ids: List[str] = []
        query_texts: List[str] = []
        query_records: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []
        attempted_count = 0

        for line_num, obj in self.jsonl_reader.read_objects(queries_path):
            query_id = str(obj.get("query_id", "")).strip()
            query_text = str(obj.get("query_text", "")).strip()
            if not query_id or not query_text:
                message = (
                    "Invalid queries input at line %s in %s: query_id/query_text must be non-empty (query_id=%r)"
                    % (line_num, queries_path, query_id)
                )
                self.logger.error(
                    "event=queries_validation_failed reason=%s context=%s",
                    message,
                    "queries_path=%s line_num=%s" % (queries_path, line_num),
                )
                return Result.failure(message)
            query_ids.append(query_id)
            query_texts.append(query_text)

        for query_id, query_text in zip(query_ids, query_texts):
            attempted_count += 1
            try:
                vectors = self.embedding_strategy.encode([query_text], is_query=True)
                vectors = ensure_embedding_shape(vectors, self.runtime.embedding_dim)
                query_records.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "query_embedding": vectors[0].tolist(),
                    }
                )
            except Exception as exc:
                failures.append(
                    FailureRecord(
                        record_type="query",
                        record_id=query_id,
                        error_message=str(exc),
                        stage="embedding",
                    ).to_dict()
                )

        query_df = pd.DataFrame(query_records, columns=["query_id", "query_text", "query_embedding"])
        return Result.success(ProcessResult(output_df=query_df, failures=failures, attempted_count=attempted_count))
