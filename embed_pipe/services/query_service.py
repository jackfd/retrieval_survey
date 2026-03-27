import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from embed_pipe.domain.exceptions import ProcessingError
from embed_pipe.domain.models import ProcessResult, RuntimeConfig
from embed_pipe.infra.embedding_strategies import EmbeddingStrategy
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
        self.logger = logging.getLogger(__name__)

    def process(self, queries_path: Path) -> ProcessResult:
        records: List[Dict[str, Any]] = []

        for line_num, obj in self.jsonl_reader.read_objects(queries_path):
            query_id = str(obj.get("query_id", "")).strip()
            query_text = str(obj.get("query_text", "")).strip()
            if not query_id or not query_text:
                self.logger.error(
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
                vectors = self.embedding_strategy.encode([query_text], is_query=True)
                records.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "query_embedding": vectors[0].tolist(),
                    }
                )
            except Exception as exc:
                self.logger.exception(
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

        df = pd.DataFrame(
            records, columns=["query_id", "query_text", "query_embedding"]
        )
        return ProcessResult(output_df=df, failures=[])
