import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from embed_pipe.domain.models import ProcessResult, RuntimeConfig
from embed_pipe.domain.records import FailureRecord
from embed_pipe.domain.result import Result
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

    def process(self, queries_path: Path) -> Result[ProcessResult]:
        records: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []

        for line_num, obj in self.jsonl_reader.read_objects(queries_path):
            query_id = str(obj.get("query_id", "")).strip()
            query_text = str(obj.get("query_text", "")).strip()
            if not query_id or not query_text:
                self.logger.error(
                    f"query id or text is empty at {queries_path}+{line_num}"
                )
                return Result.failure()
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
                failures.append(
                    FailureRecord(
                        record_type="query",
                        record_id=query_id,
                        error_message=str(exc),
                        stage="embedding",
                    ).to_dict()
                )

        df = pd.DataFrame(
            records, columns=["query_id", "query_text", "query_embedding"]
        )
        return Result.success(ProcessResult(output_df=df, failures=failures))
