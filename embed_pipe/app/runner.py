from pathlib import Path
import logging
import pandas as pd
from typing import Any, Dict

from embed_pipe.domain.models import BuilderConfig, DatasetContext
from embed_pipe.domain.result import Result
from embed_pipe.infra.embedding_strategies import EmbeddingStrategy
from embed_pipe.infra.logger import utc_now_iso
from embed_pipe.infra.output_writer import OutputWriter
from embed_pipe.services.chunking import ChunkSelector, SelectorConfig
from embed_pipe.services.document_service import DocumentService
from embed_pipe.services.query_service import QueryService


class BuilderRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        dataset_ctx: DatasetContext,
        builder_cfg: BuilderConfig,
        embedding_strategy: EmbeddingStrategy,
    ):
        self.output_writer = OutputWriter(output_dir)
        self.dataset_ctx = dataset_ctx
        self.builder_cfg = builder_cfg
        self.embedding_strategy = embedding_strategy
        self.logger = logging.getLogger("embed_pipe")

    def run(self) -> Result[None]:
        run_start = utc_now_iso()

        runtime = self.builder_cfg.runtime
        retry_id_set = self.output_writer.load_failed_doc_ids()
        retry_mode = len(retry_id_set) > 0

        selector = ChunkSelector(
            embedding_strategy=self.embedding_strategy,
            chunk_num=1,
            config=SelectorConfig(batch_size=runtime.batch_size),
        )
        document_service = DocumentService(selector=selector, runtime=runtime)
        doc_result = document_service.process(
            docs_path=self.dataset_ctx.docs_path,
            retry_mode=retry_mode,
            retry_id_set=retry_id_set,
        )
        if not doc_result.ok:
            return Result.failure(doc_result.error_message or "document processing failed")

        doc_value = doc_result.value
        if doc_value is None:
            self.logger.error("Document service returned empty result")
            return Result.failure("document service returned empty result")

        existing_docs_df = self.output_writer.load_existing_docs()
        merged_docs_df = self.output_writer.merge_docs_with_retry(
            new_docs_df=doc_value.output_df,
            existing_docs_df=existing_docs_df,
            retry_mode=retry_mode,
            retry_id_set=retry_id_set,
        )

        query_service = QueryService(self.embedding_strategy, runtime)
        query_result = query_service.process(self.dataset_ctx.queries_path)
        if not query_result.ok:
            return Result.failure(query_result.error_message or "query processing failed")

        query_value = query_result.value
        if query_value is None:
            self.logger.error("Query service returned empty result")
            return Result.failure("query service returned empty result")

        all_failures = doc_value.failures + query_value.failures

        metadata = build_run_metadata(
            run_start=run_start,
            retry_mode=retry_mode,
            dataset_ctx=self.dataset_ctx,
            builder_cfg=self.builder_cfg,
            merged_docs_df=merged_docs_df,
            query_df=query_value.output_df,
            attempted_docs=doc_value.attempted_count,
            attempted_queries=query_value.attempted_count,
            doc_failures_count=len(doc_value.failures),
            query_failures_count=len(query_value.failures),
        )
        try:
            self.output_writer.write_all_outputs(
                docs_df=merged_docs_df,
                queries_df=query_value.output_df,
                failures=all_failures,
                metadata=metadata,
            )
        except Exception as exc:
            self.logger.error(
                "event=write_outputs_failed reason=%s context=%s",
                exc,
                "output_dir=%s" % self.output_writer.output_dir,
            )
            return Result.failure(str(exc))

        self.logger.info(
            "end: attempted_doc_count=%s attempted_query_count=%s doc_count=%s query_count=%s "
            "doc_failure_count=%s query_failure_count=%s retry_mode=%s",
            doc_value.attempted_count,
            query_value.attempted_count,
            len(merged_docs_df),
            len(query_value.output_df),
            len(doc_value.failures),
            len(query_value.failures),
            retry_mode,
        )
        return Result.success(None)


def build_run_metadata(
    run_start: str,
    retry_mode: bool,
    dataset_ctx: DatasetContext,
    builder_cfg: BuilderConfig,
    merged_docs_df: pd.DataFrame,
    query_df: pd.DataFrame,
    attempted_docs: int,
    attempted_queries: int,
    doc_failures_count: int,
    query_failures_count: int,
) -> Dict[str, Any]:
    runtime = builder_cfg.runtime
    model = builder_cfg.model
    dataset_meta = dataset_ctx.dataset_meta
    run_end = utc_now_iso()
    return {
        "run": {
            "start_time_utc": run_start,
            "end_time_utc": run_end,
            "retry_mode": retry_mode,
            "resolved_dataset_dir": str(dataset_ctx.resolved_dataset_dir),
        },
        "dataset_metadata": {
            "dataset_name": dataset_meta.get("dataset_name", ""),
            "version": dataset_meta.get("version", ""),
            "subset": dataset_meta.get("subset", ""),
            "task": dataset_meta.get("task", ""),
            "domain": (dataset_meta.get("metadata") or {}).get("domain", ""),
            "language": (dataset_meta.get("metadata") or {}).get("language", ""),
        },
        "model": {
            "model_name": model.model_name,
            "provider": model.provider,
            "model_id": model.model_id,
        },
        "runtime_config": {
            "embedding_dim": runtime.embedding_dim,
            "normalize_embeddings": runtime.normalize_embeddings,
            "max_length": runtime.max_length,
            "query_prefix": runtime.query_prefix,
            "doc_prefix": runtime.doc_prefix,
            "instruction_template": runtime.instruction_template,
            "batch_size": runtime.batch_size,
            "device": runtime.device,
            "embedding_api_url": runtime.embedding_api_url,
        },
        "stats": {
            "attempted_doc_count": attempted_docs,
            "attempted_query_count": attempted_queries,
            "doc_count": int(len(merged_docs_df)),
            "query_count": int(len(query_df)),
            "doc_failure_count": int(doc_failures_count),
            "query_failure_count": int(query_failures_count),
            "failure_count": int(doc_failures_count + query_failures_count),
        },
    }
