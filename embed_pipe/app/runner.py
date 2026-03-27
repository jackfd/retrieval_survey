from pathlib import Path
import logging
import pandas as pd
from typing import Any, Dict

from embed_pipe.domain.models import BuilderConfig, DatasetContext
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

    def run(self) -> None:
        run_start = utc_now_iso()

        runtime = self.builder_cfg.runtime

        selector = ChunkSelector(
            embedding_strategy=self.embedding_strategy,
            chunk_num=1,
            config=SelectorConfig(batch_size=runtime.batch_size),
        )
        doc_service = DocumentService(selector=selector, runtime=runtime)
        doc_value = doc_service.process(self.dataset_ctx.docs_path)

        existing_docs_df = self.output_writer.load_existing_docs()
        merged_docs_df = pd.concat(
            [existing_docs_df, doc_value.output_df], ignore_index=True
        )
        if not merged_docs_df.empty:
            merged_docs_df = merged_docs_df.drop_duplicates(
                subset=["doc_id"], keep="last"
            )

        query_service = QueryService(self.embedding_strategy, runtime)
        query_value = query_service.process(self.dataset_ctx.queries_path)

        metadata = build_run_metadata(
            run_start=run_start,
            dataset_ctx=self.dataset_ctx,
            builder_cfg=self.builder_cfg,
            merged_docs_df=merged_docs_df,
            query_df=query_value.output_df,
        )

        self.output_writer.write_all_outputs(
            docs_df=merged_docs_df,
            queries_df=query_value.output_df,
            metadata=metadata,
        )

        self.logger.info(
            "end: doc_count=%s query_count=%s",
            len(merged_docs_df),
            len(query_value.output_df),
        )
        return None


def build_run_metadata(
    run_start: str,
    dataset_ctx: DatasetContext,
    builder_cfg: BuilderConfig,
    merged_docs_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> Dict[str, Any]:
    model = builder_cfg.model
    run_end = utc_now_iso()
    return {
        "run": {
            "start_time_utc": run_start,
            "end_time_utc": run_end,
            "resolved_dataset_dir": str(dataset_ctx.resolved_dataset_dir),
        },
        "model": {
            "model_name": model.model_name,
            "provider": model.provider,
            "model_id": model.model_id,
        },
        "stats": {
            "doc_count": int(len(merged_docs_df)),
            "query_count": int(len(query_df)),
        },
    }
