from pathlib import Path
import logging
import pandas as pd
from typing import Any, Dict

from embedding.domain.models import BuilderConfig, DatasetContext
from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.infra.logger import utc_now_iso
from embedding.infra.output_writer import OutputWriter
from embedding.services.chunking import ChunkSelector, SelectorConfig
from embedding.services.document_service import DocumentService
from embedding.services.query_service import QueryService

logger = logging.getLogger("embedding")


class BuilderRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        dataset_ctx: DatasetContext,
        builder_cfg: BuilderConfig,
        embedding_strategy: EmbeddingStrategy,
    ):
        self.output_writer = OutputWriter(
            output_dir, embedding_dim=builder_cfg.experiment.embedding_dim
        )
        self.dataset_ctx = dataset_ctx
        self.builder_cfg = builder_cfg
        self.embedding_strategy = embedding_strategy

    def run(self) -> None:
        run_start = utc_now_iso()

        inference = self.builder_cfg.inference

        selector = ChunkSelector(
            embedding_strategy=self.embedding_strategy,
            chunk_num=1,
            config=SelectorConfig(batch_size=inference.batch_size),
        )
        doc_service = DocumentService(selector=selector)
        doc_value = doc_service.process(self.dataset_ctx.docs_path)

        existing_docs_df = self.output_writer.load_existing_docs()
        merged_docs_df = pd.concat(
            [existing_docs_df, doc_value.output_df], ignore_index=True
        )
        if not merged_docs_df.empty:
            merged_docs_df = merged_docs_df.drop_duplicates(
                subset=["doc_id"], keep="last"
            )

        query_service = QueryService(self.embedding_strategy)
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

        logger.info(
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
            "provider": model.provider,
            "model_id": model.model_id,
        },
        "stats": {
            "doc_count": int(len(merged_docs_df)),
            "query_count": int(len(query_df)),
        },
    }
