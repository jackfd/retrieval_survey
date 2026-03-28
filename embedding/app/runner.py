from pathlib import Path
import logging

from embedding.domain.models import BuilderConfig, DatasetContext
from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.infra.logger import utc_now_iso
from embedding.infra.output_writer import OutputWriter
from embedding.services.chunk_selector import ChunkSelector
from embedding.services.document_service import DocumentService
from embedding.services.query_service import QueryService

logger = logging.getLogger(__name__)


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
        logger.info(
            "start: model_id=%s resolved_dataset_dir=%s start_time_utc=%s",
            self.builder_cfg.model.model_id,
            self.dataset_ctx.resolved_dataset_dir,
            run_start,
        )

        selector = ChunkSelector(self.embedding_strategy)
        doc_service = DocumentService(selector)
        docs_df = doc_service.process(self.dataset_ctx.docs_path).output_df

        query_service = QueryService(self.embedding_strategy)
        queries_df = query_service.process(self.dataset_ctx.queries_path).output_df

        self.output_writer.write_all_outputs(
            docs_df=docs_df,
            queries_df=queries_df,
        )

        run_end = utc_now_iso()
        logger.info(
            "end: start_time_utc=%s end_time_utc=%s doc_count=%s chunk_count=%s query_count=%s",
            run_start,
            run_end,
            docs_df["doc_id"].nunique() if not docs_df.empty else 0,
            len(docs_df),
            len(queries_df),
        )
        return None
