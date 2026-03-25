from pathlib import Path

from chunk_selector.chunk_selector import ChunkSelector
from chunk_selector.selector_config import SelectorConfig
from embed_pipe.domain.models import BuilderConfig, DatasetContext
from embed_pipe.infra.embedding_gateway import EmbeddingStrategy
from embed_pipe.infra.logger import utc_now_iso
from embed_pipe.infra.output_writer import OutputWriter
from embed_pipe.services.document_service import DocumentService
from embed_pipe.services.metadata_service import MetadataService
from embed_pipe.services.query_service import QueryService


class BuilderRunner:
    def __init__(
        self,
        *,
        dataset_name: str,
        output_dir: Path,
        dataset_ctx: DatasetContext,
        builder_cfg: BuilderConfig,
        embedding_strategy: EmbeddingStrategy,
        logger,
    ):
        self.dataset_name = dataset_name
        self.output_writer = OutputWriter(output_dir)
        self.dataset_ctx = dataset_ctx
        self.builder_cfg = builder_cfg
        self.embedding_strategy = embedding_strategy
        self.logger = logger

    def run(self) -> None:
        runtime = self.builder_cfg.runtime
        model = self.builder_cfg.model

        run_start = utc_now_iso()
        self.logger.info(
            "run_start dataset_root=%s resolved_dataset_dir=%s dataset_name=%s model_name=%s",
            self.dataset_ctx.dataset_root,
            self.dataset_ctx.resolved_dataset_dir,
            self.dataset_name,
            model.model_name,
        )

        previous_failed_ids = self.output_writer.load_failed_doc_ids()
        retry_mode = len(previous_failed_ids) > 0
        retry_id_set = set(previous_failed_ids)

        selector = ChunkSelector(
            embedding_strategy=self.embedding_strategy,
            chunk_num=1,
            config=SelectorConfig(batch_size=runtime.batch_size),
        )
        document_service = DocumentService(selector=selector, runtime=runtime)
        query_service = QueryService(embedding_strategy=self.embedding_strategy, runtime=runtime)

        doc_result = document_service.process(
            docs_path=self.dataset_ctx.docs_path,
            retry_mode=retry_mode,
            retry_id_set=retry_id_set,
        )

        existing_docs_df = self.output_writer.load_existing_docs()
        merged_docs_df = self.output_writer.merge_docs_with_retry(
            new_docs_df=doc_result.output_df,
            existing_docs_df=existing_docs_df,
            retry_mode=retry_mode,
            retry_id_set=retry_id_set,
        )
        self.output_writer.write_docs(merged_docs_df)

        query_result = query_service.process(self.dataset_ctx.queries_path)
        self.output_writer.write_queries(query_result.output_df)

        all_failures = doc_result.failures + query_result.failures
        self.output_writer.write_failures(all_failures)

        metadata_service = MetadataService()
        run_end = utc_now_iso()
        metadata = metadata_service.build_run_metadata(
            run_start=run_start,
            run_end=run_end,
            retry_mode=retry_mode,
            dataset_ctx=self.dataset_ctx,
            builder_cfg=self.builder_cfg,
            merged_docs_df=merged_docs_df,
            query_df=query_result.output_df,
            attempted_docs=doc_result.attempted_count,
            attempted_queries=query_result.attempted_count,
            doc_failures_count=len(doc_result.failures),
            query_failures_count=len(query_result.failures),
        )
        self.output_writer.write_metadata(metadata)

        self.logger.info(
            "run_end attempted_doc_count=%s attempted_query_count=%s doc_count=%s query_count=%s "
            "doc_failure_count=%s query_failure_count=%s retry_mode=%s",
            doc_result.attempted_count,
            query_result.attempted_count,
            len(merged_docs_df),
            len(query_result.output_df),
            len(doc_result.failures),
            len(query_result.failures),
            retry_mode,
        )
