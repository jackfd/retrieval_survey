import json
from pathlib import Path

from index_builder.config import BuilderConfig
from index_builder.dataset import DatasetContext
from index_builder.embedding import EmbeddingStrategy
from index_builder.io_utils import (
    load_existing_docs_parquet,
    load_failures,
    merge_docs_with_retry,
    setup_logger,
    utc_now_iso,
    write_failures,
)
from index_builder.metadata import build_run_metadata
from index_builder.processors import build_chunk_selector, process_docs, process_queries


def run_builder(
    *,
    dataset_name: str,
    output_root: Path,
    dataset_ctx: DatasetContext,
    builder_cfg: BuilderConfig,
    embedding_strategy: EmbeddingStrategy,
) -> None:
    runtime = builder_cfg.runtime
    model = builder_cfg.model
    output_dir = output_root / dataset_name / model.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "app.log")

    run_start = utc_now_iso()
    logger.info(
        "run_start dataset_root=%s resolved_dataset_dir=%s dataset_name=%s model_name=%s embedding_mode=%s",
        dataset_ctx.dataset_root,
        dataset_ctx.resolved_dataset_dir,
        dataset_name,
        model.model_name,
        runtime.embedding_mode,
    )

    docs_parquet_path = output_dir / "docs.parquet"
    queries_parquet_path = output_dir / "queries.parquet"
    failures_path = output_dir / "failures.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    previous_failed_ids = load_failures(failures_path)
    retry_mode = len(previous_failed_ids) > 0
    retry_id_set = set(previous_failed_ids)

    selector = build_chunk_selector(runtime=runtime, embedding_strategy=embedding_strategy)
    doc_result = process_docs(
        docs_path=dataset_ctx.docs_path,
        selector=selector,
        runtime=runtime,
        retry_mode=retry_mode,
        retry_id_set=retry_id_set,
        logger=logger,
    )

    existing_docs_df = load_existing_docs_parquet(docs_parquet_path)
    merged_docs_df = merge_docs_with_retry(
        new_docs_df=doc_result.docs_df,
        existing_docs_df=existing_docs_df,
        retry_mode=retry_mode,
        retry_id_set=retry_id_set,
    )
    merged_docs_df.to_parquet(docs_parquet_path, index=False)

    query_df = process_queries(
        queries_path=dataset_ctx.queries_path,
        embedding_strategy=embedding_strategy,
        runtime=runtime,
    )
    query_df.to_parquet(queries_parquet_path, index=False)

    write_failures(failures_path, doc_result.failures)

    run_end = utc_now_iso()
    metadata = build_run_metadata(
        run_start=run_start,
        run_end=run_end,
        retry_mode=retry_mode,
        dataset_ctx=dataset_ctx,
        builder_cfg=builder_cfg,
        merged_docs_df=merged_docs_df,
        query_df=query_df,
        attempted_docs=doc_result.attempted_docs,
        failures_count=len(doc_result.failures),
    )
    with metadata_path.open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)

    logger.info(
        "run_end attempted_doc_count=%s doc_count=%s query_count=%s failure_count=%s retry_mode=%s",
        doc_result.attempted_docs,
        len(merged_docs_df),
        len(query_df),
        len(doc_result.failures),
        retry_mode,
    )

