from typing import Any, Dict

import pandas as pd

from index_builder.config import BuilderConfig
from index_builder.dataset import DatasetContext


def build_run_metadata(
    *,
    run_start: str,
    run_end: str,
    retry_mode: bool,
    dataset_ctx: DatasetContext,
    builder_cfg: BuilderConfig,
    merged_docs_df: pd.DataFrame,
    query_df: pd.DataFrame,
    attempted_docs: int,
    failures_count: int,
) -> Dict[str, Any]:
    runtime = builder_cfg.runtime
    model = builder_cfg.model
    dataset_meta = dataset_ctx.dataset_meta
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
            "doc_count": int(len(merged_docs_df)),
            "query_count": int(len(query_df)),
            "failure_count": int(failures_count),
        },
    }
