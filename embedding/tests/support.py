"""Shared helpers for embedding tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from embedding.domain.models import (
    BuilderConfig,
    DatasetContext,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
)


def make_experiment_config(
    *,
    embedding_dim: int = 384,
    normalize_embeddings: bool = False,
    max_length: int = 512,
    query_prefix: str = "",
    doc_prefix: str = "",
    instruction_template: str = "",
) -> ExperimentConfig:
    return ExperimentConfig(
        embedding_dim=embedding_dim,
        normalize_embeddings=normalize_embeddings,
        max_length=max_length,
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
        instruction_template=instruction_template,
    )


def make_inference_config(
    *,
    batch_size: int = 16,
    device: str = "cpu",
    embedding_api_url: str = "",
    http_timeout: float = 10.0,
    http_max_retries: int = 3,
) -> InferenceConfig:
    return InferenceConfig(
        batch_size=batch_size,
        device=device,
        embedding_api_url=embedding_api_url,
        http_timeout=http_timeout,
        http_max_retries=http_max_retries,
    )


def make_model_config(
    *, provider: str = "test", model_id: str = "test-model"
) -> ModelConfig:
    return ModelConfig(provider=provider, model_id=model_id)


def make_builder_config(
    *,
    experiment: ExperimentConfig | None = None,
    inference: InferenceConfig | None = None,
    model: ModelConfig | None = None,
    raw_config: dict[str, Any] | None = None,
) -> BuilderConfig:
    return BuilderConfig(
        experiment=experiment or make_experiment_config(),
        inference=inference or make_inference_config(),
        model=model or make_model_config(),
        raw_config=raw_config or {},
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def make_dataset_context(
    *,
    root: Path,
    docs_rows: Sequence[dict[str, Any]],
    queries_rows: Sequence[dict[str, Any]],
    dataset_meta: dict[str, Any] | None = None,
) -> DatasetContext:
    docs_path = write_jsonl(root / "docs.jsonl", docs_rows)
    queries_path = write_jsonl(root / "train" / "queries.jsonl", queries_rows)
    return DatasetContext(
        dataset_root=root,
        resolved_dataset_dir=root,
        dataset_meta=dataset_meta or {"name": "demo"},
        docs_path=docs_path,
        queries_path=queries_path,
    )


class ScriptedEmbeddingStrategy:
    """Deterministic embedding stub used by integration-style tests."""

    def __init__(
        self,
        *,
        query_vector: Sequence[float] | None = None,
        doc_vectors: Sequence[Sequence[float]] | None = None,
        default_dim: int = 4,
    ) -> None:
        self.query_vector = list(query_vector) if query_vector is not None else None
        self.doc_vectors = (
            [list(vector) for vector in doc_vectors] if doc_vectors is not None else None
        )
        self.default_dim = default_dim
        self.calls: list[tuple[list[str], bool]] = []

    def encode(self, texts: Iterable[str], is_query: bool) -> np.ndarray:
        prepared = list(texts)
        self.calls.append((prepared, is_query))

        if is_query:
            vector = self.query_vector or [0.1, 0.2, 0.3, 0.4]
            return np.asarray([vector for _ in prepared], dtype=np.float32)

        rows: list[list[float]] = []
        for index, text in enumerate(prepared):
            if self.doc_vectors is not None and index < len(self.doc_vectors):
                rows.append(list(self.doc_vectors[index]))
            else:
                rows.append([float(len(text)), float(index), 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)
