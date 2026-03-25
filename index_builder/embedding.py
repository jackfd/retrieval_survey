import logging
from typing import List, Protocol

import numpy as np
import requests

from index_builder.config import ModelConfig, RuntimeConfig
from index_builder.errors import EmbeddingGenerationError, ModelLoadError


class EmbeddingStrategy(Protocol):
    def encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        ...


def ensure_embedding_shape(vectors: np.ndarray, expected_dim: int) -> np.ndarray:
    logger = logging.getLogger("index_builder")
    if vectors.ndim != 2:
        logger.error(
            "event=embedding_shape_invalid reason=%s context=%s",
            "Expected 2D embeddings",
            "expected_dim=%s actual_shape=%s" % (expected_dim, vectors.shape),
        )
        raise EmbeddingGenerationError(f"Expected 2D embeddings, got shape={vectors.shape}")
    if vectors.shape[1] != expected_dim:
        logger.error(
            "event=embedding_shape_invalid reason=%s context=%s",
            "Embedding dimension mismatch",
            "expected_dim=%s actual_dim=%s actual_shape=%s"
            % (expected_dim, vectors.shape[1], vectors.shape),
        )
        raise EmbeddingGenerationError(
            f"Embedding dimension mismatch, expected={expected_dim}, got={vectors.shape[1]}"
        )
    return vectors


class BaseEmbeddingStrategy:
    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime

    def _prepare_texts(self, texts: List[str], is_query: bool) -> List[str]:
        prefix = self.runtime.query_prefix if is_query else self.runtime.doc_prefix
        prepared: List[str] = []
        for text in texts:
            base = f"{prefix}{text}"
            if self.runtime.instruction_template:
                if "{text}" in self.runtime.instruction_template:
                    base = self.runtime.instruction_template.format(text=base)
                else:
                    base = self.runtime.instruction_template + base
            prepared.append(base)
        return prepared


class LocalEmbeddingStrategy(BaseEmbeddingStrategy):
    def __init__(self, runtime: RuntimeConfig, model: ModelConfig):
        super().__init__(runtime)
        self.model = model
        self._encoder = self._build_local_encoder(model=model)

    def _build_local_encoder(self, model: ModelConfig):
        logger = logging.getLogger("index_builder")
        if model.provider == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                logger.error(
                    "event=model_load_failed reason=%s context=%s",
                    exc,
                    "provider=%s model_id=%s" % (model.provider, model.model_id),
                )
                raise ModelLoadError(
                    "sentence-transformers is required for provider=sentence_transformers"
                ) from exc
            encoder = SentenceTransformer(model.model_id, device=self.runtime.device)
            if hasattr(encoder, "max_seq_length"):
                encoder.max_seq_length = int(self.runtime.max_length)
            return ("sentence_transformers", encoder)

        if model.provider == "flag_embedding":
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                logger.error(
                    "event=model_load_failed reason=%s context=%s",
                    exc,
                    "provider=%s model_id=%s" % (model.provider, model.model_id),
                )
                raise ModelLoadError("FlagEmbedding is required for provider=flag_embedding") from exc
            encoder = BGEM3FlagModel(
                model.model_id, use_fp16=str(self.runtime.device).startswith("cuda")
            )
            return ("flag_embedding", encoder)

        logger.error(
            "event=model_load_failed reason=%s context=%s",
            "Unsupported local provider",
            "provider=%s model_id=%s" % (model.provider, model.model_id),
        )
        raise ModelLoadError(f"Unsupported local provider={model.provider!r}")

    def encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        prepared = self._prepare_texts(texts, is_query=is_query)
        provider, encoder = self._encoder

        if provider == "sentence_transformers":
            vecs = encoder.encode(
                prepared,
                batch_size=int(self.runtime.batch_size),
                normalize_embeddings=bool(self.runtime.normalize_embeddings),
                convert_to_numpy=True,
            )
            output = np.asarray(vecs, dtype=np.float32)
            return ensure_embedding_shape(output, self.runtime.embedding_dim)

        try:
            payload = encoder.encode(
                prepared,
                batch_size=int(self.runtime.batch_size),
                max_length=int(self.runtime.max_length),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            payload = encoder.encode(
                prepared,
                batch_size=int(self.runtime.batch_size),
                max_length=int(self.runtime.max_length),
            )

        dense = payload.get("dense_vecs") if isinstance(payload, dict) else payload
        output = np.asarray(dense, dtype=np.float32)
        if self.runtime.normalize_embeddings:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            output = output / norms
        return ensure_embedding_shape(output, self.runtime.embedding_dim)


class HttpEmbeddingStrategy(BaseEmbeddingStrategy):
    def __init__(self, runtime: RuntimeConfig):
        super().__init__(runtime)
        self.embedding_api_url = runtime.embedding_api_url

    def encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        logger = logging.getLogger("index_builder")
        prepared = self._prepare_texts(texts, is_query=is_query)
        vectors = []
        for i in range(0, len(prepared), self.runtime.batch_size):
            batch = prepared[i : i + self.runtime.batch_size]
            batch_index = i // self.runtime.batch_size + 1

            last_error = None
            for retry in range(self.runtime.http_max_retries + 1):
                try:
                    resp = requests.post(
                        self.embedding_api_url,
                        json={"chunks": batch},
                        timeout=self.runtime.http_timeout,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    if "vectors" not in payload or not isinstance(payload["vectors"], list):
                        raise ValueError("embedding response missing 'vectors' list")

                    batch_vectors = np.asarray(payload["vectors"], dtype=np.float32)
                    if batch_vectors.ndim != 2:
                        raise ValueError("embedding vectors must be a 2D array")
                    if batch_vectors.shape[0] != len(batch):
                        raise ValueError(
                            "embedding vector count mismatch: "
                            f"expected {len(batch)}, got {batch_vectors.shape[0]}"
                        )

                    vectors.append(batch_vectors)
                    last_error = None
                    break
                except (requests.exceptions.RequestException, ValueError) as exc:
                    last_error = exc
                    logger.error(
                        "event=embedding_http_retry_failed reason=%s context=%s",
                        exc,
                        "batch_index=%s retry=%s batch_size=%s url=%s"
                        % (batch_index, retry, len(batch), self.embedding_api_url),
                    )

            if last_error is not None:
                logger.error(
                    "event=embedding_http_failed reason=%s context=%s",
                    "HTTP embedding failed",
                    "batch_index=%s batch_size=%s url=%s last_error=%s"
                    % (batch_index, len(batch), self.embedding_api_url, last_error),
                )
                raise EmbeddingGenerationError(
                    f"HTTP embedding failed for batch {batch_index}: {last_error}"
                ) from last_error

        if not vectors:
            output = np.empty((0, self.runtime.embedding_dim), dtype=np.float32)
            return ensure_embedding_shape(output, self.runtime.embedding_dim)
        output = np.vstack(vectors)

        if self.runtime.normalize_embeddings:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            output = output / norms
        return ensure_embedding_shape(output, self.runtime.embedding_dim)


def build_embedding_strategy(runtime: RuntimeConfig, model: ModelConfig) -> EmbeddingStrategy:
    if runtime.embedding_api_url.strip():
        return HttpEmbeddingStrategy(runtime=runtime)
    return LocalEmbeddingStrategy(runtime=runtime, model=model)
