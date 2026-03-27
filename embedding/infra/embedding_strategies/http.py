import logging

import numpy as np
import requests

from embedding.domain.models import ExperimentConfig, InferenceConfig
from embedding.infra.embedding_strategies.base import BaseEmbeddingStrategy
from embedding.infra.embedding_strategies.shape import ensure_embedding_shape


class HttpEmbeddingStrategy(BaseEmbeddingStrategy):
    def __init__(self, experiment: ExperimentConfig, inference: InferenceConfig):
        super().__init__(experiment=experiment, inference=inference)
        self.embedding_api_url = inference.embedding_api_url
        self.logger = logging.getLogger(__name__)

    def encode(self, texts, is_query):
        prepared = self._prepare_texts(texts, is_query=is_query)
        vectors = []
        for start in range(0, len(prepared), self.inference.batch_size):
            batch = prepared[start : start + self.inference.batch_size]
            batch_index = start // self.inference.batch_size + 1
            last_error = None
            for retry in range(self.inference.http_max_retries + 1):
                try:
                    resp = requests.post(
                        self.embedding_api_url,
                        json={"chunks": batch},
                        timeout=self.inference.http_timeout,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    if "vectors" not in payload or not isinstance(
                        payload["vectors"], list
                    ):
                        raise ValueError("embedding response missing 'vectors' list")

                    batch_vectors = np.asarray(payload["vectors"], dtype=np.float32)
                    if batch_vectors.ndim != 2:
                        raise ValueError("embedding vectors must be a 2D array")
                    if batch_vectors.shape[0] != len(batch):
                        raise ValueError(
                            f"embedding vector count mismatch: expected {len(batch)}, got {batch_vectors.shape[0]}"
                        )

                    vectors.append(batch_vectors)
                    last_error = None
                    break
                except (requests.exceptions.RequestException, ValueError) as exc:
                    last_error = exc
                    self.logger.warning(
                        "embedding_retry_failed batch_index=%s retry=%s batch_size=%s url=%s error=%s",
                        batch_index,
                        retry,
                        len(batch),
                        self.embedding_api_url,
                        exc,
                    )

            if last_error is not None:
                self.logger.error(
                    "HTTP embedding failed, batch_index=%s  last_error=%s",
                    batch_index,
                    last_error,
                )
                raise RuntimeError(
                    "HTTP embedding failed for batch %s: %s" % (batch_index, last_error)
                )

        if not vectors:
            output = np.empty((0, self.experiment.embedding_dim), dtype=np.float32)
            return ensure_embedding_shape(output, self.experiment.embedding_dim)
        output = np.vstack(vectors)

        if self.experiment.normalize_embeddings:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            output = output / norms
        return ensure_embedding_shape(output, self.experiment.embedding_dim)
