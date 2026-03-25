import logging

import numpy as np
import requests

from embed_pipe.domain.models import RuntimeConfig
from embed_pipe.infra.embedding_strategies.base import BaseEmbeddingStrategy
from embed_pipe.infra.embedding_strategies.shape import ensure_embedding_shape


class HttpEmbeddingStrategy(BaseEmbeddingStrategy):
    def __init__(self, runtime: RuntimeConfig):
        super().__init__(runtime)
        self.embedding_api_url = runtime.embedding_api_url

    def encode(self, texts, is_query):
        logger = logging.getLogger("embed_pipe")
        prepared = self._prepare_texts(texts, is_query=is_query)
        vectors = []
        for start in range(0, len(prepared), self.runtime.batch_size):
            batch = prepared[start : start + self.runtime.batch_size]
            batch_index = start // self.runtime.batch_size + 1
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
                            "embedding vector count mismatch: expected %s, got %s"
                            % (len(batch), batch_vectors.shape[0])
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
                raise RuntimeError("HTTP embedding failed for batch %s: %s" % (batch_index, last_error))

        if not vectors:
            output = np.empty((0, self.runtime.embedding_dim), dtype=np.float32)
            return ensure_embedding_shape(output, self.runtime.embedding_dim)
        output = np.vstack(vectors)

        if self.runtime.normalize_embeddings:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            output = output / norms
        return ensure_embedding_shape(output, self.runtime.embedding_dim)
