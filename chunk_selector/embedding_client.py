import logging
from typing import List

import numpy as np
import requests

from .selector_config import SelectorConfig


class EmbeddingClient:
    def __init__(self, embedding_api_url: str, config: SelectorConfig):
        self.embedding_api_url = embedding_api_url
        self.config = config

    def get_embeddings(self, chunks: List[str]) -> np.ndarray:
        logger = logging.getLogger("index_builder")
        vectors = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i : i + self.config.batch_size]
            batch_index = i // self.config.batch_size + 1

            last_error = None
            for retry in range(self.config.max_retries + 1):
                try:
                    resp = requests.post(
                        self.embedding_api_url,
                        json={"chunks": batch},
                        timeout=self.config.request_timeout,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    if "vectors" not in payload or not isinstance(payload["vectors"], list):
                        raise ValueError("embedding response missing 'vectors' list")

                    batch_vectors = np.asarray(payload["vectors"], dtype=float)
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
                    "Failed to get embeddings for batch",
                    "batch_index=%s batch_size=%s url=%s last_error=%s"
                    % (batch_index, len(batch), self.embedding_api_url, last_error),
                )
                raise requests.exceptions.RequestException(
                    f"Failed to get embeddings for batch {batch_index}: {last_error}"
                )

        if not vectors:
            logger.error(
                "event=embedding_http_degrade reason=%s context=%s",
                "Embedding provider returned empty vectors",
                "input_chunk_count=%s url=%s" % (len(chunks), self.embedding_api_url),
            )
            return np.empty((0, 0), dtype=float)
        return np.vstack(vectors)
