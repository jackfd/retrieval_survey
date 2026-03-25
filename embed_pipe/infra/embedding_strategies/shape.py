import logging

import numpy as np


def ensure_embedding_shape(vectors: np.ndarray, expected_dim: int) -> np.ndarray:
    logger = logging.getLogger("embed_pipe")
    if vectors.ndim != 2:
        logger.error(
            "event=embedding_shape_invalid reason=%s context=%s",
            "Expected 2D embeddings",
            "expected_dim=%s actual_shape=%s" % (expected_dim, vectors.shape),
        )
        raise ValueError("Expected 2D embeddings, got shape=%s" % (vectors.shape,))
    if vectors.shape[1] != expected_dim:
        logger.error(
            "event=embedding_shape_invalid reason=%s context=%s",
            "Embedding dimension mismatch",
            "expected_dim=%s actual_dim=%s actual_shape=%s" % (expected_dim, vectors.shape[1], vectors.shape),
        )
        raise ValueError("Embedding dimension mismatch, expected=%s, got=%s" % (expected_dim, vectors.shape[1]))
    return vectors
