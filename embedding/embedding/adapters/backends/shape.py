import numpy as np


def ensure_embedding_shape(vectors: np.ndarray, expected_dim: int) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("Expected 2D embeddings, got shape=%s" % (vectors.shape,))
    if vectors.shape[1] > expected_dim:
        raise ValueError(
            "Embedding dimension exceeds limit, expected<=%s, got=%s"
            % (expected_dim, vectors.shape[1])
        )
    return vectors
