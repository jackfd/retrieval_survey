# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Sequence
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 3
DEFAULT_MMR_LAMBDA = 0.7


def select_from_embeddings(
    doc_id: str,
    candidates: Sequence[Dict[str, object]],
    embeddings,
    *,
    top_n: int = DEFAULT_TOP_N,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> List[Dict]:
    if len(candidates) != len(embeddings):
        logger.error(
            "Chunk embeddings count mismatch doc_id=%s candidate_count=%s embedding_count=%s",
            doc_id,
            len(candidates),
            len(embeddings),
        )
        raise ValueError("Chunk embeddings count must match candidate count")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        logger.error("Chunk embeddings must be a 2D array, got %s", embeddings.shape)
        raise ValueError("Chunk embeddings must be a 2D array")
    embeddings = _normalize_rows(embeddings)

    centroid = _compute_centroid(embeddings)
    rep_scores = _compute_rep_scores(embeddings, centroid)
    indices, scores = _select_top_n(
        embeddings,
        rep_scores,
        top_n=top_n,
        mmr_lambda=mmr_lambda,
    )
    return _build_records(
        doc_id=doc_id,
        candidates=candidates,
        embeddings=embeddings,
        selected_indices=indices,
        selected_scores=scores,
        top_n=top_n,
    )


def _compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    centroid = np.mean(embeddings, axis=0, dtype=np.float32)
    return _normalize_vector(centroid)


def _compute_rep_scores(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return embeddings @ centroid


def _select_top_n(
    embeddings: np.ndarray,
    rep_scores: np.ndarray,
    *,
    top_n: int,
    mmr_lambda: float,
) -> tuple[List[int], List[float]]:
    candidate_count = embeddings.shape[0]
    select_count = min(candidate_count, top_n)
    selected_indices: List[int] = []
    selected_scores: List[float] = []

    while len(selected_indices) < select_count:
        best_index = -1
        best_score = float("-inf")

        for index in range(candidate_count):
            if index in selected_indices:
                continue

            if not selected_indices:
                score = float(rep_scores[index])
            else:
                max_sim_to_selected = max(
                    float(embeddings[index] @ embeddings[selected_index])
                    for selected_index in selected_indices
                )
                score = float(
                    mmr_lambda * rep_scores[index]
                    - (1.0 - mmr_lambda) * max_sim_to_selected
                )

            if _is_better_candidate(
                score=score,
                index=index,
                best_score=best_score,
                best_index=best_index,
            ):
                best_score = score
                best_index = index

        selected_indices.append(best_index)
        selected_scores.append(best_score)

    return selected_indices, selected_scores


def _is_better_candidate(
    score: float, index: int, best_score: float, best_index: int
) -> bool:
    if best_index < 0:
        return True
    if score > best_score and not np.isclose(score, best_score):
        return True
    if np.isclose(score, best_score):
        return index < best_index
    return False


def _build_records(
    *,
    doc_id: str,
    candidates: Sequence[Dict[str, object]],
    embeddings: np.ndarray,
    selected_indices: Sequence[int],
    selected_scores: Sequence[float],
    top_n: int,
) -> List[Dict]:
    total_chunks = len(candidates)
    width = max(3, len(str(total_chunks)))
    records: List[Dict] = []
    for rank, (candidate_index, score) in enumerate(
        zip(selected_indices, selected_scores), start=1
    ):
        candidate = candidates[candidate_index]
        order = int(candidate["order"])
        records.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}#c{order:0{width}d}",
                "chunk_text": str(candidate["text"]),
                "chunk_vector": embeddings[candidate_index].tolist(),
                "chunk_score": float(score),
                "chunk_rank": rank,
            }
        )
    if total_chunks > top_n * 3:
        logger.info("doc id:%s has a large chunks size:%s", doc_id, total_chunks)
    return records


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
