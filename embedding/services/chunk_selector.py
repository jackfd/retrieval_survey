# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Sequence
import numpy as np

from embedding.infra.embedding_strategies import EmbeddingStrategy
from chunk_splitter import ChunkSplitter

logger = logging.getLogger(__name__)


class ChunkSelector:

    def __init__(self, embedding_strategy: EmbeddingStrategy):
        self.embedding_strategy = embedding_strategy
        self.top_n: int = 3
        self.mmr_lambda: float = 0.7
        self.splitter = ChunkSplitter()
        self.avg_char_per_token = 4

    def run(self, text: str, doc_id: str) -> List[Dict]:
        candidates = self.splitter.split_to_candidates(text)
        embeddings = self._embed_chunks(candidates)
        centroid = self._compute_centroid(embeddings)
        rep_scores = self._compute_rep_scores(embeddings, centroid)
        selected_indices, selected_scores = self._select_top_n(embeddings, rep_scores)
        return self._build_records(
            doc_id=doc_id,
            candidates=candidates,
            embeddings=embeddings,
            selected_indices=selected_indices,
            selected_scores=selected_scores,
        )

    def _embed_chunks(self, candidates: Sequence[Dict[str, object]]) -> np.ndarray:
        texts = [str(candidate["text"]) for candidate in candidates]
        embeddings = np.asarray(
            self.embedding_strategy.encode(texts, is_query=False),
            dtype=np.float32,
        )
        if embeddings.ndim != 2:
            raise ValueError("Chunk embeddings must be a 2D array")
        return self._normalize_rows(embeddings)

    def _compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        centroid = np.mean(embeddings, axis=0, dtype=np.float32)
        return self._normalize_vector(centroid)

    def _compute_rep_scores(
        self, embeddings: np.ndarray, centroid: np.ndarray
    ) -> np.ndarray:
        return embeddings @ centroid

    def _select_top_n(
        self, embeddings: np.ndarray, rep_scores: np.ndarray
    ) -> tuple[List[int], List[float]]:
        """
        从候选嵌入中选择top-n个最相关且多样性较高的项目

        使用MMR（Maximum Marginal Relevance）算法平衡代表性得分和与已选项目的差异性。
        首先选择代表性得分最高的项目，然后迭代地选择下一个项目，使其在代表性和与已选项目差异性之间达到平衡。

        Args:
            embeddings (np.ndarray): 候选项目的嵌入向量数组，形状为(n_candidates, embedding_dim)
            rep_scores (np.ndarray): 每个候选项目的代表性得分数组，形状为(n_candidates,)
        """
        candidate_count = embeddings.shape[0]
        w = self.mmr_lambda
        select_count = min(candidate_count, self.top_n)
        selected_indices: List[int] = []
        selected_scores: List[float] = []

        # 迭代选择指定数量的项目
        while len(selected_indices) < select_count:
            best_index = -1
            best_score = float("-inf")

            # 遍历所有候选项目以找到最佳项目
            for index in range(candidate_count):
                if index in selected_indices:
                    continue

                # 对于第一个选择，直接使用代表性得分；对于后续选择，使用MMR公式
                if not selected_indices:
                    score = float(rep_scores[index])
                else:
                    # 当前候选项目与已选项目之间的最大相似度
                    max_sim_to_selected = max(
                        float(embeddings[index] @ embeddings[selected_index])
                        for selected_index in selected_indices
                    )
                    # 使用MMR公式计算综合得分：λ * 代表性得分 - (1-λ) * 最大相似度
                    score = float(
                        w * rep_scores[index] - (1.0 - w) * max_sim_to_selected
                    )

                if self._is_better_candidate(
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
        self, score: float, index: int, best_score: float, best_index: int
    ) -> bool:
        if best_index < 0:
            return True
        if score > best_score and not np.isclose(score, best_score):
            return True
        if np.isclose(score, best_score):
            return index < best_index
        return False

    def _build_records(
        self,
        *,
        doc_id: str,
        candidates: Sequence[Dict[str, object]],
        embeddings: np.ndarray,
        selected_indices: Sequence[int],
        selected_scores: Sequence[float],
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
        if total_chunks > self.top_n * 3:
            logger.info("doc id:%s has a large chunks size:%s", doc_id, total_chunks)
        return records

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm
