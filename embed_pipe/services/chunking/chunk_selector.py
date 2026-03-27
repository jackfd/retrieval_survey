# -*- coding: utf-8 -*-
from typing import Dict, List
import logging
import numpy as np
from sklearn.cluster import KMeans
from embed_pipe.infra.embedding_strategies import BaseEmbeddingStrategy
from .chunk_scorer import ChunkScorer
from .chunk_splitter import ChunkSplitter
from .selector_config import SelectorConfig
from .stopwords_loader import StopwordsLoader


class ChunkSelector:
    def __init__(
        self,
        embedding_strategy: BaseEmbeddingStrategy,
        chunk_num: int = 5,
        config: SelectorConfig | None = None,
    ):
        self.embedding_strategy = embedding_strategy
        self.chunk_num = max(1, int(chunk_num))
        self.config = config or SelectorConfig()

        cluster_num = max(1, int(self.chunk_num * self.config.cluster_ratio))
        self.text_processor = ChunkSplitter(min_sentences=3, max_tokens=8092)
        self.stop_words = StopwordsLoader.load_stopwords()

        self.cluster_num = max(1, int(cluster_num))
        self.scorer = ChunkScorer(stop_words=self.stop_words, config=self.config)
        self.logger = logging.getLogger(__name__)

    def select_chunks(self, text: str, title: str = "") -> List[Dict]:
        """
        外部保证text非空且长度合理，以及处理异常。
        """
        chunks = self.text_processor.split_paragraphs(text)
        self.scorer.compute_global_statistics(chunks)
        embeddings = self.embedding_strategy.encode(chunks, is_query=False)

        candidate_idxs = self.cluster_chunks(embeddings)
        scores = self.scorer.compute_scores(chunks, candidate_idxs, title)
        top_indices = np.arange(len(candidate_idxs))
        if len(candidate_idxs) > self.chunk_num:
            top_indices = np.argsort(scores)[-self.chunk_num :][::-1]

        result = []
        for index in top_indices:
            candidate_index = candidate_idxs[int(index)]
            result.append(
                {
                    "chunk_text": chunks[candidate_index],
                    "score": float(scores[int(index)]),
                    "embedding": embeddings[candidate_index].tolist(),
                }
            )
        return result

    def cluster_chunks(self, embeddings: np.ndarray) -> List[int]:
        cluster_num = min(self.cluster_num, embeddings.shape[0])
        kmeans = KMeans(
            n_clusters=cluster_num, random_state=42, n_init=10, max_iter=300
        )
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        candidates = []
        for cluster_idx in range(cluster_num):
            idxs = np.where(labels == cluster_idx)[0]
            if len(idxs) == 0:
                self.logger.warning(
                    "No documents found for cluster_idx=%s, skipping" % cluster_idx
                )
                continue
            dists = np.linalg.norm(embeddings[idxs] - centers[cluster_idx], axis=1)
            candidates.append(int(idxs[np.argmin(dists)]))
        return candidates
