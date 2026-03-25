# -*- coding: utf-8 -*-
import logging
from typing import Dict, List

import numpy as np

from embed_pipe.infra.embedding_strategies import BaseEmbeddingStrategy, ensure_embedding_shape
from .chunk_clusterer import ChunkClusterer
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
        text_processor: ChunkSplitter | None = None,
    ):
        self.embedding_strategy = embedding_strategy
        self.chunk_num = max(1, int(chunk_num))
        self.config = config or SelectorConfig()

        cluster_num = max(1, int(self.chunk_num * self.config.cluster_ratio))

        self.text_processor = text_processor or ChunkSplitter(min_sentences=3, max_tokens=8092)
        self.stop_words = StopwordsLoader.load_stopwords()

        self.clusterer = ChunkClusterer(cluster_num=cluster_num)
        self.scorer = ChunkScorer(stop_words=self.stop_words, config=self.config)

    def split_paragraphs(self, text: str) -> List[str]:
        return self.text_processor.split_paragraphs(text)

    def compute_global_statistics(self, chunks: List[str]) -> None:
        self.scorer.compute_global_statistics(chunks)

    def get_embeddings(self, chunks: List[str]) -> np.ndarray:
        vectors = self.embedding_strategy.encode(chunks, is_query=False)
        output = np.asarray(vectors, dtype=np.float32)
        expected_dim = int(self.embedding_strategy.runtime.embedding_dim)
        return ensure_embedding_shape(output, expected_dim)

    def cluster_chunks(self, embeddings: np.ndarray) -> List[int]:
        return self.clusterer.cluster_chunks(embeddings)

    def compute_scores(self, chunks: List[str], candidate_idxs: List[int], title: str = "") -> np.ndarray:
        return self.scorer.compute_scores(chunks, candidate_idxs, title=title)

    def select_chunks(self, text: str, title: str) -> List[Dict]:
        logger = logging.getLogger("embed_pipe")
        chunks = self.text_processor.split_paragraphs(text)
        if not chunks:
            logger.error(
                "event=chunk_select_degrade reason=%s context=%s",
                "No valid paragraphs after text splitting",
                "text_length=%s" % (len(text) if isinstance(text, str) else 0),
            )
            return []

        try:
            self.compute_global_statistics(chunks)
        except ValueError as exc:
            logger.error(
                "event=chunk_statistics_failed reason=%s context=%s",
                exc,
                "chunk_count=%s" % len(chunks),
            )
            logger.error(
                "event=chunk_select_degrade reason=%s context=%s",
                "Global statistics computation failed",
                "chunk_count=%s" % len(chunks),
            )
            return []

        try:
            embeddings = self.get_embeddings(chunks)
        except Exception as exc:
            logger.error(
                "event=embedding_retrieval_failed reason=%s context=%s",
                exc,
                "chunk_count=%s" % len(chunks),
            )
            logger.error(
                "event=chunk_select_degrade reason=%s context=%s",
                "Embedding retrieval failed",
                "chunk_count=%s" % len(chunks),
            )
            return []

        if embeddings.size == 0:
            logger.error(
                "event=chunk_select_degrade reason=%s context=%s",
                "Embeddings are empty",
                "chunk_count=%s" % len(chunks),
            )
            return []

        candidate_idxs = self.cluster_chunks(embeddings)
        if not candidate_idxs:
            logger.error(
                "event=chunk_select_degrade reason=%s context=%s",
                "No valid candidate indices after clustering",
                "embedding_shape=%s" % (embeddings.shape,),
            )
            return []

        scores = self.compute_scores(chunks, candidate_idxs, title=title)
        if len(scores) == 0:
            logger.error(
                "event=chunk_select_degrade reason=%s context=%s",
                "Computed scores are empty",
                "candidate_count=%s" % len(candidate_idxs),
            )
            return []

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
