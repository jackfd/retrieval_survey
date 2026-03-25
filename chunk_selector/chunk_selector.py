# -*- coding: utf-8 -*-
import logging
from typing import Callable, Dict, List

import numpy as np
import requests

from chunk_clusterer import ChunkClusterer
from chunk_scorer import ChunkScorer
from chunk_splitter import ChunkSplitter
from embedding_client import EmbeddingClient
from selector_config import SelectorConfig
from stopwords_loader import StopwordsLoader


class ChunkSelector:
    """
    文本块选择器：编排 Splitter / EmbeddingClient / Scorer / Clusterer。
    对外保持原有方法兼容。
    """

    def __init__(
        self,
        embedding_api_url: str,
        chunk_num: int = 5,
        config: SelectorConfig | None = None,
        text_processor: ChunkSplitter | None = None,
        embedding_provider: Callable[[List[str]], np.ndarray] | None = None,
    ):
        self.embedding_api_url = embedding_api_url
        self.chunk_num = max(1, int(chunk_num))
        self.config = config or SelectorConfig()

        cluster_num = max(1, int(self.chunk_num * self.config.cluster_ratio))

        self.text_processor = text_processor or ChunkSplitter(
            min_sentences=3, max_tokens=8092
        )
        self.stop_words = StopwordsLoader.load_stopwords()

        self.embedding_provider = embedding_provider
        self.embedding_client = None
        if self.embedding_provider is None:
            self.embedding_client = EmbeddingClient(
                embedding_api_url=self.embedding_api_url, config=self.config
            )
        self.clusterer = ChunkClusterer(cluster_num=cluster_num)
        self.scorer = ChunkScorer(stop_words=self.stop_words, config=self.config)

    def split_paragraphs(self, text: str) -> List[str]:
        return self.text_processor.split_paragraphs(text)

    def compute_global_statistics(self, chunks: List[str]) -> None:
        self.scorer.compute_global_statistics(chunks)

    def get_embeddings(self, chunks: List[str]) -> np.ndarray:
        if self.embedding_provider is not None:
            vectors = self.embedding_provider(chunks)
            return np.asarray(vectors, dtype=float)
        if self.embedding_client is None:
            return np.empty((0, 0), dtype=float)
        return self.embedding_client.get_embeddings(chunks)

    def cluster_chunks(self, embeddings: np.ndarray) -> List[int]:
        return self.clusterer.cluster_chunks(embeddings)

    def compute_scores(
        self, chunks: List[str], candidate_idxs: List[int], title: str = ""
    ) -> np.ndarray:
        return self.scorer.compute_scores(chunks, candidate_idxs, title=title)

    def select_chunks(self, text: str, title: str) -> List[Dict]:
        chunks = self.text_processor.split_paragraphs(text)
        if not chunks:
            logging.warning(
                "No valid paragraphs after text splitting, terminating processing"
            )
            return []

        try:
            self.compute_global_statistics(chunks)
        except ValueError as exc:
            logging.warning("Failed to compute global statistics: %s", exc)
            return []

        try:
            embeddings = self.get_embeddings(chunks)
        except (requests.exceptions.RequestException, Exception) as exc:
            logging.error("Failed to retrieve embeddings: %s", exc)
            return []

        if embeddings.size == 0:
            logging.warning("Embeddings are empty, terminating processing")
            return []

        candidate_idxs = self.cluster_chunks(embeddings)
        if not candidate_idxs:
            logging.warning(
                "No valid candidate indices after clustering, terminating processing"
            )
            return []

        scores = self.compute_scores(chunks, candidate_idxs, title=title)
        if len(scores) == 0:
            logging.warning("Computed scores are empty, terminating processing")
            return []

        top_indices = np.arange(len(candidate_idxs))
        if len(candidate_idxs) > self.chunk_num:
            top_indices = np.argsort(scores)[-self.chunk_num :][::-1]

        result = []
        for i in top_indices:
            idx = candidate_idxs[int(i)]
            result.append(
                {
                    "chunk": chunks[idx],
                    "chunk_text": chunks[idx],
                    "score": float(scores[int(i)]),
                    "embedding": embeddings[idx].tolist(),
                }
            )
        return result


__all__ = ["ChunkSelector", "SelectorConfig"]
