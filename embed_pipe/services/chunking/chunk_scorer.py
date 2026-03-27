import re
from typing import List

import networkx as nx
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .selector_config import SelectorConfig


class ChunkScorer:
    def __init__(self, stop_words: set[str], config: SelectorConfig):
        self.stop_words = stop_words
        self.config = config

        self.vectorizer = None
        self.tfidf_matrix = np.empty((0, 0), dtype=float)
        self.tfidf_feature_names: List[str] = []
        self.keywords: set[str] = set()
        self.text_rank_scores: List[float] = []

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", text.lower())

    def compute_global_statistics(self, chunks: List[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words=list(self.stop_words),
            tokenizer=self.tokenize,
            token_pattern=None,
            lowercase=True,
        )
        tfidf_sparse = self.vectorizer.fit_transform(chunks)
        self.tfidf_matrix = tfidf_sparse.toarray()
        self.tfidf_feature_names = list(self.vectorizer.get_feature_names_out())

        self.keywords = set()
        for row in self.tfidf_matrix:
            if row.size == 0:
                continue
            top_indices = row.argsort()[-self.config.top_keywords :][::-1]
            for idx in top_indices:
                if row[idx] > 0:
                    self.keywords.add(self.tfidf_feature_names[idx])

        graph = nx.Graph()
        for chunk in chunks:
            words = self.tokenize(chunk)
            for i, w1 in enumerate(words):
                window_words = words[i + 1 : i + 1 + self.config.cooccur_window]
                for w2 in window_words:
                    if graph.has_edge(w1, w2):
                        graph[w1][w2]["weight"] += 1
                    else:
                        graph.add_edge(w1, w2, weight=1)

        if graph.number_of_nodes() > 0:
            token_scores = nx.pagerank(graph)
        else:
            token_scores = {}

        self.text_rank_scores = []
        for chunk in chunks:
            words = self.tokenize(chunk)
            if not words:
                self.text_rank_scores.append(0.0)
                continue
            score = float(np.mean([token_scores.get(word, 0.0) for word in words]))
            self.text_rank_scores.append(score)

    def compute_scores(
        self, chunks: List[str], candidate_idxs: List[int], title: str = ""
    ) -> np.ndarray:
        tfidf_scores = np.sum(self.tfidf_matrix[candidate_idxs], axis=1)
        text_rank_scores = np.array(
            [self.text_rank_scores[i] for i in candidate_idxs], dtype=float
        )

        keyword_count = len(self.keywords)
        if keyword_count == 0:
            coverage_scores = np.zeros(len(candidate_idxs), dtype=float)
        else:
            coverage_scores = np.array(
                [
                    len(set(self.tokenize(chunks[i])) & self.keywords) / keyword_count
                    for i in candidate_idxs
                ],
                dtype=float,
            )
        title = str(title).strip()
        if title:
            title_tokens = set(self.tokenize(title))
            title_scores = np.array(
                [
                    len(set(self.tokenize(chunks[i])) & title_tokens)
                    / len(title_tokens)
                    for i in candidate_idxs
                ],
                dtype=float,
            )
            coverage_scores = 0.8 * coverage_scores + 0.2 * title_scores

        scaler = StandardScaler()
        tfidf_norm = scaler.fit_transform(tfidf_scores.reshape(-1, 1)).flatten()
        text_rank_norm = scaler.fit_transform(text_rank_scores.reshape(-1, 1)).flatten()
        coverage_norm = scaler.fit_transform(coverage_scores.reshape(-1, 1)).flatten()

        return (
            self.config.alpha * tfidf_norm
            + self.config.beta * text_rank_norm
            + self.config.gamma * coverage_norm
        )
