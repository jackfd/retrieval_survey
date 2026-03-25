import re
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

from selector_config import SelectorConfig


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

    @staticmethod
    def standardize(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        if np.allclose(values.std(), 0.0):
            return np.zeros_like(values, dtype=float)
        return (values - values.mean()) / values.std()

    def _compute_tfidf_fallback(self, chunks: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        tokens_per_chunk = [self.tokenize(chunk) for chunk in chunks]
        vocab = sorted({token for tokens in tokens_per_chunk for token in tokens})
        if not vocab:
            raise ValueError("empty vocabulary")

        token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        tf = np.zeros((len(chunks), len(vocab)), dtype=float)
        df = np.zeros(len(vocab), dtype=float)

        for row, tokens in enumerate(tokens_per_chunk):
            if not tokens:
                continue
            counts: Dict[str, int] = {}
            for token in tokens:
                if token in self.stop_words:
                    continue
                counts[token] = counts.get(token, 0) + 1
            if not counts:
                continue

            for token, count in counts.items():
                idx = token_to_idx[token]
                tf[row, idx] = count / len(tokens)
            for token in counts:
                df[token_to_idx[token]] += 1

        if np.count_nonzero(tf) == 0:
            raise ValueError("empty vocabulary")

        idf = np.log((1 + len(chunks)) / (1 + df)) + 1.0
        return tf * idf, vocab

    def compute_global_statistics(self, chunks: List[str]) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                stop_words=list(self.stop_words),
                tokenizer=self.tokenize,
                token_pattern=None,
                lowercase=True,
            )
            tfidf_sparse = self.vectorizer.fit_transform(chunks)
            self.tfidf_matrix = tfidf_sparse.toarray()
            self.tfidf_feature_names = list(self.vectorizer.get_feature_names_out())
        else:
            self.vectorizer = None
            self.tfidf_matrix, self.tfidf_feature_names = self._compute_tfidf_fallback(chunks)

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
            try:
                token_scores = nx.pagerank(graph)
            except ModuleNotFoundError:
                token_scores = nx.degree_centrality(graph)
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

    def compute_scores(self, chunks: List[str], candidate_idxs: List[int], title: str = "") -> np.ndarray:
        if not candidate_idxs:
            return np.array([], dtype=float)

        tfidf_scores = np.sum(self.tfidf_matrix[candidate_idxs], axis=1)
        text_rank_scores = np.array([self.text_rank_scores[i] for i in candidate_idxs], dtype=float)

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

        title_tokens = set(self.tokenize(title)) if title else set()
        if title_tokens:
            title_scores = np.array(
                [
                    len(set(self.tokenize(chunks[i])) & title_tokens) / len(title_tokens)
                    for i in candidate_idxs
                ],
                dtype=float,
            )
            coverage_scores = 0.8 * coverage_scores + 0.2 * title_scores

        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            tfidf_norm = scaler.fit_transform(tfidf_scores.reshape(-1, 1)).flatten()
            text_rank_norm = scaler.fit_transform(text_rank_scores.reshape(-1, 1)).flatten()
            coverage_norm = scaler.fit_transform(coverage_scores.reshape(-1, 1)).flatten()
        else:
            tfidf_norm = self.standardize(tfidf_scores)
            text_rank_norm = self.standardize(text_rank_scores)
            coverage_norm = self.standardize(coverage_scores)

        return (
            self.config.alpha * tfidf_norm
            + self.config.beta * text_rank_norm
            + self.config.gamma * coverage_norm
        )
