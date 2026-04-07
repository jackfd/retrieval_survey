from typing import List, Protocol
import numpy as np
from embedding.domain.models import ExperimentConfig, InferenceConfig


class EmbeddingStrategy(Protocol):
    def encode(self, texts: List[str], is_query: bool) -> np.ndarray: ...
    def describe_input_lengths(self, texts: List[str], is_query: bool) -> dict: ...


class BaseEmbeddingStrategy:
    def __init__(self, experiment: ExperimentConfig, inference: InferenceConfig):
        self.experiment = experiment
        self.inference = inference

    def _prepare_texts(self, texts: List[str], is_query: bool) -> List[str]:
        prefix = (
            self.experiment.query_prefix if is_query else self.experiment.doc_prefix
        )
        prepared: List[str] = []
        for text in texts:
            base = "%s%s" % (prefix, text)
            if self.experiment.instruction_template:
                if "{text}" in self.experiment.instruction_template:
                    base = self.experiment.instruction_template.format(text=base)
                else:
                    base = self.experiment.instruction_template + base
            prepared.append(base)
        return prepared

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        output = np.asarray(vectors, dtype=np.float32)
        if output.ndim != 2:
            raise ValueError("Embedding vectors must be a 2D array")
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return output / norms

    def describe_input_lengths(self, texts: List[str], is_query: bool) -> dict:
        prepared = self._prepare_texts(texts, is_query=is_query)
        if not prepared:
            return {
                "token_stats_available": False,
                "max_prepared_tokens": None,
                "avg_prepared_tokens": None,
                "prepared_over_limit_count": None,
            }

        return {
            "token_stats_available": False,
            "max_prepared_tokens": None,
            "avg_prepared_tokens": None,
            "prepared_over_limit_count": None,
        }
