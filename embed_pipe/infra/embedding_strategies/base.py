from typing import List, Protocol

import numpy as np

from embed_pipe.domain.models import ExperimentConfig, InferenceConfig


class EmbeddingStrategy(Protocol):
    def encode(self, texts: List[str], is_query: bool) -> np.ndarray: ...


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
