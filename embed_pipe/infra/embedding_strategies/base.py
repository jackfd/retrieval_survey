from typing import List, Protocol

import numpy as np

from embed_pipe.domain.models import RuntimeConfig


class EmbeddingStrategy(Protocol):
    def encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        ...


class BaseEmbeddingStrategy:
    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime

    def _prepare_texts(self, texts: List[str], is_query: bool) -> List[str]:
        prefix = self.runtime.query_prefix if is_query else self.runtime.doc_prefix
        prepared: List[str] = []
        for text in texts:
            base = "%s%s" % (prefix, text)
            if self.runtime.instruction_template:
                if "{text}" in self.runtime.instruction_template:
                    base = self.runtime.instruction_template.format(text=base)
                else:
                    base = self.runtime.instruction_template + base
            prepared.append(base)
        return prepared
