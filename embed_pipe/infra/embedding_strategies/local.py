import logging
from typing import Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from embed_pipe.domain.models import ExperimentConfig, InferenceConfig, ModelConfig
from embed_pipe.infra.embedding_strategies.base import BaseEmbeddingStrategy
from embed_pipe.infra.embedding_strategies.shape import ensure_embedding_shape


class LocalEmbeddingStrategy(BaseEmbeddingStrategy):
    def __init__(
        self,
        experiment: ExperimentConfig,
        inference: InferenceConfig,
        model: ModelConfig,
    ):
        super().__init__(experiment=experiment, inference=inference)
        self.model = model
        self._encoder = self._build_local_encoder(model=model)

    def _build_local_encoder(self, model: ModelConfig) -> Tuple[str, object]:
        logger = logging.getLogger("embed_pipe")
        if model.provider == "sentence_transformers":
            encoder = SentenceTransformer(model.model_id, device=self.inference.device)
            if hasattr(encoder, "max_seq_length"):
                encoder.max_seq_length = int(self.experiment.max_length)
            return ("sentence_transformers", encoder)

        if model.provider == "flag_embedding":
            encoder = BGEM3FlagModel(
                model.model_id, use_fp16=str(self.inference.device).startswith("cuda")
            )
            return ("flag_embedding", encoder)

        logger.error(
            "Unsupported local provider provider=%s model_id=%s",
            model.provider,
            model.model_id,
        )
        raise RuntimeError("Unsupported local provider=%r" % model.provider)

    def encode(self, texts, is_query):
        prepared = self._prepare_texts(texts, is_query=is_query)
        provider, encoder = self._encoder

        if provider == "sentence_transformers":
            vecs = encoder.encode(
                prepared,
                batch_size=int(self.inference.batch_size),
                normalize_embeddings=bool(self.experiment.normalize_embeddings),
                convert_to_numpy=True,
            )
            output = np.asarray(vecs, dtype=np.float32)
            return ensure_embedding_shape(output, self.experiment.embedding_dim)

        try:
            payload = encoder.encode(
                prepared,
                batch_size=int(self.inference.batch_size),
                max_length=int(self.experiment.max_length),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            payload = encoder.encode(
                prepared,
                batch_size=int(self.inference.batch_size),
                max_length=int(self.experiment.max_length),
            )

        dense = payload.get("dense_vecs") if isinstance(payload, dict) else payload
        output = np.asarray(dense, dtype=np.float32)
        if self.experiment.normalize_embeddings:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            output = output / norms
        return ensure_embedding_shape(output, self.experiment.embedding_dim)
