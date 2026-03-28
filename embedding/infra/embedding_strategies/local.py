import logging
from typing import Tuple

from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from embedding.domain.models import ExperimentConfig, InferenceConfig, ModelConfig
from embedding.infra.embedding_strategies.base import BaseEmbeddingStrategy
from embedding.infra.embedding_strategies.shape import ensure_embedding_shape

logger = logging.getLogger(__name__)


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
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            output = self._normalize_rows(vecs)
            return ensure_embedding_shape(output, self.experiment.embedding_dim)

        if provider == "flag_embedding":
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
            output = self._normalize_rows(dense)
            return ensure_embedding_shape(output, self.experiment.embedding_dim)

        logger.error(
            "Unsupported local provider provider=%s model_id=%s",
            self.model.provider,
            self.model.model_id,
        )
        raise RuntimeError("Unsupported local provider=%r" % self.model.provider)
