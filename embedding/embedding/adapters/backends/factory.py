import logging

from embedding.domain.exceptions import StrategyBuildError
from embedding.domain.models import ExperimentConfig, InferenceConfig, ModelConfig
from embedding.adapters.backends.base import EmbeddingStrategy
from embedding.adapters.backends.http import HttpEmbeddingStrategy
from embedding.adapters.backends.local import LocalEmbeddingStrategy

logger = logging.getLogger(__name__)


class EmbeddingStrategyFactory:
    def build(
        self,
        experiment: ExperimentConfig,
        inference: InferenceConfig,
        model: ModelConfig,
    ) -> EmbeddingStrategy:
        try:
            if inference.embedding_api_url.strip():
                return HttpEmbeddingStrategy(experiment=experiment, inference=inference)
            return LocalEmbeddingStrategy(
                experiment=experiment, inference=inference, model=model
            )
        except Exception as exc:
            logger.exception(
                "Failed to build embedding strategy model_id=%s provider=%s",
                model.model_id,
                model.provider,
            )
            raise StrategyBuildError(
                "Failed to build embedding strategy model_id=%s provider=%s error=%s"
                % (model.model_id, model.provider, exc)
            ) from exc
