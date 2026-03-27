import logging

from embed_pipe.domain.exceptions import StrategyBuildError
from embed_pipe.domain.models import ExperimentConfig, InferenceConfig, ModelConfig
from embed_pipe.infra.embedding_strategies.base import EmbeddingStrategy
from embed_pipe.infra.embedding_strategies.http import HttpEmbeddingStrategy
from embed_pipe.infra.embedding_strategies.local import LocalEmbeddingStrategy

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
