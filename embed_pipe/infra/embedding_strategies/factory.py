import logging

from embed_pipe.domain.exceptions import StrategyBuildError
from embed_pipe.domain.models import ModelConfig, RuntimeConfig
from embed_pipe.infra.embedding_strategies.base import EmbeddingStrategy
from embed_pipe.infra.embedding_strategies.http import HttpEmbeddingStrategy
from embed_pipe.infra.embedding_strategies.local import LocalEmbeddingStrategy

logger = logging.getLogger(__name__)


class EmbeddingStrategyFactory:
    def build(self, runtime: RuntimeConfig, model: ModelConfig) -> EmbeddingStrategy:
        try:
            if runtime.embedding_api_url.strip():
                return HttpEmbeddingStrategy(runtime=runtime)
            return LocalEmbeddingStrategy(runtime=runtime, model=model)
        except Exception as exc:
            logger.exception(
                "Failed to build embedding strategy model_name=%s provider=%s",
                model.model_name,
                model.provider,
            )
            raise StrategyBuildError(
                "Failed to build embedding strategy model_name=%s provider=%s error=%s"
                % (model.model_name, model.provider, exc)
            ) from exc
