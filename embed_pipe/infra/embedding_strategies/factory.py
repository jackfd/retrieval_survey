from embed_pipe.domain.models import ModelConfig, RuntimeConfig
from embed_pipe.domain.result import Result
from embed_pipe.infra.embedding_strategies.base import EmbeddingStrategy
from embed_pipe.infra.embedding_strategies.http import HttpEmbeddingStrategy
from embed_pipe.infra.embedding_strategies.local import LocalEmbeddingStrategy


class EmbeddingStrategyFactory:
    def build(self, runtime: RuntimeConfig, model: ModelConfig) -> Result[EmbeddingStrategy]:
        try:
            if runtime.embedding_api_url.strip():
                return Result.success(HttpEmbeddingStrategy(runtime=runtime))
            return Result.success(LocalEmbeddingStrategy(runtime=runtime, model=model))
        except Exception:
            return Result.failure()
