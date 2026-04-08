import logging
from typing import Any, List, Tuple

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
        self._effective_max_length = int(self.experiment.max_length)
        self._encoder = self._build_local_encoder(model=model)

    def _build_local_encoder(self, model: ModelConfig) -> Tuple[str, object]:
        if model.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(
                model.model_id,
                device=self.inference.device,
                trust_remote_code=model.trust_remote_code,
            )
            if hasattr(encoder, "max_seq_length"):
                self._effective_max_length = self._resolve_effective_max_length(
                    getattr(encoder, "max_seq_length", None)
                )
                encoder.max_seq_length = self._effective_max_length
            logger.info(
                "Initializing sentence_transformers model=%s device=%s batch_size=%s max_seq_length=%s trust_remote_code=%s",
                model.model_id,
                self.inference.device,
                self.inference.batch_size,
                encoder.max_seq_length,
                model.trust_remote_code,
            )
            return ("sentence_transformers", encoder)

        if model.provider == "flag_embedding":
            from FlagEmbedding import BGEM3FlagModel

            # BGEM3FlagModel does not consistently expose an explicit device argument
            # across library versions. Stage-one throughput work relies on the caller
            # passing larger batches; device placement stays library-managed here.
            logger.info(
                "Initializing FlagEmbedding model=%s device=%s batch_size=%s max_seq_length=%s use_fp16=%s",
                model.model_id,
                self.inference.device,
                self.inference.batch_size,
                self._effective_max_length,
                str(self.inference.device).startswith("cuda"),
            )
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
            self._log_input_lengths(prepared, encoder=encoder, is_query=is_query)
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
                logger.info(
                    "FlagEmbedding model=%s does not support dense_vecs",
                    self.model.model_id,
                )
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

    def describe_input_lengths(self, texts, is_query):
        prepared = self._prepare_texts(texts, is_query=is_query)
        provider, encoder = self._encoder

        if provider != "sentence_transformers":
            return {
                "token_stats_available": False,
                "max_prepared_tokens": None,
                "avg_prepared_tokens": None,
                "prepared_over_limit_count": None,
            }

        token_lengths = self._compute_prepared_token_lengths(prepared, encoder=encoder)
        if token_lengths is None:
            return {
                "token_stats_available": False,
                "max_prepared_tokens": None,
                "avg_prepared_tokens": None,
                "prepared_over_limit_count": None,
            }

        return {
            "token_stats_available": True,
            "max_prepared_tokens": max(token_lengths, default=0),
            "avg_prepared_tokens": (
                float(sum(token_lengths)) / len(token_lengths) if token_lengths else 0.0
            ),
            "prepared_over_limit_count": sum(
                1 for count in token_lengths if count > self._effective_max_length
            ),
        }

    def _resolve_effective_max_length(self, encoder_limit) -> int:
        requested = int(self.experiment.max_length)
        try:
            resolved = int(encoder_limit)
        except (TypeError, ValueError):
            resolved = 0
        if resolved > 0:
            return min(requested, resolved)
        return requested

    def _log_input_lengths(
        self, prepared: List[str], encoder: Any, is_query: bool
    ) -> None:
        token_lengths = self._compute_prepared_token_lengths(prepared, encoder=encoder)
        if token_lengths is None:
            return

        input_count = len(token_lengths)
        min_tokens = min(token_lengths, default=0)
        max_tokens = max(token_lengths, default=0)
        avg_tokens = float(sum(token_lengths)) / input_count if input_count else 0.0
        over_limit_count = sum(
            1 for count in token_lengths if count > self._effective_max_length
        )
        logger.info(
            "Local embedding input token stats model_id=%s input_count=%s min_tokens=%s max_tokens=%s avg_tokens=%.2f over_limit_count=%s effective_max_length=%s",
            self.model.model_id,
            input_count,
            min_tokens,
            max_tokens,
            avg_tokens,
            over_limit_count,
            self._effective_max_length,
        )

        warning_threshold = int(self._effective_max_length * 0.8)
        if warning_threshold <= 0:
            return
        for index, token_count in enumerate(token_lengths):
            if token_count >= warning_threshold:
                logger.warning(
                    "Local embedding input is unusually long model_id=%s sample_index=%s token_count=%s warning_threshold=%s effective_max_length=%s",
                    self.model.model_id,
                    index,
                    token_count,
                    warning_threshold,
                    self._effective_max_length,
                )

    def _compute_prepared_token_lengths(
        self, prepared: List[str], encoder: Any
    ) -> List[int] | None:
        if not prepared:
            return []
        tokenizer = getattr(encoder, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            payload = tokenizer(
                prepared,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            input_ids = payload.get("input_ids") if isinstance(payload, dict) else None
            if input_ids is None:
                return None
            return [len(ids) for ids in input_ids]
        except Exception as exc:
            logger.warning(
                "Local embedding token stats unavailable model_id=%s error_type=%s",
                self.model.model_id,
                type(exc).__name__,
            )
            return None
