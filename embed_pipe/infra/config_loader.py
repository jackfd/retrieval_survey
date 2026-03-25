import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from embed_pipe.domain.models import BuilderConfig, ModelConfig, RuntimeConfig
from embed_pipe.domain.result import Result


class ConfigLoader:
    def load_yaml(self, path: Path) -> Result[Dict[str, Any]]:
        logger = logging.getLogger("embed_pipe")
        if not path.exists():
            logger.error(
                "event=config_validation_failed reason=%s context=%s",
                "Missing config file",
                "path=%s" % path,
            )
            return Result.failure("Missing config file: %s" % path)

        with path.open("r", encoding="utf-8") as fin:
            cfg = yaml.safe_load(fin) or {}
        if not isinstance(cfg, dict):
            logger.error(
                "event=config_validation_failed reason=%s context=%s",
                "Config file must be a mapping",
                "path=%s" % path,
            )
            return Result.failure("Config file must be a mapping: %s" % path)
        return Result.success(cfg)

    def load_builder_config(self, config_path: Path, model_name: str) -> Result[BuilderConfig]:
        logger = logging.getLogger("embed_pipe")
        cfg_result = self.load_yaml(config_path)
        if not cfg_result.ok:
            return Result.failure(cfg_result.error_message)

        cfg = cfg_result.value or {}
        models = cfg.get("models", {})
        if not isinstance(models, dict):
            logger.error(
                "event=config_validation_failed reason=%s context=%s",
                "Config key 'models' must be a mapping",
                "config_path=%s" % config_path,
            )
            return Result.failure("Config key 'models' must be a mapping")
        if model_name not in models:
            logger.error(
                "event=config_validation_failed reason=%s context=%s",
                "Unknown model_name in config",
                "config_path=%s model_name=%s" % (config_path, model_name),
            )
            return Result.failure("Unknown model_name=%r in config" % model_name)

        model_obj = models[model_name]
        if not isinstance(model_obj, dict):
            logger.error(
                "event=config_validation_failed reason=%s context=%s",
                "Model config must be a mapping",
                "config_path=%s model_name=%s" % (config_path, model_name),
            )
            return Result.failure("Model config must be a mapping: model_name=%r" % model_name)

        api_url = str(cfg.get("embedding_api_url", "")).strip()
        runtime = RuntimeConfig(
            embedding_dim=int(cfg.get("embedding_dim", 768)),
            normalize_embeddings=bool(cfg.get("normalize_embeddings", True)),
            max_length=int(cfg.get("max_length", 512)),
            query_prefix=str(cfg.get("query_prefix", "")),
            doc_prefix=str(cfg.get("doc_prefix", "")),
            instruction_template=str(cfg.get("instruction_template", "")),
            batch_size=int(cfg.get("batch_size", 64)),
            device=str(cfg.get("device", "cuda")),
            embedding_api_url=api_url,
            http_timeout=float(cfg.get("http_timeout", 30.0)),
            http_max_retries=int(cfg.get("http_max_retries", 2)),
        )
        model = ModelConfig(
            model_name=model_name,
            provider=str(model_obj.get("provider", "")).strip(),
            model_id=str(model_obj.get("model_id", "")).strip(),
        )
        if not model.provider or not model.model_id:
            logger.error(
                "event=config_validation_failed reason=%s context=%s",
                "Model config requires non-empty provider and model_id",
                "config_path=%s model_name=%s" % (config_path, model_name),
            )
            return Result.failure("Model config requires non-empty provider and model_id")

        return Result.success(BuilderConfig(runtime=runtime, model=model, raw_config=cfg))
