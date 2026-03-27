import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from embed_pipe.domain.exceptions import ConfigError
from embed_pipe.domain.models import BuilderConfig, ModelConfig, RuntimeConfig


class ConfigLoader:
    def _load_yaml_mapping(self, path: Path) -> Dict[str, Any]:
        logger = logging.getLogger("embed_pipe")
        if not path.exists():
            logger.error("Missing config file path=%s", path)
            raise ConfigError("Missing config file path=%s" % path)

        with path.open("r", encoding="utf-8") as fin:
            cfg = yaml.safe_load(fin) or {}
        if not isinstance(cfg, dict):
            logger.error("Config file must be a mapping path=%s", path)
            raise ConfigError("Config file must be a mapping path=%s" % path)
        return cfg

    def _parse_runtime_config(self, cfg: Dict[str, Any]) -> RuntimeConfig:
        api_url = str(cfg.get("embedding_api_url", "")).strip()
        return RuntimeConfig(
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

    def _extract_models_mapping(
        self, cfg: Dict[str, Any], config_path: Path
    ) -> Dict[str, Dict[str, Any]]:
        logger = logging.getLogger("embed_pipe")
        models = cfg.get("models")
        if not isinstance(models, dict):
            logger.error(
                "Config key 'models' must be a mapping config_path=%s", config_path
            )
            raise ConfigError(
                "Config key 'models' must be a mapping config_path=%s" % config_path
            )
        if not models:
            logger.error(
                "Config key 'models' must not be empty config_path=%s", config_path
            )
            raise ConfigError(
                "Config key 'models' must not be empty config_path=%s" % config_path
            )
        return models

    def _build_model_config(
        self, model_name: str, model_obj: Dict[str, Any], config_path: Path
    ) -> ModelConfig:
        logger = logging.getLogger("embed_pipe")
        model = ModelConfig(
            model_name=model_name,
            provider=str(model_obj.get("provider", "")).strip(),
            model_id=str(model_obj.get("model_id", "")).strip(),
        )
        if not model.provider or not model.model_id:
            logger.error(
                "Model config requires non-empty provider and model_id config_path=%s model_name=%s",
                config_path,
                model_name,
            )
            raise ConfigError(
                "Model config requires non-empty provider and model_id config_path=%s model_name=%s"
                % (config_path, model_name)
            )
        return model

    def load_configs(self, config_path: Path) -> Dict[str, BuilderConfig]:
        logger = logging.getLogger("embed_pipe")
        cfg = self._load_yaml_mapping(config_path)
        runtime = self._parse_runtime_config(cfg)
        models = self._extract_models_mapping(cfg, config_path)
        builder_configs: Dict[str, BuilderConfig] = {}

        for model_name, model_obj in models.items():
            if not isinstance(model_obj, dict):
                logger.error(
                    "Model config must be a mapping config_path=%s model_name=%s",
                    config_path,
                    model_name,
                )
                raise ConfigError(
                    "Model config must be a mapping config_path=%s model_name=%s"
                    % (config_path, model_name)
                )
            model = self._build_model_config(model_name, model_obj, config_path)
            builder_configs[model_name] = BuilderConfig(
                runtime=runtime,
                model=model,
                raw_config=cfg,
            )
        return builder_configs
