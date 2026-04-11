import logging
from pathlib import Path
from typing import Any, Dict, List
import yaml

from embedding.domain.exceptions import ConfigError
from embedding.domain.models import (
    BuilderConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    def _load_yaml_mapping(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.error("Missing config file path=%s", path)
            raise ConfigError("Missing config file path=%s" % path)

        with path.open("r", encoding="utf-8") as fin:
            cfg = yaml.safe_load(fin) or {}
        if not isinstance(cfg, dict):
            logger.error("Config file must be a mapping path=%s", path)
            raise ConfigError("Config file must be a mapping path=%s" % path)
        return cfg

    def _parse_experiment_config(self, cfg: Dict[str, Any]) -> ExperimentConfig:
        experiment_obj = cfg.get("experiment")
        if not isinstance(experiment_obj, dict):
            logger.error("Config key 'experiment' must be a mapping")
            raise ConfigError("Config key 'experiment' must be a mapping")
        return ExperimentConfig(
            embedding_dim=int(experiment_obj.get("embedding_dim", 768)),
            max_length=int(experiment_obj.get("max_length", 512)),
            query_prefix=str(experiment_obj.get("query_prefix", "")),
            doc_prefix=str(experiment_obj.get("doc_prefix", "")),
            instruction_template=str(experiment_obj.get("instruction_template", "")),
        )

    def _parse_inference_config(self, cfg: Dict[str, Any]) -> InferenceConfig:
        inference_obj = cfg.get("inference")
        if not isinstance(inference_obj, dict):
            logger.error("Config key 'inference' must be a mapping")
            raise ConfigError("Config key 'inference' must be a mapping")

        return InferenceConfig(
            batch_size=int(inference_obj.get("batch_size", 64)),
            device=str(inference_obj.get("device", "cuda")),
            embedding_api_url=str(inference_obj.get("embedding_api_url", "")).strip(),
            http_timeout=float(inference_obj.get("http_timeout", 30.0)),
            http_max_retries=int(inference_obj.get("http_max_retries", 2)),
        )

    def _build_model_experiment_config(
        self, default_experiment: ExperimentConfig, model_obj: Dict[str, Any]
    ) -> ExperimentConfig:
        return ExperimentConfig(
            embedding_dim=default_experiment.embedding_dim,
            max_length=default_experiment.max_length,
            query_prefix=str(
                model_obj.get("query_prefix", default_experiment.query_prefix)
            ),
            doc_prefix=str(model_obj.get("doc_prefix", default_experiment.doc_prefix)),
            instruction_template=str(
                model_obj.get(
                    "instruction_template", default_experiment.instruction_template
                )
            ),
        )

    def _build_model_inference_config(
        self, default_inference: InferenceConfig, model_obj: Dict[str, Any]
    ) -> InferenceConfig:
        return InferenceConfig(
            batch_size=int(model_obj.get("batch_size", default_inference.batch_size)),
            device=default_inference.device,
            embedding_api_url=default_inference.embedding_api_url,
            http_timeout=default_inference.http_timeout,
            http_max_retries=default_inference.http_max_retries,
        )

    def _parse_datasets(self, cfg: Dict[str, Any]) -> List[str]:
        datasets_obj = cfg.get("datasets")
        if not isinstance(datasets_obj, list):
            logger.error("Config key 'datasets' must be a list")
            raise ConfigError("Config key 'datasets' must be a list")
        if not datasets_obj:
            logger.error("Config key 'datasets' must not be empty")
            raise ConfigError("Config key 'datasets' must not be empty")

        datasets: List[str] = []
        for index, item in enumerate(datasets_obj):
            dataset_name = str(item).strip()
            if not dataset_name:
                logger.error(
                    "Dataset config item must be a non-empty string index=%s", index
                )
                raise ConfigError(
                    "Dataset config item must be a non-empty string index=%s" % index
                )
            datasets.append(dataset_name)
        return datasets

    def _extract_models_list(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        models = cfg.get("models")
        if not isinstance(models, list):
            logger.error("Config key 'models' must be a list")
            raise ConfigError("Config key 'models' must be a list")
        if not models:
            logger.error("Config key 'models' must not be empty")
            raise ConfigError("Config key 'models' must not be empty")
        return models

    def _build_model_config(self, model_obj: Dict[str, Any]) -> ModelConfig:
        model_id = str(model_obj.get("model_id", "")).strip()
        provider = str(model_obj.get("provider", "")).strip()
        trust_remote_code = bool(model_obj.get("trust_remote_code", False))
        model = ModelConfig(provider, model_id, trust_remote_code)
        if not model.provider or not model.model_id:
            logger.error(
                "Model config requires non-empty provider and model_id model_id=%s",
                model_id,
            )
            raise ConfigError(
                "Model config requires non-empty provider and model_id model_id=%s"
                % (model_id)
            )
        return model

    def load_configs(self, config_path: Path) -> PipelineConfig:
        logger.info("Loading configs from %s", config_path)
        cfg = self._load_yaml_mapping(config_path)
        datasets = self._parse_datasets(cfg)
        default_experiment = self._parse_experiment_config(cfg)
        default_inference = self._parse_inference_config(cfg)
        models = self._extract_models_list(cfg)

        builder_configs: Dict[str, BuilderConfig] = {}
        for index, model_obj in enumerate(models):
            if not isinstance(model_obj, dict):
                logger.error(
                    "Model config item must be a mapping config_path=%s index=%s",
                    config_path,
                    index,
                )
                raise ConfigError(
                    "Model config item must be a mapping config_path=%s index=%s"
                    % (config_path, index)
                )
            model = self._build_model_config(model_obj)
            experiment = self._build_model_experiment_config(
                default_experiment, model_obj
            )
            inference = self._build_model_inference_config(
                default_inference, model_obj
            )
            if model.model_id in builder_configs:
                logger.error(
                    "Duplicate model_id in config config_path=%s model_id=%s",
                    config_path,
                    model.model_id,
                )
                raise ConfigError(
                    "Duplicate model_id in config config_path=%s model_id=%s"
                    % (config_path, model.model_id)
                )
            builder_configs[model.model_id] = BuilderConfig(
                experiment=experiment,
                inference=inference,
                model=model,
                raw_config=cfg,
            )
        return PipelineConfig(datasets=datasets, builders=builder_configs)
