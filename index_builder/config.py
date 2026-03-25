from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from index_builder.errors import InputValidationError


@dataclass
class RuntimeConfig:
    embedding_dim: int
    normalize_embeddings: bool
    max_length: int
    query_prefix: str
    doc_prefix: str
    instruction_template: str
    batch_size: int
    device: str
    embedding_mode: str
    embedding_api_url: str
    http_timeout: float
    http_max_retries: int


@dataclass
class ModelConfig:
    model_name: str
    provider: str
    model_id: str


@dataclass
class BuilderConfig:
    runtime: RuntimeConfig
    model: ModelConfig
    raw_config: Dict[str, Any]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise InputValidationError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fin:
        cfg = yaml.safe_load(fin) or {}
    if not isinstance(cfg, dict):
        raise InputValidationError(f"Config file must be a mapping: {path}")
    return cfg


def load_builder_config(config_path: Path, model_name: str) -> BuilderConfig:
    cfg = load_yaml(config_path)
    models = cfg.get("models", {})
    if not isinstance(models, dict):
        raise InputValidationError("Config key 'models' must be a mapping")
    if model_name not in models:
        raise InputValidationError(f"Unknown model_name={model_name!r} in config")

    model_obj = models[model_name]
    if not isinstance(model_obj, dict):
        raise InputValidationError(f"Model config must be a mapping: model_name={model_name!r}")

    mode = str(cfg.get("embedding_mode", "local")).strip().lower()
    if mode not in {"local", "http"}:
        raise InputValidationError("embedding_mode must be one of: local, http")

    api_url = str(cfg.get("embedding_api_url", "")).strip()
    if mode == "http" and not api_url:
        raise InputValidationError("embedding_api_url is required when embedding_mode=http")

    runtime = RuntimeConfig(
        embedding_dim=int(cfg.get("embedding_dim", 768)),
        normalize_embeddings=bool(cfg.get("normalize_embeddings", True)),
        max_length=int(cfg.get("max_length", 512)),
        query_prefix=str(cfg.get("query_prefix", "")),
        doc_prefix=str(cfg.get("doc_prefix", "")),
        instruction_template=str(cfg.get("instruction_template", "")),
        batch_size=int(cfg.get("batch_size", 64)),
        device=str(cfg.get("device", "cuda")),
        embedding_mode=mode,
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
        raise InputValidationError("Model config requires non-empty provider and model_id")

    return BuilderConfig(runtime=runtime, model=model, raw_config=cfg)
