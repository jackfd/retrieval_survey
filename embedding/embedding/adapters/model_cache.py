"""Utilities for configuring shared model cache directories."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_CACHE_ROOT = Path.home() / ".cache" / "retrieval_survey" / "models"
MODEL_CACHE_ENV_VARS = {
    "HF_HOME": "hf_home",
    "HF_HUB_CACHE": "hf_hub",
    "HUGGINGFACE_HUB_CACHE": "hf_hub",
    "SENTENCE_TRANSFORMERS_HOME": "sentence_transformers",
}


def initialize_model_cache() -> Path:
    """Create and expose a stable local cache root for model downloads."""

    configured_root = os.environ.get("RETRIEVAL_SURVEY_MODEL_CACHE_DIR")
    cache_root = Path(configured_root or DEFAULT_MODEL_CACHE_ROOT).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    for env_name, subdir_name in MODEL_CACHE_ENV_VARS.items():
        cache_dir = cache_root / subdir_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(cache_dir))

    os.environ.setdefault("RETRIEVAL_SURVEY_MODEL_CACHE_DIR", str(cache_root))
    return cache_root
