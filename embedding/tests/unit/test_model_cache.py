"""共享模型缓存初始化的单元测试，覆盖默认缓存目录创建和已有环境变量保留行为。"""

from __future__ import annotations

import os
from pathlib import Path

from embedding.adapters import model_cache


def test_initialize_model_cache_sets_default_env_vars(monkeypatch, tmp_path):
    cache_root = tmp_path / "models"
    monkeypatch.delenv("RETRIEVAL_SURVEY_MODEL_CACHE_DIR", raising=False)
    for env_name in model_cache.MODEL_CACHE_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setattr(model_cache, "DEFAULT_MODEL_CACHE_ROOT", cache_root)

    result = model_cache.initialize_model_cache()

    assert result == cache_root
    assert Path(result).exists()
    assert Path(result).is_dir()
    assert Path(os.environ["RETRIEVAL_SURVEY_MODEL_CACHE_DIR"]) == cache_root

    for env_name, subdir_name in model_cache.MODEL_CACHE_ENV_VARS.items():
        expected = cache_root / subdir_name
        assert Path(os.environ[env_name]) == expected
        assert expected.exists()
        assert expected.is_dir()


def test_initialize_model_cache_preserves_existing_env_vars(monkeypatch, tmp_path):
    custom_cache_root = tmp_path / "custom-cache"
    sentinel_values = {
        "RETRIEVAL_SURVEY_MODEL_CACHE_DIR": str(custom_cache_root),
        "HF_HOME": str(tmp_path / "hf-home"),
        "HF_HUB_CACHE": str(tmp_path / "hf-hub"),
        "HUGGINGFACE_HUB_CACHE": str(tmp_path / "legacy-hf-hub"),
        "TRANSFORMERS_CACHE": str(tmp_path / "transformers"),
        "SENTENCE_TRANSFORMERS_HOME": str(tmp_path / "sentence-transformers"),
    }

    for env_name, env_value in sentinel_values.items():
        monkeypatch.setenv(env_name, env_value)

    result = model_cache.initialize_model_cache()

    assert result == custom_cache_root
    for env_name, env_value in sentinel_values.items():
        assert os.environ[env_name] == env_value
