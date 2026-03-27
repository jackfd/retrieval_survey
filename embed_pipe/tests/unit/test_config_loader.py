from pathlib import Path

import pytest

from embed_pipe.domain.exceptions import ConfigError
from embed_pipe.infra.config_loader import ConfigLoader


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_configs_success(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
experiment:
  embedding_dim: 1024
  normalize_embeddings: false
  max_length: 2048
  query_prefix: "Q: "
  doc_prefix: "D: "
  instruction_template: "ins"
inference:
  batch_size: 16
  device: "cpu"
  embedding_api_url: "http://localhost:8080"
  http_timeout: 12.5
  http_max_retries: 7
models:
  - model_id: abc
    provider: sentence_transformers
  - model_id: def
    provider: flag_embedding
""",
    )

    loader = ConfigLoader()
    all_cfg = loader.load_configs(config_path)

    assert list(all_cfg.keys()) == ["abc", "def"]
    assert all_cfg["abc"].model.model_id == "abc"
    assert all_cfg["abc"].model.provider == "sentence_transformers"
    assert all_cfg["def"].model.model_id == "def"
    assert all_cfg["def"].model.provider == "flag_embedding"

    experiment = all_cfg["abc"].experiment
    assert experiment.embedding_dim == 1024
    assert experiment.normalize_embeddings is False
    assert experiment.max_length == 2048
    assert experiment.query_prefix == "Q: "
    assert experiment.doc_prefix == "D: "
    assert experiment.instruction_template == "ins"

    inference = all_cfg["abc"].inference
    assert inference.batch_size == 16
    assert inference.device == "cpu"
    assert inference.embedding_api_url == "http://localhost:8080"
    assert inference.http_timeout == 12.5
    assert inference.http_max_retries == 7


def test_load_configs_missing_file_raises() -> None:
    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(Path("not-exist-model-config.yaml"))


def test_load_configs_top_level_non_mapping_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(config_path, "- a\n- b\n")

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(config_path)


def test_load_configs_missing_experiment_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
inference:
  batch_size: 16
  device: "cpu"
  embedding_api_url: ""
  http_timeout: 12.5
  http_max_retries: 7
models:
  - model_id: abc
    provider: sentence_transformers
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(config_path)


def test_load_configs_missing_inference_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
experiment:
  embedding_dim: 1024
  normalize_embeddings: false
  max_length: 2048
  query_prefix: "Q: "
  doc_prefix: "D: "
  instruction_template: "ins"
models:
  - model_id: abc
    provider: sentence_transformers
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(config_path)


def test_load_configs_models_not_list_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
experiment:
  embedding_dim: 1024
  normalize_embeddings: false
  max_length: 2048
  query_prefix: "Q: "
  doc_prefix: "D: "
  instruction_template: "ins"
inference:
  batch_size: 16
  device: "cpu"
  embedding_api_url: ""
  http_timeout: 12.5
  http_max_retries: 7
models: {}
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(config_path)


def test_load_configs_models_empty_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
experiment:
  embedding_dim: 1024
  normalize_embeddings: false
  max_length: 2048
  query_prefix: "Q: "
  doc_prefix: "D: "
  instruction_template: "ins"
inference:
  batch_size: 16
  device: "cpu"
  embedding_api_url: ""
  http_timeout: 12.5
  http_max_retries: 7
models: []
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(config_path)


def test_load_configs_model_item_non_mapping_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
experiment:
  embedding_dim: 1024
  normalize_embeddings: false
  max_length: 2048
  query_prefix: "Q: "
  doc_prefix: "D: "
  instruction_template: "ins"
inference:
  batch_size: 16
  device: "cpu"
  embedding_api_url: ""
  http_timeout: 12.5
  http_max_retries: 7
models:
  - "not a mapping"
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(config_path)


def test_load_configs_model_required_fields_empty_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
experiment:
  embedding_dim: 1024
  normalize_embeddings: false
  max_length: 2048
  query_prefix: "Q: "
  doc_prefix: "D: "
  instruction_template: "ins"
inference:
  batch_size: 16
  device: "cpu"
  embedding_api_url: ""
  http_timeout: 12.5
  http_max_retries: 7
models:
  - model_id: ""
    provider: sentence_transformers
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_configs(config_path)
