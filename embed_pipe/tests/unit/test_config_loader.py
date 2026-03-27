from pathlib import Path

import pytest

from embed_pipe.domain.exceptions import ConfigError
from embed_pipe.infra.config_loader import ConfigLoader


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_all_builder_configs_success(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
embedding_dim: 1024
normalize_embeddings: false
max_length: 2048
query_prefix: "Q: "
doc_prefix: "D: "
instruction_template: "ins"
batch_size: 16
device: "cpu"
embedding_api_url: "http://localhost:8080"
http_timeout: 12.5
http_max_retries: 7
models:
  m1:
    provider: sentence_transformers
    model_id: abc
  m2:
    provider: flag_embedding
    model_id: def
""",
    )

    loader = ConfigLoader()
    all_cfg = loader.load_all_builder_configs(config_path)

    assert list(all_cfg.keys()) == ["m1", "m2"]
    assert all_cfg["m1"].model.model_name == "m1"
    assert all_cfg["m1"].model.provider == "sentence_transformers"
    assert all_cfg["m1"].model.model_id == "abc"
    assert all_cfg["m2"].model.model_name == "m2"
    assert all_cfg["m2"].model.provider == "flag_embedding"
    assert all_cfg["m2"].model.model_id == "def"

    runtime = all_cfg["m1"].runtime
    assert runtime.embedding_dim == 1024
    assert runtime.normalize_embeddings is False
    assert runtime.max_length == 2048
    assert runtime.query_prefix == "Q: "
    assert runtime.doc_prefix == "D: "
    assert runtime.instruction_template == "ins"
    assert runtime.batch_size == 16
    assert runtime.device == "cpu"
    assert runtime.embedding_api_url == "http://localhost:8080"
    assert runtime.http_timeout == 12.5
    assert runtime.http_max_retries == 7

    assert all_cfg["m1"].raw_config["models"]["m2"]["model_id"] == "def"


def test_load_all_builder_configs_missing_file_raises() -> None:
    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_all_builder_configs(Path("not-exist-model-config.yaml"))


def test_load_all_builder_configs_top_level_non_mapping_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(config_path, "- a\n- b\n")

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_all_builder_configs(config_path)


def test_load_all_builder_configs_models_not_mapping_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
models:
  - m1
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_all_builder_configs(config_path)


def test_load_all_builder_configs_models_empty_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
models: {}
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_all_builder_configs(config_path)


def test_load_all_builder_configs_model_node_non_mapping_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
models:
  m1: "not a mapping"
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_all_builder_configs(config_path)


def test_load_all_builder_configs_model_required_fields_empty_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "model_config.yaml"
    _write_text(
        config_path,
        """
models:
  m1:
    provider: ""
    model_id: abc
""",
    )

    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.load_all_builder_configs(config_path)
