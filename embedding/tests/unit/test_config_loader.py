"""配置加载与校验的单元测试，覆盖 YAML 读取、默认值解析、模型列表提取、模型配置构建、重复 model_id 和非法模型项。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from embedding.domain.exceptions import ConfigError
from embedding.infra.config_loader import ConfigLoader


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


class TestConfigLoader:
    def test_load_yaml_mapping_success(self, tmp_path):
        config_data = {
            "experiment": {"embedding_dim": 768},
            "inference": {"batch_size": 16},
            "models": [{"model_id": "test-model", "provider": "test-provider"}],
        }

        loader = ConfigLoader()
        result = loader._load_yaml_mapping(_write_config(tmp_path, config_data))

        assert result == config_data

    def test_load_yaml_mapping_missing_file_raises_config_error(self):
        loader = ConfigLoader()

        with pytest.raises(ConfigError):
            loader._load_yaml_mapping(Path("missing-config.yaml"))

    def test_parse_experiment_config_uses_defaults(self):
        loader = ConfigLoader()
        result = loader._parse_experiment_config({"experiment": {}})

        assert result.embedding_dim == 768
        assert result.max_length == 512
        assert result.query_prefix == ""
        assert result.doc_prefix == ""
        assert result.instruction_template == ""

    def test_parse_experiment_config_requires_mapping(self):
        loader = ConfigLoader()

        with pytest.raises(ConfigError):
            loader._parse_experiment_config({})

    def test_parse_inference_config_uses_defaults(self):
        loader = ConfigLoader()
        result = loader._parse_inference_config({"inference": {}})

        assert result.batch_size == 64
        assert result.device == "cuda"
        assert result.embedding_api_url == ""
        assert result.http_timeout == 30.0
        assert result.http_max_retries == 2

    def test_parse_inference_config_requires_mapping(self):
        loader = ConfigLoader()

        with pytest.raises(ConfigError):
            loader._parse_inference_config({})

    def test_extract_models_list_success(self):
        loader = ConfigLoader()
        models = loader._extract_models_list(
            {
                "models": [
                    {"model_id": "model-a", "provider": "provider-a"},
                    {"model_id": "model-b", "provider": "provider-b"},
                ]
            }
        )

        assert [model["model_id"] for model in models] == ["model-a", "model-b"]

    def test_extract_models_list_rejects_missing_or_empty_models(self):
        loader = ConfigLoader()

        with pytest.raises(ConfigError):
            loader._extract_models_list({})

        with pytest.raises(ConfigError):
            loader._extract_models_list({"models": []})

    def test_build_model_config_trims_whitespace(self):
        loader = ConfigLoader()
        model = loader._build_model_config(
            {"model_id": "  model-x  ", "provider": "  provider-x  "}
        )

        assert model.model_id == "model-x"
        assert model.provider == "provider-x"

    def test_build_model_config_rejects_missing_fields(self):
        loader = ConfigLoader()

        with pytest.raises(ConfigError):
            loader._build_model_config({"model_id": "", "provider": "provider-x"})

        with pytest.raises(ConfigError):
            loader._build_model_config({"model_id": "model-x", "provider": ""})

    def test_load_configs_success(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "experiment": {
                    "embedding_dim": 384,
                    "max_length": 256,
                    "query_prefix": "Q: ",
                    "doc_prefix": "D: ",
                    "instruction_template": "Encode: {text}",
                },
                "inference": {
                    "batch_size": 8,
                    "device": "cpu",
                    "embedding_api_url": "",
                    "http_timeout": 12.0,
                    "http_max_retries": 3,
                },
                "models": [
                    {"model_id": "model-a", "provider": "provider-a"},
                    {"model_id": "model-b", "provider": "provider-b"},
                ],
            },
        )

        loader = ConfigLoader()
        configs = loader.load_configs(config_path)

        assert set(configs) == {"model-a", "model-b"}
        model_a = configs["model-a"]
        assert model_a.model.model_id == "model-a"
        assert model_a.model.provider == "provider-a"
        assert model_a.experiment.embedding_dim == 384
        assert model_a.experiment.instruction_template == "Encode: {text}"
        assert model_a.inference.batch_size == 8

    def test_load_configs_rejects_duplicate_model_id(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "experiment": {"embedding_dim": 384},
                "inference": {"batch_size": 8},
                "models": [
                    {"model_id": "model-a", "provider": "provider-a"},
                    {"model_id": "model-a", "provider": "provider-b"},
                ],
            },
        )

        loader = ConfigLoader()

        with pytest.raises(ConfigError):
            loader.load_configs(config_path)

    def test_load_configs_rejects_invalid_model_item(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            {
                "experiment": {"embedding_dim": 384},
                "inference": {"batch_size": 8},
                "models": ["not-a-mapping"],
            },
        )

        loader = ConfigLoader()

        with pytest.raises(ConfigError):
            loader.load_configs(config_path)
