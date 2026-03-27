"""Unit tests for the CLI-level main orchestration."""

from __future__ import annotations

import importlib
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from unittest.mock import Mock, patch


@dataclass
class _DummyModel:
    provider: str
    model_id: str


@dataclass
class _DummyBuilderConfig:
    experiment: object
    inference: object
    model: _DummyModel
    raw_config: dict


def _load_main_module(builder_configs):
    fake_runner_module = types.ModuleType("embedding.app.runner")
    fake_runner_module.BuilderRunner = object

    fake_config_loader_module = types.ModuleType("embedding.infra.config_loader")

    class _ConfigLoader:
        def load_configs(self, config_path):
            return builder_configs

    fake_config_loader_module.ConfigLoader = _ConfigLoader
    fake_config_loader_module.BuilderConfig = _DummyBuilderConfig

    fake_dataset_loader_module = types.ModuleType("embedding.infra.dataset_loader")

    class _DatasetLoader:
        def load_dataset_context(self, dataset_root, dataset_name):
            return Mock(
                dataset_root=dataset_root,
                resolved_dataset_dir=f"{dataset_root}/{dataset_name}",
                docs_path=f"{dataset_root}/{dataset_name}/docs.jsonl",
                queries_path=f"{dataset_root}/{dataset_name}/queries.jsonl",
            )

    fake_dataset_loader_module.DatasetLoader = _DatasetLoader

    fake_embedding_strategies_module = types.ModuleType(
        "embedding.infra.embedding_strategies"
    )

    class _EmbeddingStrategyFactory:
        def __init__(self):
            self.build_calls = []

        def build(self, experiment, inference, model):
            self.build_calls.append((experiment, inference, model))
            return object()

    fake_embedding_strategies_module.EmbeddingStrategyFactory = _EmbeddingStrategyFactory

    fake_logger_module = types.ModuleType("embedding.infra.logger")
    fake_logger_module.setup_logger = lambda *args, **kwargs: Mock(info=Mock())

    for module_name in [
        "embedding.main",
        "embedding.app.runner",
        "embedding.infra.config_loader",
        "embedding.infra.dataset_loader",
        "embedding.infra.embedding_strategies",
        "embedding.infra.logger",
    ]:
        sys.modules.pop(module_name, None)

    with patch.dict(
        sys.modules,
        {
            "embedding.app.runner": fake_runner_module,
            "embedding.infra.config_loader": fake_config_loader_module,
            "embedding.infra.dataset_loader": fake_dataset_loader_module,
            "embedding.infra.embedding_strategies": fake_embedding_strategies_module,
            "embedding.infra.logger": fake_logger_module,
        },
    ):
        return importlib.import_module("embedding.main")


def test_main_reuses_embedding_strategy_per_model_id():
    builder_cfg = _DummyBuilderConfig(
        experiment=Mock(name="experiment"),
        inference=Mock(name="inference"),
        model=_DummyModel(provider="provider-a", model_id="model-a"),
        raw_config={},
    )
    main_module = _load_main_module(OrderedDict([("model-a", builder_cfg)]))

    run_once_calls = []
    build_calls = []

    factory_instance = Mock()
    factory_instance.build.side_effect = lambda experiment, inference, model: object()

    def fake_factory():
        return factory_instance

    def fake_run_once(dataset_root, dataset_name, received_builder_cfg, embedding_strategy):
        run_once_calls.append(
            (dataset_root, dataset_name, received_builder_cfg.model.model_id, embedding_strategy)
        )

    with patch.object(main_module, "EmbeddingStrategyFactory", side_effect=fake_factory):
        with patch.object(main_module, "run_once", side_effect=fake_run_once):
            result = main_module.main("/datasets")

    assert result == 0
    assert factory_instance.build.call_count == 1
    assert [call[1] for call in run_once_calls] == main_module.DATA_SETS
    assert {call[2] for call in run_once_calls} == {"model-a"}
    assert len({id(call[3]) for call in run_once_calls}) == 1
