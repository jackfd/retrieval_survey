"""Entrypoint tests for main() failure handling and input validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from embedding.domain.exceptions import DatasetError
from embedding.domain.models import BuilderConfig, ExperimentConfig, InferenceConfig, ModelConfig
from embedding import main as main_module


def _builder_config() -> BuilderConfig:
    return BuilderConfig(
        experiment=ExperimentConfig(
            embedding_dim=384,
            max_length=512,
            query_prefix="",
            doc_prefix="",
            instruction_template="",
        ),
        inference=InferenceConfig(
            batch_size=8,
            device="cpu",
            embedding_api_url="",
            http_timeout=30.0,
            http_max_retries=2,
        ),
        model=ModelConfig(provider="sentence_transformers", model_id="model-a"),
        raw_config={},
    )


def test_main_logs_unexpected_exception_with_traceback(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text("experiment: {}\ninference: {}\nmodels: []\n", encoding="utf-8")

    logger = Mock()
    monkeypatch.setattr(main_module, "setup_logger", lambda _path: logger)
    monkeypatch.setattr(main_module, "initialize_model_cache", lambda: tmp_path / "cache")
    monkeypatch.setattr(
        main_module.ConfigLoader,
        "load_configs",
        lambda _self, _path: {"model-a": _builder_config()},
    )
    monkeypatch.setattr(
        main_module.EmbeddingStrategyFactory,
        "build",
        lambda _self, _exp, _inf, _model: object(),
    )
    monkeypatch.setattr(
        main_module,
        "run_once",
        lambda _dataset_path, _dataset_name, _cfg, _embedding: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    result = main_module.main(str(dataset_root), str(config_path))

    assert result == 1
    logger.exception.assert_called_once()
    message, error_type, error, model_id, dataset_name = logger.exception.call_args.args
    assert "Pipeline failed with unexpected error" in message
    assert error_type == "RuntimeError"
    assert str(error) == "boom"
    assert model_id == "model-a"
    assert dataset_name == main_module.DATA_SETS[0]


def test_main_logs_domain_exception_with_context(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text("experiment: {}\ninference: {}\nmodels: []\n", encoding="utf-8")

    logger = Mock()
    monkeypatch.setattr(main_module, "setup_logger", lambda _path: logger)
    monkeypatch.setattr(main_module, "initialize_model_cache", lambda: tmp_path / "cache")
    monkeypatch.setattr(
        main_module.ConfigLoader,
        "load_configs",
        lambda _self, _path: {"model-a": _builder_config()},
    )
    monkeypatch.setattr(
        main_module.EmbeddingStrategyFactory,
        "build",
        lambda _self, _exp, _inf, _model: object(),
    )
    monkeypatch.setattr(
        main_module,
        "run_once",
        lambda _dataset_path, _dataset_name, _cfg, _embedding: (_ for _ in ()).throw(
            DatasetError("missing dataset")
        ),
    )

    result = main_module.main(str(dataset_root), str(config_path))

    assert result == 1
    logger.error.assert_called_once()
    message, error, model_id, dataset_name = logger.error.call_args.args
    assert "Pipeline failed with domain error" in message
    assert str(error) == "missing dataset"
    assert model_id == "model-a"
    assert dataset_name == main_module.DATA_SETS[0]


def test_main_rejects_missing_dataset_path(monkeypatch, tmp_path):
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text("experiment: {}\ninference: {}\nmodels: []\n", encoding="utf-8")

    logger = Mock()
    monkeypatch.setattr(main_module, "setup_logger", lambda _path: logger)

    result = main_module.main(str(tmp_path / "missing-datasets"), str(config_path))

    assert result == 1
    logger.exception.assert_called_once()
    message, error_type, error, model_id, dataset_name = logger.exception.call_args.args
    assert "Pipeline failed with unexpected error" in message
    assert error_type == "FileNotFoundError"
    assert "Dataset path does not exist" in str(error)
    assert model_id is None
    assert dataset_name is None


def test_run_once_builds_shared_candidates_once(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    ds_context = Mock(
        docs_path=tmp_path / "datasets" / "docs.jsonl",
        queries_path=tmp_path / "datasets" / "train" / "queries.jsonl",
    )
    output_writer = Mock()
    def _build_candidates(_docs_path, target_path, _max_length):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("", encoding="utf-8")

    build_doc_candidates = Mock(side_effect=_build_candidates)
    process_doc = Mock()
    process_query = Mock()

    monkeypatch.setattr(main_module, "OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setattr(
        main_module.DatasetLoader,
        "load_dataset_context",
        lambda _self, _root, _dataset: ds_context,
    )
    monkeypatch.setattr(
        main_module, "OutputWriter", lambda *_args, **_kwargs: output_writer
    )
    monkeypatch.setattr(main_module, "build_doc_candidates", build_doc_candidates)
    monkeypatch.setattr(main_module, "process_doc", process_doc)
    monkeypatch.setattr(main_module, "process_query", process_query)

    cfg = _builder_config()
    embedding = object()

    main_module.run_once(str(dataset_root), "scifact_v1", cfg, embedding)
    main_module.run_once(str(dataset_root), "scifact_v1", cfg, embedding)

    candidates_path = Path(main_module.OUTPUT_ROOT) / "scifact_v1_candidates.jsonl"
    assert build_doc_candidates.call_count == 1
    assert build_doc_candidates.call_args_list[0].args == (
        ds_context.docs_path,
        candidates_path,
        cfg.experiment.max_length,
    )
    assert process_doc.call_args_list[0].args == (
        embedding,
        candidates_path,
        output_writer,
    )
