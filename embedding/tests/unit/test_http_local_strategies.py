"""Unit tests for HTTP and local embedding strategies."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ensure_base_and_shape_modules():
    base_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "base.py"
    )
    shape_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "shape.py"
    )
    base_module = _load_module_from_file(
        "embedding.infra.embedding_strategies.base", base_path
    )
    shape_module = _load_module_from_file(
        "embedding.infra.embedding_strategies.shape", shape_path
    )

    fake_pkg = types.ModuleType("embedding.infra.embedding_strategies")
    fake_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "infra" / "embedding_strategies")
    ]
    fake_pkg.BaseEmbeddingStrategy = base_module.BaseEmbeddingStrategy
    fake_pkg.EmbeddingStrategy = base_module.EmbeddingStrategy
    fake_pkg.ensure_embedding_shape = shape_module.ensure_embedding_shape
    return fake_pkg, base_module, shape_module


def _load_http_module():
    fake_pkg, base_module, shape_module = _ensure_base_and_shape_modules()
    http_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "http.py"
    )
    with patch.dict(
        sys.modules,
        {
            "embedding.infra.embedding_strategies": fake_pkg,
            "embedding.infra.embedding_strategies.base": base_module,
            "embedding.infra.embedding_strategies.shape": shape_module,
        },
    ):
        return _load_module_from_file(
            "embedding.infra.embedding_strategies.http", http_path
        )


def _load_local_module(sentence_transformers_ctor=None, flag_ctor=None):
    fake_pkg, base_module, shape_module = _ensure_base_and_shape_modules()
    local_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "local.py"
    )

    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = sentence_transformers_ctor or Mock()
    fake_flag_module = types.ModuleType("FlagEmbedding")
    fake_flag_module.BGEM3FlagModel = flag_ctor or Mock()

    with patch.dict(
        sys.modules,
        {
            "embedding.infra.embedding_strategies": fake_pkg,
            "embedding.infra.embedding_strategies.base": base_module,
            "embedding.infra.embedding_strategies.shape": shape_module,
            "sentence_transformers": fake_st_module,
            "FlagEmbedding": fake_flag_module,
        },
    ):
        return _load_module_from_file(
            "embedding.infra.embedding_strategies.local", local_path
        )


def _build_experiment_cfg(
    *,
    embedding_dim: int = 384,
    normalize_embeddings: bool = False,
    instruction_template: str = "",
):
    cfg = Mock()
    cfg.embedding_dim = embedding_dim
    cfg.normalize_embeddings = normalize_embeddings
    cfg.max_length = 512
    cfg.query_prefix = ""
    cfg.doc_prefix = ""
    cfg.instruction_template = instruction_template
    return cfg


def _build_inference_cfg(device="cpu", batch_size=2, *, max_retries=3):
    cfg = Mock()
    cfg.batch_size = batch_size
    cfg.device = device
    cfg.embedding_api_url = ""
    cfg.http_timeout = 10
    cfg.http_max_retries = max_retries
    return cfg


class TestHttpEmbeddingStrategy:
    def test_encode_single_batch_success(self):
        http_module = _load_http_module()
        strategy = http_module.HttpEmbeddingStrategy(
            experiment=_build_experiment_cfg(),
            inference=Mock(
                batch_size=2,
                device="cpu",
                embedding_api_url="http://test.api/embed",
                http_timeout=10,
                http_max_retries=3,
            ),
        )

        texts = ["Hello world", "Test sentence"]
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "vectors": [[0.1] * 384, [0.2] * 384]
            }
            mock_post.return_value = mock_response

            result = strategy.encode(texts, is_query=True)

        assert result.shape == (2, 384)
        assert abs(float(result[0][0]) - 0.1) < 1e-6
        assert abs(float(result[1][0]) - 0.2) < 1e-6
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["timeout"] == 10
        assert mock_post.call_args.kwargs["json"] == {
            "chunks": ["Hello world", "Test sentence"]
        }

    def test_encode_multiple_batches_success(self):
        http_module = _load_http_module()
        strategy = http_module.HttpEmbeddingStrategy(
            experiment=_build_experiment_cfg(),
            inference=Mock(
                batch_size=2,
                device="cpu",
                embedding_api_url="http://test.api/embed",
                http_timeout=10,
                http_max_retries=3,
            ),
        )

        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        with patch("requests.post") as mock_post:

            def side_effect(*args, **kwargs):
                mock_resp = Mock()
                mock_resp.status_code = 200
                mock_resp.raise_for_status.return_value = None
                chunks = kwargs["json"]["chunks"]
                mock_resp.json.return_value = {
                    "vectors": [[i * 0.1] * 384 for i in range(1, len(chunks) + 1)]
                }
                return mock_resp

            mock_post.side_effect = side_effect
            result = strategy.encode(texts, is_query=True)

        assert result.shape == (5, 384)
        assert mock_post.call_count == 3
        assert mock_post.call_args_list[0].kwargs["json"]["chunks"] == [
            "Text 1",
            "Text 2",
        ]
        assert mock_post.call_args_list[1].kwargs["json"]["chunks"] == [
            "Text 3",
            "Text 4",
        ]
        assert mock_post.call_args_list[2].kwargs["json"]["chunks"] == ["Text 5"]

    def test_encode_empty_texts_returns_empty_matrix(self):
        http_module = _load_http_module()
        strategy = http_module.HttpEmbeddingStrategy(
            experiment=_build_experiment_cfg(),
            inference=Mock(
                batch_size=2,
                device="cpu",
                embedding_api_url="http://test.api/embed",
                http_timeout=10,
                http_max_retries=3,
            ),
        )

        with patch("requests.post") as mock_post:
            result = strategy.encode([], is_query=False)

        assert result.shape == (0, 384)
        mock_post.assert_not_called()

    def test_encode_retries_and_raises_runtime_error(self):
        http_module = _load_http_module()
        strategy = http_module.HttpEmbeddingStrategy(
            experiment=_build_experiment_cfg(),
            inference=Mock(
                batch_size=2,
                device="cpu",
                embedding_api_url="http://test.api/embed",
                http_timeout=10,
                http_max_retries=1,
            ),
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"vectors": []}
            mock_post.return_value = mock_response

            with pytest.raises(RuntimeError):
                strategy.encode(["Text 1"], is_query=True)

        assert mock_post.call_count == 2


class TestLocalEmbeddingStrategy:
    def test_sentence_transformers_provider_initializes_and_encodes(self):
        created = {}

        class DummySentenceTransformer:
            def __init__(self, model_id, device):
                created["model_id"] = model_id
                created["device"] = device
                self.max_seq_length = 0

            def encode(
                self,
                texts,
                batch_size,
                normalize_embeddings,
                convert_to_numpy,
            ):
                return np.asarray([[0.1, 0.2, 0.3, 0.4] for _ in texts], dtype=np.float32)

        local_module = _load_local_module(sentence_transformers_ctor=DummySentenceTransformer)

        strategy = local_module.LocalEmbeddingStrategy(
            experiment=_build_experiment_cfg(embedding_dim=4),
            inference=_build_inference_cfg(device="cpu", batch_size=4),
            model=Mock(provider="sentence_transformers", model_id="test-model-id"),
        )

        assert strategy._encoder[0] == "sentence_transformers"
        assert created == {"model_id": "test-model-id", "device": "cpu"}
        assert strategy._encoder[1].max_seq_length == 512

        result = strategy.encode(["hello"], is_query=True)

        assert result.shape == (1, 4)
        assert abs(float(result[0][0]) - 0.1) < 1e-6

    def test_flag_embedding_provider_initializes_and_encodes(self):
        created = {}

        class DummyFlagModel:
            def __init__(self, model_id, use_fp16):
                created["model_id"] = model_id
                created["use_fp16"] = use_fp16

            def encode(
                self,
                texts,
                batch_size,
                max_length,
                return_dense=None,
                return_sparse=None,
                return_colbert_vecs=None,
            ):
                return {
                    "dense_vecs": [
                        [3.0, 4.0, 0.0, 0.0] for _ in texts
                    ]
                }

        local_module = _load_local_module(flag_ctor=DummyFlagModel)

        strategy = local_module.LocalEmbeddingStrategy(
            experiment=_build_experiment_cfg(
                embedding_dim=4, normalize_embeddings=True
            ),
            inference=_build_inference_cfg(device="cuda:0", batch_size=4),
            model=Mock(provider="flag_embedding", model_id="bge-m3"),
        )

        assert strategy._encoder[0] == "flag_embedding"
        assert created == {"model_id": "bge-m3", "use_fp16": True}

        result = strategy.encode(["hello"], is_query=False)

        assert result.shape == (1, 4)
        assert abs(float(result[0][0]) - 0.6) < 1e-6
        assert abs(float(result[0][1]) - 0.8) < 1e-6

    def test_unsupported_provider_raises_runtime_error(self):
        local_module = _load_local_module()

        with pytest.raises(RuntimeError):
            local_module.LocalEmbeddingStrategy(
                experiment=_build_experiment_cfg(),
                inference=_build_inference_cfg(device="cpu"),
                model=Mock(provider="unknown", model_id="bad-model"),
            )
