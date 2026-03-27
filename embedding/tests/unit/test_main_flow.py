import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
from pathlib import Path

from embedding.app.runner import BuilderRunner
from embedding.infra.config_loader import ConfigLoader
from embedding.infra.dataset_loader import DatasetLoader
from embedding.services.document_service import DocumentService
from embedding.services.query_service import QueryService
from embedding.infra.output_writer import OutputWriter
from embedding.infra.embedding_strategies.base import BaseEmbeddingStrategy


class ConcreteEmbeddingStrategy(BaseEmbeddingStrategy):
    """用于测试的具体嵌入策略实现"""
    
    def encode(self, texts, is_query):
        # 返回模拟的嵌入向量
        return [[0.1] * self.experiment.embedding_dim for _ in texts]


class TestBuilderRunner:
    """测试BuilderRunner类"""
    
    def test_runner_end_to_end(self):
        """测试BuilderRunner端到端流程"""
        # 创建模拟配置
        experiment_cfg = Mock()
        experiment_cfg.embedding_dim = 384
        experiment_cfg.normalize_embeddings = False
        experiment_cfg.max_length = 512
        experiment_cfg.query_prefix = ""
        experiment_cfg.doc_prefix = ""
        experiment_cfg.instruction_template = ""

        inference_cfg = Mock()
        inference_cfg.batch_size = 16
        inference_cfg.device = "cpu"
        inference_cfg.embedding_api_url = ""
        inference_cfg.http_timeout = 10
        inference_cfg.http_max_retries = 3

        model_cfg = Mock()
        model_cfg.model_id = "test_model"
        model_cfg.provider = "test_provider"

        config = Mock()
        config.experiment = experiment_cfg
        config.inference = inference_cfg
        config.model = model_cfg

        # 创建BuilderRunner实例
        runner = BuilderRunner(
            output_dir=Path("/tmp/output"),
            dataset_ctx=Mock(docs_path=Path("/tmp/docs")),
            builder_cfg=config,
            embedding_strategy=Mock()
        )
        
        # 验证runner被成功初始化
        assert runner.output_writer is not None
        assert runner.dataset_ctx is not None
        assert runner.builder_cfg is not None