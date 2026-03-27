import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from embedding.infra.embedding_strategies.base import BaseEmbeddingStrategy


class ConcreteEmbeddingStrategy(BaseEmbeddingStrategy):
    """为了测试BaseEmbeddingStrategy而创建的具体实现"""
    
    def encode(self, texts, is_query):
        # 返回模拟的嵌入向量
        return [[0.1] * self.experiment.embedding_dim for _ in texts]


class TestBaseEmbeddingStrategy:
    """测试基础嵌入策略"""
    
    def test_base_class_can_be_extended(self):
        """测试可以继承BaseEmbeddingStrategy"""
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

        strategy = ConcreteEmbeddingStrategy(
            experiment=experiment_cfg,
            inference=inference_cfg
        )
        
        # 测试encode方法
        sentences = ["这是一个测试句子"]
        embeddings = strategy.encode(sentences, is_query=False)
        
        # 验证返回了正确的嵌入向量
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384  # 维度应该匹配embedding_dim
    
    def test_prepare_texts_adds_prefixes(self):
        """测试文本前缀添加功能"""
        experiment_cfg = Mock()
        experiment_cfg.embedding_dim = 384
        experiment_cfg.normalize_embeddings = False
        experiment_cfg.max_length = 512
        experiment_cfg.query_prefix = "Q: "
        experiment_cfg.doc_prefix = "D: "
        experiment_cfg.instruction_template = ""

        inference_cfg = Mock()
        inference_cfg.batch_size = 16
        inference_cfg.device = "cpu"
        inference_cfg.embedding_api_url = ""
        inference_cfg.http_timeout = 10
        inference_cfg.http_max_retries = 3

        strategy = ConcreteEmbeddingStrategy(
            experiment=experiment_cfg,
            inference=inference_cfg
        )
        
        # 测试准备查询文本
        queries = ["这是一个查询"]
        prepared_queries = strategy._prepare_texts(queries, is_query=True)
        assert prepared_queries[0] == "Q: 这是一个查询"
        
        # 测试准备文档文本
        docs = ["这是一个文档"]
        prepared_docs = strategy._prepare_texts(docs, is_query=False)
        assert prepared_docs[0] == "D: 这是一个文档"
        
        # 测试指令模板
        experiment_cfg.instruction_template = "Encode for retrieval: {text}"
        strategy_with_template = ConcreteEmbeddingStrategy(
            experiment=experiment_cfg,
            inference=inference_cfg
        )
        templated_docs = strategy_with_template._prepare_texts(["这是一个文档"], is_query=False)
        assert templated_docs[0] == "Encode for retrieval: D: 这是一个文档"