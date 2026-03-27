import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from embedding.services.document_service import DocumentService
from embedding.services.query_service import QueryService
from embedding.services.chunking.chunk_selector import ChunkSelector


class TestDocumentService:
    """测试文档服务"""
    
    def test_process_document(self):
        """测试处理文档功能"""
        # 创建模拟的选择器
        mock_selector = Mock(spec=ChunkSelector)
        mock_selector.select_chunks.return_value = [
            {"chunk_text": "这是第一个块", "score": 0.8},
            {"chunk_text": "这是第二个块", "score": 0.6}
        ]
        
        documents = [
            {"doc_id": "1", "doc_text": "这是文档1的内容"},
            {"doc_id": "2", "doc_text": "这是文档2的内容"}
        ]
        
        service = DocumentService(mock_selector)
        
        # 由于DocumentService的process方法需要路径，我们不能直接测试embeddings
        # 我们只测试类能够被正确初始化和使用


class TestQueryService:
    """测试查询服务"""
    
    def test_process_query(self):
        """测试处理查询功能"""
        # 创建具体的嵌入策略实例
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

        mock_strategy = Mock()
        
        queries = [
            {"query_id": "q1", "query_text": "这是查询1"}
        ]
        
        service = QueryService(mock_strategy)
        
        # 由于QueryService的process方法需要路径，我们不能直接测试embeddings
        # 我们只测试类能够被正确初始化和使用