from pathlib import Path
import pytest
import yaml

from embedding.services.chunking.chunk_splitter import ChunkSplitter
from embedding.services.chunking.chunk_selector import ChunkSelector
from embedding.services.chunking.chunk_scorer import ChunkScorer
from embedding.services.chunking.selector_config import SelectorConfig


def test_chunk_splitter_basic_functionality():
    """测试ChunkSplitter基本功能"""
    splitter = ChunkSplitter(min_sentences=2, max_tokens=100)
    
    # 测试简单文本分割
    doc_text = "这是第一句话。这是第二句话。\n\n这是新段落的第一句。这是新段落的第二句。"
    paragraphs = splitter.split_paragraphs(doc_text)
    
    # 至少会有几个段落
    assert len(paragraphs) >= 1
    # 每个段落都不为空
    for para in paragraphs:
        assert para.strip() != ""


def test_chunk_scorer_basic_functionality():
    """测试ChunkScorer基本功能"""
    config = SelectorConfig()
    scorer = ChunkScorer(stop_words=set(), config=config)
    
    chunks = [
        "这是一段包含关键词的文本",
        "这是另一段文本，可能包含不同的关键词"
    ]
    
    # 初始化全局统计
    scorer.compute_global_statistics(chunks)
    
    # 计算得分
    scores = scorer.compute_scores(chunks, [0, 1], "标题")
    
    # 应该返回每个chunk的得分
    assert len(scores) == len(chunks)
    # 得分应该在合理范围内
    for score in scores:
        assert isinstance(score, (int, float))


def test_chunk_selector_basic():
    """测试ChunkSelector基本功能"""
    # 这里我们只是测试类能否正常初始化，实际功能需要更复杂的设置
    from embedding.infra.embedding_strategies.base import BaseEmbeddingStrategy
    from unittest.mock import Mock
    
    class MockEmbeddingStrategy(BaseEmbeddingStrategy):
        def encode(self, texts, is_query):
            # 返回模拟的嵌入向量
            return [[0.1] * self.experiment.embedding_dim for _ in texts]
    
    # 创建模拟配置
    experiment_cfg = Mock()
    experiment_cfg.embedding_dim = 384
    
    inference_cfg = Mock()
    
    mock_strategy = MockEmbeddingStrategy(experiment=experiment_cfg, inference=inference_cfg)
    
    selector = ChunkSelector(
        embedding_strategy=mock_strategy,
        chunk_num=5
    )
    
    # 验证selector被正确初始化
    assert selector.chunk_num == 5