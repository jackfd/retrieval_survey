"""chunk 选择逻辑单元测试，覆盖 MMR 排序、稳定合并和异常传播。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SelectorConfig:
    def __init__(self, **kwargs):
        self.batch_size = 64
        self.hard_max_tokens = 8092
        self.target_tokens = 3200
        self.min_independent_tokens = 500
        self.top_n = 3
        self.mmr_lambda = 0.7
        self.__dict__.update(kwargs)


def _load_chunk_selector_module():
    base_path = (
        Path(__file__).resolve().parents[2]
        / "infra"
        / "embedding_strategies"
        / "base.py"
    )
    base_spec = importlib.util.spec_from_file_location(
        "embedding.infra.embedding_strategies.base", base_path
    )
    base_module = importlib.util.module_from_spec(base_spec)
    assert base_spec.loader is not None
    base_spec.loader.exec_module(base_module)

    fake_embedding_pkg = types.ModuleType("embedding.infra.embedding_strategies")
    fake_embedding_pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "infra" / "embedding_strategies")
    ]
    fake_embedding_pkg.BaseEmbeddingStrategy = base_module.BaseEmbeddingStrategy
    fake_embedding_pkg.EmbeddingStrategy = base_module.EmbeddingStrategy

    selector_path = Path(__file__).resolve().parents[2] / "services" / "chunk_selector.py"
    selector_spec = importlib.util.spec_from_file_location(
        "embedding.services.chunk_selector", selector_path
    )
    selector_module = importlib.util.module_from_spec(selector_spec)
    assert selector_spec.loader is not None

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "embedding.infra.embedding_strategies", fake_embedding_pkg)
        mp.setitem(sys.modules, "embedding.infra.embedding_strategies.base", base_module)
        selector_spec.loader.exec_module(selector_module)

    return selector_module


def test_chunk_selector_run_returns_ranked_top3_with_normalized_vectors():
    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(
        top_n=3,
        min_independent_tokens=1,
        target_tokens=100,
    )

    selected = selector.run(
        "第一段内容。\n\n第二段内容。\n\n第三段内容。\n\n第四段内容。",
        "doc123",
    )

    assert [item["chunk_rank"] for item in selected] == [1, 2, 3]
    assert [item["chunk_id"] for item in selected] == [
        "doc123#c002",
        "doc123#c003",
        "doc123#c001",
    ]
    assert all(
        np.isclose(np.linalg.norm(np.asarray(item["chunk_vector"], dtype=float)), 1.0)
        for item in selected
    )
    mock_strategy.encode.assert_called_once_with(
        ["第一段内容。", "第二段内容。", "第三段内容。", "第四段内容。"],
        is_query=False,
    )


def test_chunk_selector_merges_small_chunk_with_better_neighbor():
    mock_strategy = Mock()
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(target_tokens=15, min_independent_tokens=5)

    merged = selector._merge_small_chunks(
        [
            "甲甲甲甲甲甲甲甲甲甲",
            "乙乙",
            "丙丙丙丙丙丙丙丙丙丙丙丙丙丙",
        ]
    )

    assert merged == ["甲甲甲甲甲甲甲甲甲甲\n\n乙乙", "丙丙丙丙丙丙丙丙丙丙丙丙丙丙"]


def test_chunk_selector_returns_all_candidates_when_count_not_exceeding_top_n():
    mock_strategy = Mock()
    mock_strategy.encode.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(
        top_n=3,
        min_independent_tokens=1,
        target_tokens=100,
    )

    selected = selector.run("第一段。\n\n第二段。", "doc1")

    assert len(selected) == 2
    assert [item["chunk_rank"] for item in selected] == [1, 2]
    assert {item["chunk_id"] for item in selected} == {"doc1#c001", "doc1#c002"}


def test_chunk_selector_propagates_embedding_strategy_error():
    chunk_selector_module = _load_chunk_selector_module()
    mock_strategy = Mock()
    mock_strategy.encode.side_effect = RuntimeError("boom")
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=mock_strategy)
    selector.config = SelectorConfig(min_independent_tokens=1, target_tokens=100)

    with pytest.raises(RuntimeError):
        selector.run("主题部分。主题部分。主题部分。", "doc1")


def test_split_sentences_basic():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = "这是第一句。这是第二句。这是第三句。"
    result = selector._split_sentences(text)

    assert result == ["这是第一句。", "这是第二句。", "这是第三句。"]


def test_split_sentences_with_different_punctuation():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = "这是第一句。这是第二句!这是第三句?这是第四句."
    result = selector._split_sentences(text)

    assert result == ["这是第一句。", "这是第二句!", "这是第三句?", "这是第四句."]


def test_split_sentences_with_english_punctuation():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = "This is sentence one. This is sentence two! This is sentence three?"
    result = selector._split_sentences(text)

    assert result == [
        "This is sentence one.",
        "This is sentence two!",
        "This is sentence three?",
    ]


def test_split_sentences_with_semicolon_and_newlines():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = "这是第一句;这是第二句。\n这是第三句。\n\n这是第四句。"
    result = selector._split_sentences(text)

    # 分号默认作为软边界，不在句子级切分时断开
    assert result == ["这是第一句;这是第二句。", "这是第三句。", "这是第四句。"]


def test_split_block_uses_semicolon_as_soft_boundary_when_over_target():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())
    selector.config = SelectorConfig(target_tokens=8, hard_max_tokens=100)

    text = "甲甲甲甲甲甲甲甲甲甲；乙乙乙乙乙乙乙乙乙乙。"
    chunks = selector._split_block(text)

    assert chunks == ["甲甲甲甲甲甲甲甲甲甲；", "乙乙乙乙乙乙乙乙乙乙。"]


def test_split_sentences_empty_and_whitespace():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = "  \n  \n  "
    result = selector._split_sentences(text)

    assert result == []


def test_split_sentences_no_final_punctuation():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = "这是第一句。这是第二句"
    result = selector._split_sentences(text)

    # Should include the last sentence even without ending punctuation
    assert result == ["这是第一句。", "这是第二句"]


def test_split_sentences_english_long_text():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = """
    Natural language processing is a fascinating field. It combines linguistics, computer science, and artificial intelligence.
    The applications are widespread and diverse today! Researchers work on improving machine translation systems constantly.
    Text summarization is another exciting area of study; it helps to reduce large documents to essential summaries.
    Many modern systems use deep learning techniques, which have shown remarkable results in various tasks.
    """
    result = selector._split_sentences(text)

    # Filter out empty strings if any
    result = [s for s in result if s.strip()]

    # 分号不作为硬边界，句子数应少于旧规则
    assert len(result) == 6

    # Check that each sentence ends with proper punctuation
    for sent in result:
        # Remove leading/trailing whitespace and check that it ends with punctuation
        stripped = sent.strip()
        if stripped:  # Only check non-empty sentences
            assert stripped.endswith((".", "!", "?"))


def test_split_sentences_english_with_numbers_and_periods():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    text = "The cost was $19.99. This is another sentence! Chapter 3.1 discusses this topic? Yes indeed."
    result = selector._split_sentences(text)
    print(result)
    # The regex splits on punctuation, so periods in numbers like $19.99 are still sentence terminators
    expected = [
        "The cost was $19.99.",
        "This is another sentence!",
        "Chapter 3.1 discusses this topic?",
        "Yes indeed.",
    ]
    assert result == expected


def test_split_sentences_english_complex_sentences():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    # Complex English sentences with abbreviations and decimal numbers
    text = "Dr. Smith went to the U.S.A. last week. He bought 2.5 kg of apples for $3.99! What did he do next? He continued his research; the experiment was ongoing."
    result = selector._split_sentences(text)
    print(result)
    # Due to the regex pattern, every punctuation mark (including periods in abbreviations) acts as a sentence boundary
    expected = [
        "Dr. Smith went to the U.S.A. last week.",
        "He bought 2.5 kg of apples for $3.99!",
        "What did he do next?",
        "He continued his research; the experiment was ongoing.",
    ]

    assert result == expected


def test_split_sentences_english_mixed_punctuation():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    # Testing various punctuation marks together
    text = "Hello world. How are you today? I am fine! Well, maybe; it depends on context: what do you think? That's interesting..."
    result = selector._split_sentences(text)
    expected = [
        "Hello world.",
        "How are you today?",
        "I am fine!",
        "Well, maybe; it depends on context: what do you think?",
        "That's interesting...",
    ]

    assert result == expected


def test_split_sentences_english_with_newlines():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    # Text with embedded newlines
    text = "First sentence ends here.\nSecond sentence!\n\nThird sentence?\nFourth; with semicolon."
    result = selector._split_sentences(text)

    expected = [
        "First sentence ends here.",
        "Second sentence!",
        "Third sentence?",
        "Fourth; with semicolon.",
    ]

    assert result == expected


def test_split_sentences_very_long_english_text():
    chunk_selector_module = _load_chunk_selector_module()
    selector = chunk_selector_module.ChunkSelector(embedding_strategy=Mock())

    # A very long English text with many sentences
    text = """
    Machine learning is a branch of artificial intelligence. It focuses on algorithms that learn from data.
    Deep learning is a subset of machine learning; it uses neural networks with multiple layers.
    Supervised learning requires labeled training data. Unsupervised learning finds patterns in unlabeled data.
    Reinforcement learning uses rewards and penalties to guide learning. Natural language processing is an application area.
    Computer vision is another major application of machine learning! It enables machines to interpret visual information.
    Recommendation systems are widely used in online platforms today? They help users discover relevant content.
    Clustering algorithms group similar data points together; k-means is a popular approach.
    Decision trees are interpretable models for classification tasks. Random forests combine multiple decision trees.
    Support vector machines find optimal boundaries between classes. Neural networks are inspired by biological brains.
    Convolutional neural networks excel at image recognition tasks. Recurrent neural networks handle sequential data well.
    Transformers have revolutionized natural language processing recently! BERT and GPT are famous examples.
    Training models requires significant computational resources sometimes? GPUs accelerate this process significantly.
    Overfitting occurs when models memorize training data instead of learning general patterns. Regularization prevents this.
    Cross-validation estimates how well models generalize to unseen data. Evaluation metrics measure model performance.
    Feature engineering transforms raw data into meaningful inputs for models. Deep learning reduces need for manual features.
    """

    result = selector._split_sentences(text)
    result = [
        s.strip() for s in result if s.strip()
    ]  # Clean up and remove empty strings

    # 分号改为软边界后，句子数会低于旧规则
    assert len(result) >= 20

    # Each sentence should end with appropriate punctuation
    for sentence in result:
        # Sentence should end with one of the punctuation marks we look for
        assert sentence.endswith(
            (".", "!", "?")
        ), f"Sentence does not end with proper punctuation: '{sentence}'"
