"""ChunkSplitter unit tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


def _load_chunk_splitter_module(monkeypatch: pytest.MonkeyPatch, offsets_fn=None):
    if offsets_fn is not None:
        fake_blingfire = types.ModuleType("blingfire")
        fake_blingfire.text_to_sentences_and_offsets = offsets_fn
        monkeypatch.setitem(sys.modules, "blingfire", fake_blingfire)

    splitter_path = Path(__file__).resolve().parents[2] / "services" / "chunk_splitter.py"
    splitter_spec = importlib.util.spec_from_file_location("chunk_splitter", splitter_path)
    splitter_module = importlib.util.module_from_spec(splitter_spec)
    assert splitter_spec.loader is not None
    splitter_spec.loader.exec_module(splitter_module)
    return splitter_module


def test_split_to_candidates_merges_consecutive_list_items(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()
    splitter.target_tokens = 10_000
    splitter.min_independent_tokens = 1

    text = "概述段落。\n\n1. 第一项内容。\n\n2. 第二项内容。\n\n收尾段落。"
    result = splitter.split_to_candidates(text)

    assert result == [
        {"order": 1, "text": "概述段落。 1. 第一项内容。 2. 第二项内容。"},
        {"order": 2, "text": "收尾段落。"},
    ]


def test_split_sentences_uses_offsets_and_strips_whitespace(monkeypatch: pytest.MonkeyPatch):
    def fake_offsets(_text: str):
        return "", [(0, 8), (8, 17)]

    splitter_module = _load_chunk_splitter_module(monkeypatch, offsets_fn=fake_offsets)
    splitter = splitter_module.ChunkSplitter()

    text = "  First.  Second?  "
    result = splitter._split_sentences(text)

    assert result == ["First.", "Second?"]


def test_split_long_text_falls_back_to_char_split_when_unsplittable(
    monkeypatch: pytest.MonkeyPatch,
):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()
    splitter.avg_char_per_token = 1

    text = "abcdefghij"
    chunks = splitter._split_long_text(text, token_limit=3)

    assert "".join(chunks) == text
    assert all(len(chunk) <= 3 for chunk in chunks)


def test_merge_small_chunks_prefers_neighbor_within_target(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()
    splitter.target_tokens = 10
    splitter.min_independent_tokens = 5

    chunks = ["甲" * 9, "乙" * 2, "丙" * 6]
    merged = splitter._merge_small_chunks(chunks)

    assert merged == ["甲" * 9, "乙" * 2 + "\n\n" + "丙" * 6]


def test_split_sentences_english_long_text(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()

    text = """
    Natural language processing is a fascinating field. It combines linguistics, computer science, and artificial intelligence.
    The applications are widespread and diverse today! Researchers work on improving machine translation systems constantly.
    Text summarization is another exciting area of study; it helps to reduce large documents to essential summaries.
    Many modern systems use deep learning techniques, which have shown remarkable results in various tasks.
    """
    result = [sentence.strip() for sentence in splitter._split_sentences(text) if sentence.strip()]

    assert len(result) >= 5
    assert result[0].startswith("Natural language processing is a fascinating field")
    assert result[-1].endswith("various tasks.")
    assert all(sentence.endswith((".", "!", "?")) for sentence in result)


def test_split_sentences_english_with_numbers_and_periods(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()

    text = "The cost was $19.99. This is another sentence! Chapter 3.1 discusses this topic? Yes indeed."
    result = splitter._split_sentences(text)
    expected = [
        "The cost was $19.99.",
        "This is another sentence!",
        "Chapter 3.1 discusses this topic?",
        "Yes indeed.",
    ]
    assert result == expected


def test_split_sentences_english_complex_sentences(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()

    text = "Dr. Smith went to the U.S.A. last week. He bought 2.5 kg of apples for $3.99! What did he do next? He continued his research; the experiment was ongoing."
    result = splitter._split_sentences(text)
    expected = [
        "Dr. Smith went to the U.S.A. last week.",
        "He bought 2.5 kg of apples for $3.99!",
        "What did he do next?",
        "He continued his research; the experiment was ongoing.",
    ]
    assert result == expected


def test_split_sentences_english_mixed_punctuation(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()

    text = "Hello world. How are you today? I am fine! Well, maybe; it depends on context: what do you think? That's interesting..."
    result = splitter._split_sentences(text)
    expected = [
        "Hello world.",
        "How are you today?",
        "I am fine!",
        "Well, maybe; it depends on context: what do you think?",
        "That's interesting...",
    ]
    assert result == expected


def test_split_sentences_english_with_newlines(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()

    text = "First sentence ends here.\nSecond sentence!\n\nThird sentence?\nFourth; with semicolon."
    result = splitter._split_sentences(text)
    expected = [
        "First sentence ends here.",
        "Second sentence!",
        "Third sentence?",
        "Fourth; with semicolon.",
    ]
    assert result == expected


def test_split_sentences_very_long_english_text(monkeypatch: pytest.MonkeyPatch):
    splitter_module = _load_chunk_splitter_module(monkeypatch)
    splitter = splitter_module.ChunkSplitter()

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
    result = [sentence.strip() for sentence in splitter._split_sentences(text) if sentence.strip()]

    assert len(result) >= 20
    assert result[0].startswith("Machine learning is a branch of artificial intelligence")
    assert result[-1].endswith("manual features.")
    assert all(sentence.endswith((".", "!", "?")) for sentence in result)
