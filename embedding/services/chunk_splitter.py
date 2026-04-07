# -*- coding: utf-8 -*-
import logging
import re
from collections.abc import Sequence
from typing import Dict, List

from blingfire import text_to_sentences_and_offsets

logger = logging.getLogger(__name__)


class ChunkSplitter:
    """公开数据集候选 chunk 构建器，按内容统一切分输入文本片段。"""

    _LIST_ITEM_PATTERN = re.compile(
        r"^(?:[-*•]|(?:\(?\d+\)?|[A-Z]|[IVXivx]+)[\.\):])\s+"
    )

    def __init__(self, max_length: int = 8092):
        self.hard_max_tokens: int = max_length
        self.target_tokens: int = min(
            self.hard_max_tokens,
            min(1200, max(400, int(self.hard_max_tokens * 0.5))),
        )
        self.avg_char_per_token = 4

    def split_to_candidates(self, text: Sequence[str]) -> List[Dict[str, object]]:
        """
        将有序文本片段序列切分为候选块。

        输入可以是单个原始文本块，也可以是上游提供的逻辑文本片段序列；
        不假设多元素输入已经完成句切。
        """
        if not isinstance(text, Sequence) or isinstance(text, (str, bytes)):
            raise TypeError("text must be sequence[str]")

        blocks = self._normalize_blocks(text)
        if not blocks:
            return []

        chunks: List[str] = []
        for block in blocks:
            chunks.extend(self._split_block(block))

        self._validate_chunk_limits(chunks)
        return [
            {"order": index + 1, "text": chunk} for index, chunk in enumerate(chunks)
        ]

    def _normalize_blocks(self, text: Sequence[str]) -> List[str]:
        fragments = [s for item in text if (s := str(item).strip())]
        if not fragments:
            return []

        raw_blocks = (
            [part.strip() for part in re.split(r"\n{2,}", fragments[0]) if part.strip()]
            if len(fragments) == 1
            else fragments
        )

        blocks: List[str] = []
        for block in raw_blocks:
            if blocks and self._LIST_ITEM_PATTERN.match(block):
                blocks[-1] += " " + block
                continue
            blocks.append(block)
        return blocks

    def _validate_chunk_limits(self, chunks: Sequence[str]) -> None:
        for index, chunk in enumerate(chunks, start=1):
            token_count = self._count_tokens(chunk)
            if token_count > self.hard_max_tokens:
                raise ValueError(
                    "chunk exceeds max_length order=%s token_count=%s max_length=%s"
                    % (index, token_count, self.hard_max_tokens)
                )

    def _split_block(self, text: str) -> List[str]:
        if self._count_tokens(text) <= self.target_tokens:
            return [text]

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._split_oversize_fragment(text)

        return self._pack_fragments_with_limit(sentences, self.target_tokens)

    def _pack_fragments_with_limit(
        self, fragments: Sequence[str], token_limit: int
    ) -> List[str]:
        chunks: List[str] = []
        current = ""

        for fragment in fragments:
            if not fragment:
                continue

            fragment_tokens = self._count_tokens(fragment)
            if fragment_tokens > self.hard_max_tokens:
                logger.warning("too long fragment, fragment tokens:%s", fragment_tokens)
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_oversize_fragment(fragment))
                continue

            candidate = fragment if not current else current + " " + fragment
            if self._count_tokens(candidate) <= token_limit:
                current = candidate
                continue

            if current:
                chunks.append(current)
            current = fragment

        if current:
            chunks.append(current)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        stripped = text.strip()
        if not stripped:
            return []

        _, offsets = text_to_sentences_and_offsets(text)
        if not offsets:
            return [stripped]

        sentences = [text[start:end].strip() for start, end in offsets]
        return [sentence for sentence in sentences if sentence]

    def _split_oversize_fragment(self, text: str) -> List[str]:
        text_tokens = self._count_tokens(text)
        if text_tokens <= self.hard_max_tokens:
            return [text]

        logger.warning(
            "fallback to char split, text_tokens:%s token_limit:%s",
            text_tokens,
            self.hard_max_tokens,
        )
        return self._split_by_chars(text, self.hard_max_tokens)

    def _split_by_chars(self, text: str, token_limit: int) -> List[str]:
        chunks: List[str] = []
        current = ""
        for char in text:
            candidate = current + char
            if current and self._count_tokens(candidate) > token_limit:
                chunks.append(current)
                current = char
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 1
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - cjk_chars
        return max(1, cjk_chars + other_chars // self.avg_char_per_token)
