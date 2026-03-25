# -*- coding: utf-8 -*-
import logging
import re
from typing import Callable, List


class ChunkSplitter:
    _LIST_ITEM_PATTERN = re.compile(r"^(?:[-*−•·▪‣]|\d+[\.)。、])\s+")
    _SENTENCE_SPLIT_PATTERN = re.compile(r"[。.！!?；;]+")

    def __init__(
        self,
        min_sentences: int = 3,
        max_tokens: int = 8092,
        token_counter: Callable[[str], int] | None = None,
    ):
        self.min_sentences = min_sentences
        self.max_tokens = max_tokens
        self.avg_char_per_token = 4
        self.token_counter = token_counter

    def split_paragraphs(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            logging.getLogger("embed_pipe").error(
                "event=splitter_degrade reason=%s context=%s",
                "Invalid input text for split_paragraphs",
                "text_type=%s" % type(text).__name__,
            )
            return []

        raw_paragraphs = re.split(r"\n{2,}", text)
        chunks = []
        for paragraph in raw_paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if self._count_tokens(paragraph) > self.max_tokens:
                chunks.extend(self._split_large_paragraph(paragraph))
            elif chunks and self._LIST_ITEM_PATTERN.match(paragraph):
                chunks[-1] += " " + paragraph
            elif chunks and self._sentence_count(paragraph) < self.min_sentences:
                if self._LIST_ITEM_PATTERN.match(paragraph):
                    chunks[-1] += " " + paragraph
                elif re.search(r"[。.！!?；;]\s*$", paragraph):
                    chunks.append(paragraph)
                else:
                    chunks[-1] += "\n" + paragraph
            else:
                chunks.append(paragraph)
        return chunks

    def _sentence_count(self, text: str) -> int:
        sentences = [segment.strip() for segment in self._SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
        return len(sentences)

    def _count_tokens(self, text: str) -> int:
        if self.token_counter:
            return max(1, int(self.token_counter(text)))

        if not text:
            return 1

        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - cjk_chars
        estimated = cjk_chars + other_chars // self.avg_char_per_token
        return max(1, estimated)

    def _split_large_paragraph(self, text: str) -> List[str]:
        sentences = re.split(r"([。.！!?；;—])", text)
        parts = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence.strip():
                parts.append(sentence.strip())

        if len(sentences) % 2 == 1:
            last_part = sentences[-1]
            if last_part.strip():
                parts.append(last_part.strip())

        chunks = []
        current_chunk = ""
        for sentence in parts:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            if self._count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if self._count_tokens(sentence) > self.max_tokens:
                    chunks.extend(self._split_long_sentence(sentence))
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        chars = list(sentence)
        chunks = []
        current_chunk = ""
        for char in chars:
            test_chunk = current_chunk + char
            if self._count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = char
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
