# -*- coding: utf-8 -*-
import logging
import re
from typing import Dict, List, Sequence
from blingfire import text_to_sentences_and_offsets

logger = logging.getLogger(__name__)


class ChunkSplitter:
    """适合公开数据集的 chunk 切分，正式环境需要结构化的输入"""

    _LIST_ITEM_PATTERN = re.compile(
        r"^(?:[-*•]|(?:\(?\d+\)?|[A-Z]|[IVXivx]+)[\.\):])\s+"
    )

    def __init__(self, max_length: int = 8092):
        self.hard_max_tokens: int = max_length
        self.target_tokens: int = min(self.hard_max_tokens, 600)
        self.min_independent_tokens: int = 200
        self.avg_char_per_token = 4

    def split_to_candidates(self, text: List[str]) -> List[Dict[str, object]]:
        """
        将输入文本分割成候选块列表
        """
        sentences = [str(item).strip() for item in text if str(item).strip()]
        if not sentences:
            return []

        if len(sentences) == 1:
            split_chunks = self._split_text_flow(sentences[0])
        else:
            split_chunks = self._chunk_from_sentences(sentences)

        # 合并过小的块
        merged_chunks = self._merge_small_chunks(split_chunks)
        return [
            {"order": index + 1, "text": chunk}
            for index, chunk in enumerate(merged_chunks)
        ]

    def _split_text_flow(self, text: str) -> List[str]:
        paragraphs = [
            part.strip() for part in re.split(r"\n{2,}", text) if part.strip()
        ]

        blocks: List[str] = []
        for paragraph in paragraphs:
            if blocks and self._LIST_ITEM_PATTERN.match(paragraph):
                blocks[-1] += " " + paragraph
                continue
            blocks.append(paragraph)

        split_chunks: List[str] = []
        for block in blocks:
            split_chunks.extend(self._split_block(block))
        return split_chunks

    def _split_block(self, text: str) -> List[str]:
        token_count = self.estimate_tokens(text)
        if token_count <= self.target_tokens:
            return [text]

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._split_long_text(text, self.hard_max_tokens)
        return self._chunk_from_sentences(sentences)

    def _chunk_from_sentences(self, sentences: Sequence[str]) -> List[str]:
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            # 如果单个句子超出硬性token限制，则单独处理
            st_tokens = self.estimate_tokens(sentence)
            if st_tokens > self.hard_max_tokens:
                logger.warning("too long sentence, sentence tokens:%s", st_tokens)
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long_text(sentence, self.hard_max_tokens))
                continue

            # 尝试将当前句子与前一个句子合并
            candidate = sentence if not current else current + " " + sentence
            if self.estimate_tokens(candidate) <= self.target_tokens:
                current = candidate
                continue

            # 如果合并后的文本超出了目标token数，则将当前文本块加入列表
            if current:
                chunks.append(current)
            current = sentence

        if current:
            chunks.append(current)

        # 最终检查，确保没有任何块超过硬性最大token限制
        final_chunks: List[str] = []
        for chunk in chunks:
            if self.estimate_tokens(chunk) > self.hard_max_tokens:
                final_chunks.extend(self._split_long_text(chunk, self.hard_max_tokens))
            else:
                final_chunks.append(chunk)
        return final_chunks

    def _split_sentences(self, text: str) -> List[str]:
        stripped = text.strip()
        if not stripped:
            return []

        _, offsets = text_to_sentences_and_offsets(text)
        if not offsets:
            return [stripped]

        sentences = [text[start:end].strip() for start, end in offsets]
        return [sentence for sentence in sentences if sentence]

    def _split_long_text(self, text: str, token_limit: int) -> List[str]:
        """将 text 按指定的 token_limit 分割成多个部分"""
        text_tokens = self.estimate_tokens(text)
        if text_tokens <= token_limit:
            return [text]

        # 按照标点符号和空格分割文本
        parts = [
            part.strip() for part in re.split(r"([,，:：、\s]+)", text) if part.strip()
        ]
        # 如果无法按标点符号分割，则使用字符分割方法
        if len(parts) <= 1:
            logger.warning(
                "Sentence splitting by punctuation failed, text_tokens:%s token_limit:%s",
                text_tokens,
                token_limit,
            )
            return self._split_by_chars(text, token_limit)

        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + part
            # 如果合并后的文本token数未超过限制，则更新current
            if self.estimate_tokens(candidate) <= token_limit:
                current = candidate
                continue

            # 当前合并文本超出限制时，将current添加到结果列表
            if current:
                chunks.append(current.strip())
            # 如果单个part就超过了token限制，则进一步分割它
            if self.estimate_tokens(part) > token_limit:
                chunks.extend(self._split_by_chars(part, token_limit))
                current = ""
            else:
                # 否则将part作为新的current
                current = part

        # 添加最后的current文本
        if current:
            chunks.append(current.strip())
        return chunks

    def _split_by_chars(self, text: str, token_limit: int) -> List[str]:
        chunks: List[str] = []
        current = ""
        for char in text:
            candidate = current + char
            if current and self.estimate_tokens(candidate) > token_limit:
                chunks.append(current)
                current = char
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _merge_small_chunks(self, chunks: Sequence[str]) -> List[str]:
        # 过滤掉空的或只包含空白字符的chunks
        merged = [chunk for chunk in chunks if chunk.strip()]
        if not merged:
            return []

        # 循环处理，直到没有小块或者只剩一个块
        while len(merged) > 1:
            # 查找第一个token数量小于最小独立token数的chunk索引
            small_index = next(
                (
                    index
                    for index, chunk in enumerate(merged)
                    if self.estimate_tokens(chunk) < self.min_independent_tokens
                ),
                None,
            )
            if small_index is None:
                break

            # 选择要合并的邻居chunk
            neighbor_index = self._choose_merge_neighbor(merged, small_index)
            if neighbor_index < small_index:
                # 将较小的chunk与它的邻居合并（邻居在前面）
                merged[neighbor_index] = (
                    merged[neighbor_index].rstrip()
                    + "\n\n"
                    + merged[small_index].lstrip()
                )
                del merged[small_index]
            else:
                # 将较小的chunk与它的邻居合并（较小的chunk在前面）
                merged[small_index] = (
                    merged[small_index].rstrip()
                    + "\n\n"
                    + merged[neighbor_index].lstrip()
                )
                del merged[neighbor_index]

        return merged

    def _choose_merge_neighbor(self, chunks: Sequence[str], index: int) -> int:
        options: List[tuple[int, int, int, int]] = []
        if index > 0:
            prev_tokens = self.estimate_tokens(
                chunks[index - 1] + "\n\n" + chunks[index]
            )
            prev_priority = 0 if prev_tokens <= self.target_tokens else 1
            prev_distance = abs(self.target_tokens - prev_tokens)
            options.append((prev_priority, prev_distance, prev_tokens, index - 1))

        if index + 1 < len(chunks):
            next_tokens = self.estimate_tokens(
                chunks[index] + "\n\n" + chunks[index + 1]
            )
            next_priority = 0 if next_tokens <= self.target_tokens else 1
            next_distance = abs(self.target_tokens - next_tokens)
            options.append((next_priority, next_distance, next_tokens, index + 1))

        return min(options, key=lambda item: (item[0], item[1], item[2], item[3]))[3]

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 1
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - cjk_chars
        return max(1, cjk_chars + other_chars // self.avg_char_per_token)
