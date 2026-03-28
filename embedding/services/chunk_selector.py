# -*- coding: utf-8 -*-
import logging
import re
from typing import Dict, List, Sequence
from dataclasses import dataclass

import numpy as np

from embedding.infra.embedding_strategies import EmbeddingStrategy


@dataclass
class SelectorConfig:
    hard_max_tokens: int = 8092
    target_tokens: int = 3200
    min_independent_tokens: int = 500
    top_n: int = 3
    mmr_lambda: float = 0.7


class ChunkSelector:
    _LIST_ITEM_PATTERN = re.compile(r"^(?:[-*−•·▪‣]|\d+[\.)。、])\s+")
    _SENTENCE_PATTERN = re.compile(r"[^。.！!?；;\n]+[。.！!?；;]?")

    def __init__(self, embedding_strategy: EmbeddingStrategy):
        self.embedding_strategy = embedding_strategy
        self.config = SelectorConfig()
        self.avg_char_per_token = 4
        self.logger = logging.getLogger(__name__)

    def run(self, text: str, doc_id: str) -> List[Dict]:
        candidates = self._split_to_candidates(text)
        embeddings = self._embed_chunks(candidates)
        centroid = self._compute_centroid(embeddings)
        rep_scores = embeddings @ centroid
        selected_indices, selected_scores = self._select_top_n(embeddings, rep_scores)
        return self._build_records(
            doc_id=doc_id,
            candidates=candidates,
            embeddings=embeddings,
            selected_indices=selected_indices,
            selected_scores=selected_scores,
        )

    def _split_to_candidates(self, text: str) -> List[Dict[str, object]]:
        """
        将输入文本分割成候选块列表
        """
        # 按照两个或多个换行符分割文本，并去除空白段落
        paragraphs = [
            part.strip() for part in re.split(r"\n{2,}", text) if part.strip()
        ]

        # 合并连续的列表项到同一个块中
        blocks: List[str] = []
        for paragraph in paragraphs:
            if blocks and self._LIST_ITEM_PATTERN.match(paragraph):
                blocks[-1] += " " + paragraph
                continue
            blocks.append(paragraph)

        # 对每个块进行进一步分割
        split_chunks: List[str] = []
        for block in blocks:
            split_chunks.extend(self._split_block(block))

        # 合并过小的块
        merged_chunks = self._merge_small_chunks(split_chunks)
        return [
            {"order": index + 1, "text": chunk}
            for index, chunk in enumerate(merged_chunks)
        ]

    def _split_block(self, text: str) -> List[str]:
        """
        将文本分割成多个块，确保每个块不超过目标token数
        """
        # 计算输入文本的token数量
        token_count = self._count_tokens(text)
        if token_count <= self.config.target_tokens:
            return [text]

        # 按句子分割文本
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            # 如果只有一个句子，则使用长文本分割方法
            return self._split_long_text(text, self.config.hard_max_tokens)

        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果单个句子超过最大限制，则单独处理该句子
            if self._count_tokens(sentence) > self.config.hard_max_tokens:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    self._split_long_text(sentence, self.config.hard_max_tokens)
                )
                continue

            # 尝试将当前句子与现有文本合并
            candidate = sentence if not current else current + " " + sentence
            if self._count_tokens(candidate) <= self.config.target_tokens:
                current = candidate
                continue

            # 如果合并后超出目标长度，则保存当前文本并开始新的合并尝试
            if current:
                chunks.append(current)
            current = sentence

        if current:
            chunks.append(current)

        # 最终检查：确保所有块都不超过硬性最大token限制
        final_chunks: List[str] = []
        for chunk in chunks:
            if self._count_tokens(chunk) > self.config.hard_max_tokens:
                final_chunks.extend(
                    self._split_long_text(chunk, self.config.hard_max_tokens)
                )
            else:
                final_chunks.append(chunk)
        return final_chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentences = [
            match.group(0).strip() for match in self._SENTENCE_PATTERN.finditer(text)
        ]
        return [sentence for sentence in sentences if sentence]

    def _split_long_text(self, text: str, token_limit: int) -> List[str]:
        """将 text 按指定的 token_limit 分割成多个部分"""
        if self._count_tokens(text) <= token_limit:
            return [text]

        # 按标点符号和空格分割文本
        parts = [
            part.strip() for part in re.split(r"([,，:：、\s]+)", text) if part.strip()
        ]
        if len(parts) <= 1:
            return self._split_by_chars(text, token_limit)

        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else current + part
            if self._count_tokens(candidate) <= token_limit:
                current = candidate
                continue

            # 当前候选超出token限制，将current加入结果并处理part
            if current:
                chunks.append(current.strip())
            if self._count_tokens(part) > token_limit:
                # 单个部分就超出token限制，使用字符分割方法进一步拆分
                chunks.extend(self._split_by_chars(part, token_limit))
                current = ""
            else:
                current = part

        if current:
            chunks.append(current.strip())
        return chunks

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

    def _merge_small_chunks(self, chunks: Sequence[str]) -> List[str]:
        merged = [chunk for chunk in chunks if chunk.strip()]
        if not merged:
            return []

        while len(merged) > 1:
            small_index = next(
                (
                    index
                    for index, chunk in enumerate(merged)
                    if self._count_tokens(chunk) < self.config.min_independent_tokens
                ),
                None,
            )
            if small_index is None:
                break

            neighbor_index = self._choose_merge_neighbor(merged, small_index)
            if neighbor_index < small_index:
                merged[neighbor_index] = (
                    merged[neighbor_index].rstrip()
                    + "\n\n"
                    + merged[small_index].lstrip()
                )
                del merged[small_index]
            else:
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
            prev_tokens = self._count_tokens(chunks[index - 1] + "\n\n" + chunks[index])
            prev_priority = 0 if prev_tokens <= self.config.target_tokens else 1
            prev_distance = abs(self.config.target_tokens - prev_tokens)
            options.append((prev_priority, prev_distance, prev_tokens, index - 1))

        if index + 1 < len(chunks):
            next_tokens = self._count_tokens(chunks[index] + "\n\n" + chunks[index + 1])
            next_priority = 0 if next_tokens <= self.config.target_tokens else 1
            next_distance = abs(self.config.target_tokens - next_tokens)
            options.append((next_priority, next_distance, next_tokens, index + 1))

        return min(options, key=lambda item: (item[0], item[1], item[2], item[3]))[3]

    def _embed_chunks(self, candidates: Sequence[Dict[str, object]]) -> np.ndarray:
        texts = [str(candidate["text"]) for candidate in candidates]
        embeddings = np.asarray(
            self.embedding_strategy.encode(texts, is_query=False),
            dtype=np.float32,
        )
        if embeddings.ndim != 2:
            raise ValueError("Chunk embeddings must be a 2D array")
        return embeddings

    def _compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        centroid = np.mean(embeddings, axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm == 0.0:
            return centroid
        return centroid / norm

    def _select_top_n(
        self, embeddings: np.ndarray, rep_scores: np.ndarray
    ) -> tuple[List[int], List[float]]:
        candidate_count = embeddings.shape[0]
        select_count = min(candidate_count, self.config.top_n)
        selected_indices: List[int] = []
        selected_scores: List[float] = []

        while len(selected_indices) < select_count:
            best_index = -1
            best_score = float("-inf")
            for index in range(candidate_count):
                if index in selected_indices:
                    continue

                if not selected_indices:
                    score = float(rep_scores[index])
                else:
                    max_sim_to_selected = max(
                        float(embeddings[index] @ embeddings[selected_index])
                        for selected_index in selected_indices
                    )
                    score = float(
                        self.config.mmr_lambda * rep_scores[index]
                        - (1.0 - self.config.mmr_lambda) * max_sim_to_selected
                    )

                if self._is_better_candidate(
                    score=score,
                    index=index,
                    best_score=best_score,
                    best_index=best_index,
                ):
                    best_score = score
                    best_index = index

            selected_indices.append(best_index)
            selected_scores.append(best_score)

        return selected_indices, selected_scores

    def _is_better_candidate(
        self, score: float, index: int, best_score: float, best_index: int
    ) -> bool:
        if best_index < 0:
            return True
        if score > best_score and not np.isclose(score, best_score):
            return True
        if np.isclose(score, best_score):
            return index < best_index
        return False

    def _build_records(
        self,
        *,
        doc_id: str,
        candidates: Sequence[Dict[str, object]],
        embeddings: np.ndarray,
        selected_indices: Sequence[int],
        selected_scores: Sequence[float],
    ) -> List[Dict]:
        width = max(3, len(str(len(candidates))))
        records: List[Dict] = []
        for rank, (candidate_index, score) in enumerate(
            zip(selected_indices, selected_scores), start=1
        ):
            candidate = candidates[candidate_index]
            order = int(candidate["order"])
            records.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}#c{order:0{width}d}",
                    "chunk_text": str(candidate["text"]),
                    "chunk_vector": embeddings[candidate_index].tolist(),
                    "chunk_score": float(score),
                    "chunk_rank": rank,
                }
            )
        return records

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 1
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - cjk_chars
        return max(1, cjk_chars + other_chars // self.avg_char_per_token)
