# -*- coding: utf-8 -*-
import logging
import re
from typing import Dict, List, Sequence
from dataclasses import dataclass
import numpy as np
from blingfire import text_to_sentences_and_offsets

from embedding.infra.embedding_strategies import EmbeddingStrategy

logger = logging.getLogger(__name__)


@dataclass
class SelectorConfig:
    hard_max_tokens: int = 8092
    target_tokens: int = 3200
    min_independent_tokens: int = 500
    top_n: int = 3
    mmr_lambda: float = 0.7


class ChunkSelector:
    _LIST_ITEM_PATTERN = re.compile(r"^(?:[-*−•·▪‣]|\d+[\.)。、])\s+")

    def __init__(self, embedding_strategy: EmbeddingStrategy):
        self.embedding_strategy = embedding_strategy
        self.config = SelectorConfig()
        self.avg_char_per_token = 4

    def run(self, text: str, doc_id: str) -> List[Dict]:
        candidates = self._split_to_candidates(text)
        embeddings = self._embed_chunks(candidates)
        centroid = self._compute_centroid(embeddings)
        rep_scores = self._compute_rep_scores(embeddings, centroid)
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
        将文本块拆分为较小的子块，确保每个子块不超过目标token数。

        首先检查整个文本的 token 数是否已低于目标值，如果超过则按句子拆分。
        然后尝试将句子组合成不大于目标 toke n数的块。
        对于超过硬性限制的单个句子，则使用更细粒度的方法将其拆分。
        """
        token_count = self._count_tokens(text)
        if token_count <= self.config.target_tokens:
            return [text]

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._split_long_text(text, self.config.hard_max_tokens)

        # 按句子拆分并将它们组合成合适的块
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果单个句子超出硬性token限制，则单独处理
            st_tokens = self._count_tokens(sentence)
            if st_tokens > self.config.hard_max_tokens:
                logger.warning("too long sentence, sentence tokens:%s", st_tokens)
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    self._split_long_text(sentence, self.config.hard_max_tokens)
                )
                continue

            # 尝试将当前句子与前一个句子合并
            candidate = sentence if not current else current + " " + sentence
            if self._count_tokens(candidate) <= self.config.target_tokens:
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
            if self._count_tokens(chunk) > self.config.hard_max_tokens:
                final_chunks.extend(
                    self._split_long_text(chunk, self.config.hard_max_tokens)
                )
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
        text_tokens = self._count_tokens(text)
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
            if self._count_tokens(candidate) <= token_limit:
                current = candidate
                continue

            # 当前合并文本超出限制时，将current添加到结果列表
            if current:
                chunks.append(current.strip())
            # 如果单个part就超过了token限制，则进一步分割它
            if self._count_tokens(part) > token_limit:
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
            if current and self._count_tokens(candidate) > token_limit:
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
                    if self._count_tokens(chunk) < self.config.min_independent_tokens
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
        return self._normalize_rows(embeddings)

    def _compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        centroid = np.mean(embeddings, axis=0, dtype=np.float32)
        return self._normalize_vector(centroid)

    def _compute_rep_scores(
        self, embeddings: np.ndarray, centroid: np.ndarray
    ) -> np.ndarray:
        return embeddings @ centroid

    def _select_top_n(
        self, embeddings: np.ndarray, rep_scores: np.ndarray
    ) -> tuple[List[int], List[float]]:
        """
        从候选嵌入中选择top-n个最相关且多样性较高的项目

        使用MMR（Maximum Marginal Relevance）算法平衡代表性得分和与已选项目的差异性。
        首先选择代表性得分最高的项目，然后迭代地选择下一个项目，使其在代表性和与已选项目差异性之间达到平衡。

        Args:
            embeddings (np.ndarray): 候选项目的嵌入向量数组，形状为(n_candidates, embedding_dim)
            rep_scores (np.ndarray): 每个候选项目的代表性得分数组，形状为(n_candidates,)
        """
        candidate_count = embeddings.shape[0]
        w = self.config.mmr_lambda
        select_count = min(candidate_count, self.config.top_n)
        selected_indices: List[int] = []
        selected_scores: List[float] = []

        # 迭代选择指定数量的项目
        while len(selected_indices) < select_count:
            best_index = -1
            best_score = float("-inf")

            # 遍历所有候选项目以找到最佳项目
            for index in range(candidate_count):
                if index in selected_indices:
                    continue

                # 对于第一个选择，直接使用代表性得分；对于后续选择，使用MMR公式
                if not selected_indices:
                    score = float(rep_scores[index])
                else:
                    # 当前候选项目与已选项目之间的最大相似度
                    max_sim_to_selected = max(
                        float(embeddings[index] @ embeddings[selected_index])
                        for selected_index in selected_indices
                    )
                    # 使用MMR公式计算综合得分：λ * 代表性得分 - (1-λ) * 最大相似度
                    score = float(
                        w * rep_scores[index] - (1.0 - w) * max_sim_to_selected
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
        total_chunks = len(candidates)
        width = max(3, len(str(total_chunks)))
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
        if total_chunks > self.config.top_n * 3:
            logger.info("doc id:%s has a large chunks size:%s", doc_id, total_chunks)
        return records

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 1
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - cjk_chars
        return max(1, cjk_chars + other_chars // self.avg_char_per_token)

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm
