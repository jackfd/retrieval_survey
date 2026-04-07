import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from time import perf_counter
import numpy as np

from embedding.domain.exceptions import ProcessingError
from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.infra.jsonl_reader import read_objects
from embedding.infra.output_writer import OutputWriter
from embedding.services.chunk_selector import select_from_embeddings
from embedding.services.chunk_splitter import ChunkSplitter

logger = logging.getLogger(__name__)

DOC_WINDOW_SIZE = 3000


def process_doc(
    embedding: EmbeddingStrategy, doc_path: Path, max_length: int, output: OutputWriter
) -> None:
    run_start = perf_counter()
    splitter = ChunkSplitter(max_length=max_length)
    total_selected_chunks = 0
    read_iter = iter(read_objects(doc_path))

    while True:
        docs_window, chunk_candicates = _collect_doc_window(
            read_iter, splitter, DOC_WINDOW_SIZE, doc_path
        )
        if not docs_window:
            break

        total_selected_chunks += _process_doc_window(
            embedding, splitter, docs_window, chunk_candicates, output
        )

    output.close()
    if total_selected_chunks == 0:
        logger.error("No docs chunks found in %s", doc_path)
        raise ProcessingError(f"No documents found in {doc_path}")

    elapsed_sec = perf_counter() - run_start
    logger.info(
        "docs process finished doc_count:%s total_sec:%.3f, avg_ms:%.3f",
        total_selected_chunks,
        elapsed_sec,
        elapsed_sec * 1000 / total_selected_chunks,
    )


def _collect_doc_window(
    read_iter: Iterator[Tuple[int, Dict[str, Any]]],
    splitter: ChunkSplitter,
    doc_window_size: int,
    doc_path: Path,
) -> tuple[List[Dict[str, Any]], List[str]]:
    docs_window: List[Dict[str, Any]] = []
    window_chunk_texts: List[str] = []

    while len(docs_window) < doc_window_size:
        try:
            line_num, obj = next(read_iter)
        except StopIteration:
            break

        doc_id = str(obj.get("doc_id", "")).strip()
        doc_text = _normalize_doc_text(obj.get("doc_text"))
        if not doc_id or not doc_text:
            logger.error(
                "docs_path=%s line_num=%s id=%r, id/text must be non-empty",
                doc_path,
                line_num,
                doc_id,
            )
            raise ProcessingError(
                "Invalid docs input docs_path=%s line_num=%s doc_id=%r"
                % (doc_path, line_num, doc_id)
            )

        try:
            candidate_chunks = splitter.split_to_candidates(doc_text)
        except Exception as exc:
            logger.exception(
                "doc split to chunks failed, path=%s line_num=%s doc_id=%s error_type=%s",
                doc_path,
                line_num,
                doc_id,
                type(exc).__name__,
            )
            raise ProcessingError(
                "doc split to chunks failed, path=%s line_num=%s doc_id=%s"
                % (doc_path, line_num, doc_id)
            ) from exc

        chunk_start = len(window_chunk_texts)
        for chunk in candidate_chunks:
            window_chunk_texts.append(str(chunk["text"]))
        chunk_end = len(window_chunk_texts)

        docs_window.append(
            {
                "doc_id": doc_id,
                "line_num": line_num,
                "candidates": candidate_chunks,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            }
        )
    logger.info(
        f"  1-docs collected, doc_count={len(docs_window)}, chunk_count={len(window_chunk_texts)}"
    )
    return docs_window, window_chunk_texts


def _process_doc_window(
    embedding: EmbeddingStrategy,
    splitter: ChunkSplitter,
    docs_window: List[Dict[str, Any]],
    chunk_candicates: List[str],
    output: OutputWriter,
) -> int:
    if not chunk_candicates:
        return 0
    embed_start = perf_counter()
    start_doc_id = str(docs_window[0]["doc_id"])
    end_doc_id = str(docs_window[-1]["doc_id"])
    chunks_count = len(chunk_candicates)
    chunk_length_stats = _chunk_token_stats(
        chunk_candicates, splitter, splitter.hard_max_tokens
    )
    prepared_length_stats = _describe_input_lengths(
        embedding, chunk_candicates, is_query=False
    )

    try:
        vectors = embedding.encode(chunk_candicates, is_query=False)
    except Exception as exc:
        logger.exception(
            "docs embedding batch failed, start_doc_id=%s end_doc_id=%s chunk_count=%s configured_max_length_tokens=%s max_chunk_estimated_tokens=%s avg_chunk_estimated_tokens=%.1f estimated_chunk_over_limit_count=%s token_stats_available=%s max_prepared_tokens=%s avg_prepared_tokens=%s prepared_over_limit_count=%s error_type=%s",
            start_doc_id,
            end_doc_id,
            chunks_count,
            splitter.hard_max_tokens,
            chunk_length_stats["max_chunk_estimated_tokens"],
            chunk_length_stats["avg_chunk_estimated_tokens"],
            chunk_length_stats["estimated_chunk_over_limit_count"],
            prepared_length_stats["token_stats_available"],
            prepared_length_stats["max_prepared_tokens"],
            prepared_length_stats["avg_prepared_tokens"],
            prepared_length_stats["prepared_over_limit_count"],
            type(exc).__name__,
        )
        raise ProcessingError(
            "docs embedding batch failed start_doc_id=%s end_doc_id=%s chunk_count=%s"
            % (start_doc_id, end_doc_id, chunks_count)
        ) from exc

    if len(vectors) != chunks_count:
        logger.error(
            "docs embedding batch size mismatch start_doc_id=%s end_doc_id=%s expected=%s actual=%s",
            start_doc_id,
            end_doc_id,
            chunks_count,
            len(vectors),
        )
        raise ProcessingError(
            "docs embedding batch size mismatch start_doc_id=%s end_doc_id=%s expected=%s actual=%s"
            % (start_doc_id, end_doc_id, chunks_count, len(vectors))
        )
    elapsed_sec = perf_counter() - embed_start
    logger.info(
        "  2-docs embedding, dim=%s chunks=%s configured_max_length_tokens=%s max_chunk_estimated_tokens=%s avg_chunk_estimated_tokens=%.1f estimated_chunk_over_limit_count=%s token_stats_available=%s max_prepared_tokens=%s avg_prepared_tokens=%s prepared_over_limit_count=%s total_sec:%.3f, avg_ms:%.3f",
        vectors.shape[1],
        chunks_count,
        splitter.hard_max_tokens,
        chunk_length_stats["max_chunk_estimated_tokens"],
        chunk_length_stats["avg_chunk_estimated_tokens"],
        chunk_length_stats["estimated_chunk_over_limit_count"],
        prepared_length_stats["token_stats_available"],
        prepared_length_stats["max_prepared_tokens"],
        prepared_length_stats["avg_prepared_tokens"],
        prepared_length_stats["prepared_over_limit_count"],
        elapsed_sec,
        elapsed_sec * 1000 / chunks_count,
    )
    vector_matrix = np.asarray(vectors, dtype=np.float32)
    total_selected_chunks = 0

    for doc in docs_window:
        chunk_candicates = doc["candidates"]
        if not chunk_candicates:
            continue

        chunk_start = int(doc["chunk_start"])
        chunk_end = int(doc["chunk_end"])
        chunk_vectors = vector_matrix[chunk_start:chunk_end]
        doc_id = doc["doc_id"]
        try:
            selected = select_from_embeddings(doc_id, chunk_candicates, chunk_vectors)
        except Exception as exc:
            logger.exception(
                "docs chunk selection failed doc_id=%s line_num=%s error_type=%s",
                doc_id,
                doc["line_num"],
                type(exc).__name__,
            )
            raise ProcessingError(
                "docs chunk selection failed doc_id=%s line_num=%s"
                % (doc_id, doc["line_num"])
            ) from exc

        if selected:
            output.write_doc_chunks(selected)
            total_selected_chunks += len(selected)

    write_elapsed_sec = perf_counter() - embed_start
    logger.info(
        "  3-doc selected, total chunks:%s total_secs:%.3f",
        total_selected_chunks,
        write_elapsed_sec,
    )
    return total_selected_chunks


def _chunk_token_stats(
    chunks: List[str], splitter: ChunkSplitter, max_length: int
) -> dict[str, int | float]:
    if not chunks:
        return {
            "max_chunk_estimated_tokens": 0,
            "avg_chunk_estimated_tokens": 0.0,
            "estimated_chunk_over_limit_count": 0,
        }

    lengths = [splitter.estimate_tokens(chunk) for chunk in chunks]
    return {
        "max_chunk_estimated_tokens": max(lengths),
        "avg_chunk_estimated_tokens": float(sum(lengths)) / float(len(lengths)),
        "estimated_chunk_over_limit_count": sum(
            1 for length in lengths if length > max_length
        ),
    }


def _describe_input_lengths(
    embedding: EmbeddingStrategy, texts: List[str], *, is_query: bool
) -> dict[str, int | float | bool | None]:
    describe = getattr(embedding, "describe_input_lengths", None)
    if not callable(describe):
        return {
            "token_stats_available": False,
            "max_prepared_tokens": None,
            "avg_prepared_tokens": None,
            "prepared_over_limit_count": None,
        }

    stats = describe(texts, is_query=is_query)
    if not isinstance(stats, dict):
        return {
            "token_stats_available": False,
            "max_prepared_tokens": None,
            "avg_prepared_tokens": None,
            "prepared_over_limit_count": None,
        }

    return {
        "token_stats_available": bool(stats.get("token_stats_available", False)),
        "max_prepared_tokens": stats.get("max_prepared_tokens"),
        "avg_prepared_tokens": stats.get("avg_prepared_tokens"),
        "prepared_over_limit_count": stats.get("prepared_over_limit_count"),
    }


def _normalize_doc_text(value: Any) -> List[str]:
    """Normalize raw doc_text input into an ordered text-fragment sequence.

    This is a service-boundary adapter only. It accepts the currently supported
    input shapes (`str` and `list`) and returns the fragment sequence consumed
    by `ChunkSplitter`. It does not infer or guarantee sentence boundaries.
    """
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, list):
        normalized: List[str] = []
        for item in value:
            fragment = str(item).strip()
            if fragment:
                normalized.append(fragment)
        return normalized

    return []
