import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from time import perf_counter
import numpy as np

from embedding.domain.exceptions import ProcessingError
from embedding.infra.embedding_strategies import EmbeddingStrategy
from embedding.infra.jsonl_reader import read_objects
from embedding.infra.output_writer import OutputWriter
from embedding.services.chunk_selector import build_candidates, select_from_embeddings
from embedding.services.chunk_splitter import ChunkSplitter

logger = logging.getLogger(__name__)

DOC_WINDOW_SIZE = 3000


def process_doc(
    embedding: EmbeddingStrategy, doc_path: Path, output: OutputWriter
) -> None:
    run_start = perf_counter()
    splitter = ChunkSplitter()
    total_selected_chunks = 0
    read_iter = iter(read_objects(doc_path))

    while True:
        docs_window, chunk_candicates = _collect_doc_window(
            read_iter, splitter, DOC_WINDOW_SIZE, doc_path
        )
        if not docs_window:
            break

        total_selected_chunks += _process_doc_window(
            embedding, docs_window, chunk_candicates, output
        )

    output.close()
    if total_selected_chunks == 0:
        logger.error("No document chunks found in %s", doc_path)
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
            candidate_chunks = build_candidates(doc_text, splitter)
        except Exception as exc:
            logger.exception(
                "Document candidate preparation failed docs_path=%s line_num=%s doc_id=%s error_type=%s",
                doc_path,
                line_num,
                doc_id,
                type(exc).__name__,
            )
            raise ProcessingError(
                "Document candidate preparation failed docs_path=%s line_num=%s doc_id=%s"
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
        f"Document window collected, doc_count={len(docs_window)}, chunk_count={len(window_chunk_texts)}"
    )
    return docs_window, window_chunk_texts


def _process_doc_window(
    embedding: EmbeddingStrategy,
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

    try:
        vectors = embedding.encode(chunk_candicates, is_query=False)
    except Exception as exc:
        logger.exception(
            "Document embedding batch failed start_doc_id=%s end_doc_id=%s chunk_count=%s error_type=%s",
            start_doc_id,
            end_doc_id,
            chunks_count,
            type(exc).__name__,
        )
        raise ProcessingError(
            "Document embedding batch failed start_doc_id=%s end_doc_id=%s chunk_count=%s"
            % (start_doc_id, end_doc_id, chunks_count)
        ) from exc

    if len(vectors) != chunks_count:
        logger.error(
            "Document embedding batch size mismatch start_doc_id=%s end_doc_id=%s expected=%s actual=%s",
            start_doc_id,
            end_doc_id,
            chunks_count,
            len(vectors),
        )
        raise ProcessingError(
            "Document embedding batch size mismatch start_doc_id=%s end_doc_id=%s expected=%s actual=%s"
            % (start_doc_id, end_doc_id, chunks_count, len(vectors))
        )
    elapsed_sec = perf_counter() - embed_start
    logger.info(
        "Document embedding window, chunks=%s total_sec:%.3f, avg_ms:%.3f ",
        chunks_count,
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
                "Document chunk selection failed doc_id=%s line_num=%s error_type=%s",
                doc_id,
                doc["line_num"],
                type(exc).__name__,
            )
            raise ProcessingError(
                "Document chunk selection failed doc_id=%s line_num=%s"
                % (doc_id, doc["line_num"])
            ) from exc

        if selected:
            output.write_doc_chunks(selected)
            total_selected_chunks += len(selected)

    write_elapsed_sec = perf_counter() - embed_start
    logger.info(
        "Doc window completed, total chunks:%s total_secs:%.3f",
        total_selected_chunks,
        write_elapsed_sec,
    )
    return total_selected_chunks


def _normalize_doc_text(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, list):
        normalized: List[str] = []
        for item in value:
            sentence = str(item).strip()
            if sentence:
                normalized.append(sentence)
        return normalized

    return []
