import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List

import numpy as np

from embedding.domain.exceptions import ProcessingError
from embedding.adapters.backends import EmbeddingStrategy
from embedding.adapters.jsonl_reader import read_objects
from embedding.adapters.output_writer import OutputWriter
from embedding.domain.chunk_selector import select_from_embeddings
from embedding.domain.chunk_splitter import ChunkSplitter

logger = logging.getLogger(__name__)

DOC_WINDOW_SIZE = 6400


def build_doc_candidates(doc_path: Path, output_path: Path, max_length: int) -> None:
    splitter = ChunkSplitter(max_length=max_length)
    doc_count = 0
    total_candidates = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for line_num, obj in read_objects(doc_path):
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

            fout.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "candidate_count": len(candidate_chunks),
                        "candidates": candidate_chunks,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            doc_count += 1
            total_candidates += len(candidate_chunks)

    logger.info(
        "doc candidates docs_path=%s doc_count=%s chunk_count=%s",
        doc_path,
        doc_count,
        total_candidates,
    )


def process_doc(
    embedding: EmbeddingStrategy, candidates_path: Path, output: OutputWriter
) -> None:
    logger.info("docs process started docs_path=%s", candidates_path)
    run_start = perf_counter()
    total_selected_chunks = 0
    read_iter = (obj for _line_num, obj in read_objects(candidates_path))

    while True:
        docs_window, chunk_candidates = _collect_doc_window(read_iter, DOC_WINDOW_SIZE)
        logger.info(
            "  1-docs collected, doc_count=%s chunk_count=%s",
            len(docs_window),
            len(chunk_candidates),
        )

        if not docs_window:
            break

        if not chunk_candidates:
            continue

        total_selected_chunks += _process_doc_window(
            embedding, docs_window, chunk_candidates, output
        )

    if total_selected_chunks == 0:
        logger.error("No docs chunks found in %s", candidates_path)
        raise ProcessingError(f"No documents found in {candidates_path}")

    elapsed_sec = perf_counter() - run_start
    logger.info(
        "docs process finished doc_count:%s total_sec:%.3f, avg_ms:%.3f",
        total_selected_chunks,
        elapsed_sec,
        elapsed_sec * 1000 / total_selected_chunks,
    )


def _collect_doc_window(
    read_iter: Iterator[Dict[str, Any]], doc_window_size: int
) -> tuple[List[Dict[str, Any]], List[str]]:
    docs_window: List[Dict[str, Any]] = []
    window_chunk_texts: List[str] = []

    while len(docs_window) < doc_window_size:
        try:
            obj = next(read_iter)
        except StopIteration:
            break

        candidate_chunks = obj["candidates"]
        chunk_start = len(window_chunk_texts)
        for chunk in candidate_chunks:
            window_chunk_texts.append(str(chunk["text"]))
        chunk_end = len(window_chunk_texts)

        docs_window.append(
            {
                "doc_id": obj["doc_id"],
                "candidates": candidate_chunks,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            }
        )
    return docs_window, window_chunk_texts


def _process_doc_window(
    embedding: EmbeddingStrategy,
    docs_window: List[Dict[str, Any]],
    chunk_candidates: List[str],
    output: OutputWriter,
) -> int:
    embed_start = perf_counter()
    start_doc_id = str(docs_window[0]["doc_id"])
    end_doc_id = str(docs_window[-1]["doc_id"])
    chunks_count = len(chunk_candidates)

    try:
        vectors = embedding.encode(chunk_candidates, is_query=False)
    except Exception as exc:
        logger.exception(
            "docs embedding batch failed, start_doc_id=%s end_doc_id=%s chunk_count=%s error_type=%s",
            start_doc_id,
            end_doc_id,
            chunks_count,
            type(exc).__name__,
        )
        raise ProcessingError(
            "docs embedding batch failed start_doc_id=%s end_doc_id=%s chunk_count=%s"
            % (start_doc_id, end_doc_id, chunks_count)
        ) from exc

    if len(vectors) != chunks_count:
        logger.error(
            "docs embedding batch size mismatch expected=%s actual=%s",
            chunks_count,
            len(vectors),
        )
        raise ProcessingError(
            "docs embedding batch size mismatch  expected=%s actual=%s"
            % (chunks_count, len(vectors))
        )
    elapsed_sec = perf_counter() - embed_start
    logger.info(
        "  2-docs embedding, dim=%s total_sec:%.3f, avg_ms:%.3f",
        vectors.shape[1],
        elapsed_sec,
        elapsed_sec * 1000 / chunks_count,
    )
    vector_matrix = np.asarray(vectors, dtype=np.float32)
    total_selected_chunks = 0

    for doc in docs_window:
        doc_candidates = doc["candidates"]
        if not doc_candidates:
            continue

        chunk_start = int(doc["chunk_start"])
        chunk_end = int(doc["chunk_end"])
        chunk_vectors = vector_matrix[chunk_start:chunk_end]
        doc_id = doc["doc_id"]
        try:
            selected = select_from_embeddings(doc_id, doc_candidates, chunk_vectors)
        except Exception as exc:
            logger.exception(
                "docs chunk selection failed doc_id=%s error_type=%s",
                doc_id,
                type(exc).__name__,
            )
            raise ProcessingError(
                "docs chunk selection failed doc_id=%s" % doc_id
            ) from exc

        if selected:
            output.write_doc_chunks(selected)
            total_selected_chunks += len(selected)

    return total_selected_chunks


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
