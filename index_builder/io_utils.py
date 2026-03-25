import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fin:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            yield line_num, obj


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("index_builder")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_failures(path: Path) -> List[str]:
    if not path.exists():
        return []
    doc_ids: List[str] = []
    for _, obj in read_jsonl(path):
        record_type = str(obj.get("record_type", "")).strip()
        if record_type and record_type != "doc":
            continue
        record_id = str(obj.get("record_id", "")).strip()
        if record_id:
            doc_ids.append(record_id)
            continue
        # Backward compatibility for old failures.jsonl format.
        doc_id = str(obj.get("doc_id", "")).strip()
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def load_existing_docs_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["doc_id", "chunk_text", "chunk_embedding"])
    return pd.read_parquet(path)


def write_failures(path: Path, failures: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for item in failures:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_docs_with_retry(
    *,
    new_docs_df: pd.DataFrame,
    existing_docs_df: pd.DataFrame,
    retry_mode: bool,
    retry_id_set: set[str],
) -> pd.DataFrame:
    if retry_mode and not existing_docs_df.empty:
        no_retry_df = existing_docs_df[~existing_docs_df["doc_id"].isin(retry_id_set)]
        merged_docs_df = pd.concat([no_retry_df, new_docs_df], ignore_index=True)
    elif retry_mode and existing_docs_df.empty:
        merged_docs_df = new_docs_df
    else:
        merged_docs_df = new_docs_df

    if not merged_docs_df.empty:
        merged_docs_df = merged_docs_df.drop_duplicates(subset=["doc_id"], keep="last")
    return merged_docs_df
