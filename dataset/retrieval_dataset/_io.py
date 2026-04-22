"""Shared I/O utilities for dataset build scripts."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl_record(fout, record: dict) -> None:
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def log_issue(
    log_records: List[dict],
    error_counter: Counter,
    *,
    file: Path,
    line_num: int,
    issue_type: str,
    action: str = "skip",
    record_type: str = "",
    record_id: str = "",
    split: str = "",
    detail: str = "",
) -> None:
    error_counter[issue_type] += 1
    rec = {
        "file": str(file),
        "line_num": line_num,
        "issue_type": issue_type,
        "action": action,
    }
    if record_type:
        rec["record_type"] = record_type
    if record_id:
        rec["record_id"] = record_id
    if split:
        rec["split"] = split
    if detail:
        rec["detail"] = detail
    log_records.append(rec)


def write_build_log(
    output_file: Path,
    log_records: List[dict],
    error_counter: Counter,
    stats: Dict,
) -> None:
    payload = {
        "stats": stats,
        "error_summary": dict(error_counter),
        "issue_logs": log_records,
    }
    with output_file.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
