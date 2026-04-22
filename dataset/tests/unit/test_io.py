import io
import json
from collections import Counter
from pathlib import Path

import pytest

from retrieval_dataset._io import (
    ensure_dir,
    log_issue,
    normalize_text,
    write_build_log,
    write_jsonl_record,
)


# ── normalize_text ────────────────────────────────────────────────────────────

def test_normalize_text_none():
    assert normalize_text(None) == ""


def test_normalize_text_strips_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_normalize_text_empty_string():
    assert normalize_text("") == ""


def test_normalize_text_non_string():
    assert normalize_text(42) == "42"
    assert normalize_text(3.14) == "3.14"


def test_normalize_text_list():
    assert normalize_text([1, 2]) == "[1, 2]"


# ── write_jsonl_record ────────────────────────────────────────────────────────

def test_write_jsonl_record_basic():
    buf = io.StringIO()
    write_jsonl_record(buf, {"a": 1, "b": "hello"})
    line = buf.getvalue()
    assert line.endswith("\n")
    obj = json.loads(line)
    assert obj == {"a": 1, "b": "hello"}


def test_write_jsonl_record_unicode():
    buf = io.StringIO()
    write_jsonl_record(buf, {"text": "中文"})
    obj = json.loads(buf.getvalue())
    assert obj["text"] == "中文"


def test_write_jsonl_record_nested():
    buf = io.StringIO()
    record = {"doc_id": "1", "doc_text": ["sentence one.", "sentence two."]}
    write_jsonl_record(buf, record)
    obj = json.loads(buf.getvalue())
    assert obj["doc_text"] == ["sentence one.", "sentence two."]


# ── log_issue ─────────────────────────────────────────────────────────────────

def test_log_issue_increments_counter():
    log_records: list = []
    counter: Counter = Counter()
    log_issue(log_records, counter, file=Path("f.txt"), line_num=1, issue_type="bad_id")
    assert counter["bad_id"] == 1


def test_log_issue_record_has_required_fields():
    log_records: list = []
    counter: Counter = Counter()
    log_issue(log_records, counter, file=Path("x.jsonl"), line_num=5, issue_type="missing_doc_id")
    rec = log_records[0]
    assert rec["file"] == "x.jsonl"
    assert rec["line_num"] == 5
    assert rec["issue_type"] == "missing_doc_id"
    assert rec["action"] == "skip"


def test_log_issue_optional_fields_absent_when_empty():
    log_records: list = []
    counter: Counter = Counter()
    log_issue(log_records, counter, file=Path("f.txt"), line_num=1, issue_type="x")
    rec = log_records[0]
    assert "record_type" not in rec
    assert "record_id" not in rec
    assert "split" not in rec
    assert "detail" not in rec


def test_log_issue_optional_fields_present_when_set():
    log_records: list = []
    counter: Counter = Counter()
    log_issue(
        log_records,
        counter,
        file=Path("f.txt"),
        line_num=1,
        issue_type="x",
        record_type="doc",
        record_id="doc_001",
        split="train",
        detail="some detail",
    )
    rec = log_records[0]
    assert rec["record_type"] == "doc"
    assert rec["record_id"] == "doc_001"
    assert rec["split"] == "train"
    assert rec["detail"] == "some detail"


def test_log_issue_accumulates_multiple():
    log_records: list = []
    counter: Counter = Counter()
    log_issue(log_records, counter, file=Path("f.txt"), line_num=1, issue_type="a")
    log_issue(log_records, counter, file=Path("f.txt"), line_num=2, issue_type="a")
    log_issue(log_records, counter, file=Path("f.txt"), line_num=3, issue_type="b")
    assert counter["a"] == 2
    assert counter["b"] == 1
    assert len(log_records) == 3


# ── write_build_log ───────────────────────────────────────────────────────────

def test_write_build_log_structure(tmp_path):
    log_records = [{"file": "f.txt", "line_num": 1, "issue_type": "x", "action": "skip"}]
    counter = Counter({"x": 1})
    stats = {"docs": {"doc_count": 10, "skipped_count": 1}}
    out = tmp_path / "build_log.json"
    write_build_log(out, log_records, counter, stats)

    with out.open() as f:
        payload = json.load(f)

    assert payload["stats"] == stats
    assert payload["error_summary"] == {"x": 1}
    assert payload["issue_logs"] == log_records


def test_write_build_log_empty(tmp_path):
    out = tmp_path / "build_log.json"
    write_build_log(out, [], Counter(), {})
    with out.open() as f:
        payload = json.load(f)
    assert payload["issue_logs"] == []
    assert payload["error_summary"] == {}


# ── ensure_dir ────────────────────────────────────────────────────────────────

def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    ensure_dir(target)
    assert target.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    ensure_dir(target)  # should not raise
    assert target.is_dir()
