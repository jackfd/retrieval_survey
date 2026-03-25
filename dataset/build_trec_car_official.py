import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl_record(fout, record: dict) -> None:
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(value) -> str:
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


def get_read_data_module():
    try:
        from trec_car import read_data  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "trec-car-tools is required to read cbor files. "
            "Install with: pip install trec-car-tools"
        ) from exc
    return read_data


def extract_paragraph_fields(paragraph) -> Tuple[str, str]:
    doc_id = normalize_text(
        getattr(paragraph, "para_id", "")
        or getattr(paragraph, "paragraph_id", "")
        or getattr(paragraph, "id", "")
    )

    doc_text = ""
    get_text_fn = getattr(paragraph, "get_text", None)
    if callable(get_text_fn):
        try:
            doc_text = normalize_text(get_text_fn())
        except Exception:
            doc_text = ""

    if not doc_text:
        doc_text = normalize_text(
            getattr(paragraph, "text", "")
            or getattr(paragraph, "paragraph_text", "")
            or getattr(paragraph, "paragraph", "")
        )

    if not doc_text:
        bodies = getattr(paragraph, "bodies", None)
        if isinstance(bodies, list):
            body_texts = []
            for body in bodies:
                text = normalize_text(getattr(body, "text", ""))
                if text:
                    body_texts.append(text)
            doc_text = " ".join(body_texts)

    return doc_id, doc_text


def build_docs(
    input_file: Path,
    output_file: Path,
    log_records: List[dict],
    error_counter: Counter,
) -> Tuple[int, Set[str], int]:
    read_data = get_read_data_module()

    doc_ids: Set[str] = set()
    written = 0
    skipped = 0

    with input_file.open("rb") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_num, paragraph in enumerate(read_data.iter_paragraphs(fin), 1):
            doc_id, doc_text = extract_paragraph_fields(paragraph)
            if not doc_id:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="missing_doc_id",
                    record_type="doc",
                )
                continue

            if not doc_text:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="empty_doc_text",
                    record_type="doc",
                    record_id=doc_id,
                )
                continue

            if doc_id in doc_ids:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="duplicate_doc_id",
                    record_type="doc",
                    record_id=doc_id,
                )
                continue

            doc_ids.add(doc_id)
            write_jsonl_record(fout, {"doc_id": doc_id, "doc_text": doc_text})
            written += 1

    return written, doc_ids, skipped


def build_queries_from_outlines(
    input_file: Path,
    output_file: Path,
    split: str,
    log_records: List[dict],
    error_counter: Counter,
) -> Tuple[int, Set[str], int]:
    read_data = get_read_data_module()

    query_ids: Set[str] = set()
    written = 0
    skipped = 0

    with input_file.open("rb") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_num, page in enumerate(read_data.iter_outlines(fin), 1):
            query_id = normalize_text(getattr(page, "page_id", ""))
            query_text = normalize_text(getattr(page, "page_name", ""))

            if not query_id:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="missing_query_id",
                    record_type="query",
                    split=split,
                )
                continue

            if not query_text:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="empty_query_text",
                    record_type="query",
                    record_id=query_id,
                    split=split,
                )
                continue

            if query_id in query_ids:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="duplicate_query_id",
                    record_type="query",
                    record_id=query_id,
                    split=split,
                )
                continue

            query_ids.add(query_id)
            write_jsonl_record(fout, {"query_id": query_id, "query_text": query_text})
            written += 1

    return written, query_ids, skipped


def build_qrels(
    input_file: Path,
    output_file: Path,
    split: str,
    valid_query_ids: Set[str],
    valid_doc_ids: Set[str],
    log_records: List[dict],
    error_counter: Counter,
) -> Tuple[int, int, int]:
    seen_pairs: Set[Tuple[str, str]] = set()
    written = 0
    skipped = 0

    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="invalid_qrel_row_format",
                    record_type="qrel",
                    split=split,
                    detail=line[:300],
                )
                continue

            query_id = normalize_text(parts[0])
            doc_id = normalize_text(parts[2])
            if not query_id:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="missing_qrel_query_id",
                    record_type="qrel",
                    split=split,
                )
                continue

            if not doc_id:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="missing_qrel_doc_id",
                    record_type="qrel",
                    split=split,
                    record_id=query_id,
                )
                continue

            if query_id not in valid_query_ids:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="missing_query_ref",
                    record_type="qrel",
                    split=split,
                    record_id=f"{query_id}::{doc_id}",
                )
                continue

            if doc_id not in valid_doc_ids:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="missing_doc_ref",
                    record_type="qrel",
                    split=split,
                    record_id=f"{query_id}::{doc_id}",
                )
                continue

            pair = (query_id, doc_id)
            if pair in seen_pairs:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="duplicate_qrel_same_relevance",
                    record_type="qrel",
                    split=split,
                    record_id=f"{query_id}::{doc_id}",
                    detail="1",
                )
                continue

            seen_pairs.add(pair)
            write_jsonl_record(
                fout,
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": 1,
                },
            )
            written += 1

    positive_query_ids = {qid for qid, _ in seen_pairs}
    return written, len(positive_query_ids), skipped


def count_orphan_queries(query_ids: Set[str], qrels_file: Path) -> int:
    referenced_query_ids: Set[str] = set()
    with qrels_file.open("r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(obj.get("relevance", 0)) > 0:
                query_id = normalize_text(obj.get("query_id"))
                if query_id:
                    referenced_query_ids.add(query_id)
    return len(query_ids - referenced_query_ids)


def write_dataset_json(output_file: Path, stats: Dict) -> None:
    dataset = {
        "dataset_name": "trec_car",
        "version": "v1",
        "subset": "article",
        "task": "article_paragraph_retrieval",
        "docs_file": "docs.jsonl",
        "splits": {
            "train": {
                "queries_file": "train/queries.jsonl",
                "qrels_file": "train/qrels.jsonl",
            }
        },
        "metadata": {
            "domain": "general",
            "language": "en",
            "source_dataset": "trec_car",
            "source_subset": "article",
            "version": "v1",
        },
        "stats": stats,
    }
    with output_file.open("w", encoding="utf-8") as fout:
        json.dump(dataset, fout, ensure_ascii=False, indent=2)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TREC CAR article-level benchmark layout in weak-validation mode."
    )
    parser.add_argument("--input-dir", default=".", help="Directory containing TREC CAR source files.")
    parser.add_argument("--output-dir", default="trec_car_v1", help="Output benchmark directory.")
    parser.add_argument(
        "--docs-file",
        default="base.train.cbor-paragraphs.cbor",
        help="Paragraph cbor file.",
    )
    parser.add_argument(
        "--outlines-file",
        default="base.train.cbor-outlines.cbor",
        help="Outline cbor file for train queries.",
    )
    parser.add_argument(
        "--qrels-file",
        default="base.train.cbor-article.qrels",
        help="Article-level qrels file.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    docs_file = input_dir / args.docs_file
    outlines_file = input_dir / args.outlines_file
    qrels_file = input_dir / args.qrels_file

    required_files = [docs_file, outlines_file, qrels_file]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "train")

    log_records: List[dict] = []
    error_counter: Counter = Counter()

    docs_count, doc_ids, skipped_docs = build_docs(
        docs_file,
        output_dir / "docs.jsonl",
        log_records,
        error_counter,
    )

    train_queries_count, train_query_ids, skipped_train_queries = build_queries_from_outlines(
        outlines_file,
        output_dir / "train" / "queries.jsonl",
        "train",
        log_records,
        error_counter,
    )

    train_qrels_count, train_positive_queries, skipped_train_qrels = build_qrels(
        qrels_file,
        output_dir / "train" / "qrels.jsonl",
        "train",
        train_query_ids,
        doc_ids,
        log_records,
        error_counter,
    )

    train_orphan_queries = count_orphan_queries(
        train_query_ids,
        output_dir / "train" / "qrels.jsonl",
    )

    stats = {
        "docs": {
            "doc_count": docs_count,
            "skipped_count": skipped_docs,
        },
        "train": {
            "query_count": train_queries_count,
            "qrels_count": train_qrels_count,
            "positive_query_count": train_positive_queries,
            "orphan_query_count": train_orphan_queries,
            "skipped_query_count": skipped_train_queries,
            "skipped_qrels_count": skipped_train_qrels,
        },
    }

    write_dataset_json(output_dir / "dataset.json", stats)
    write_build_log(output_dir / "build_log.json", log_records, error_counter, stats)

    print("Build completed successfully.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Error summary:")
    print(json.dumps(dict(error_counter), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
