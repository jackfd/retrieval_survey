import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from retrieval_dataset._io import (
    ensure_dir,
    log_issue,
    normalize_text,
    write_build_log,
    write_jsonl_record,
)


def build_docs(
    input_file: Path,
    output_file: Path,
    log_records: List[dict],
    error_counter: Counter,
) -> Tuple[int, Set[str], int]:
    doc_ids: Set[str] = set()
    written = 0
    skipped = 0

    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="invalid_doc_row_format",
                    record_type="doc",
                    detail=line[:300],
                )
                continue

            doc_id, doc_text = parts
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()

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


def build_queries(
    input_file: Path,
    output_file: Path,
    split: str,
    log_records: List[dict],
    error_counter: Counter,
) -> Tuple[int, Set[str], int]:
    query_ids: Set[str] = set()
    written = 0
    skipped = 0

    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="invalid_query_row_format",
                    record_type="query",
                    split=split,
                    detail=line[:300],
                )
                continue

            query_id, query_text = parts
            query_id = query_id.strip()
            query_text = query_text.strip()

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
    seen_pairs: Dict[Tuple[str, str], int] = {}
    written = 0
    skipped = 0

    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t")
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

            query_id, _, doc_id, relevance = parts
            query_id = query_id.strip()
            doc_id = doc_id.strip()
            relevance = relevance.strip()

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
                    record_id=query_id,
                    split=split,
                )
                continue

            try:
                relevance_value = int(relevance)
            except ValueError:
                skipped += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="invalid_relevance",
                    record_type="qrel",
                    record_id=f"{query_id}::{doc_id}",
                    split=split,
                    detail=relevance,
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
                    record_id=f"{query_id}::{doc_id}",
                    split=split,
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
                    record_id=f"{query_id}::{doc_id}",
                    split=split,
                )
                continue

            pair = (query_id, doc_id)
            if pair in seen_pairs:
                existing_rel = seen_pairs[pair]
                if existing_rel == relevance_value:
                    skipped += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="duplicate_qrel_same_relevance",
                        record_type="qrel",
                        record_id=f"{query_id}::{doc_id}",
                        split=split,
                        detail=str(relevance_value),
                    )
                    continue
                else:
                    skipped += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="duplicate_qrel_conflicting_relevance",
                        record_type="qrel",
                        record_id=f"{query_id}::{doc_id}",
                        split=split,
                        detail=f"existing={existing_rel}, new={relevance_value}",
                    )
                    continue

            seen_pairs[pair] = relevance_value
            write_jsonl_record(
                fout,
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": relevance_value,
                },
            )
            written += 1

    positive_query_ids = {qid for qid, _ in seen_pairs.keys()}
    return written, len(positive_query_ids), skipped


def count_orphan_queries(query_ids: Set[str], qrels_file: Path) -> int:
    referenced_query_ids: Set[str] = set()

    with qrels_file.open("r", encoding="utf-8") as fin:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = obj.get("query_id")
            if query_id:
                referenced_query_ids.add(str(query_id))

    return len(query_ids - referenced_query_ids)


def write_dataset_json(output_file: Path, stats: Dict) -> None:
    dataset = {
        "dataset_name": "msmarco",
        "version": "v1",
        "subset": "",
        "task": "doc_retrieval",
        "docs_file": "docs.jsonl",
        "splits": {
            "train": {
                "queries_file": "train/queries.jsonl",
                "qrels_file": "train/qrels.jsonl",
            },
            "dev": {
                "queries_file": "dev/queries.jsonl",
                "qrels_file": "dev/qrels.jsonl",
            },
        },
        "metadata": {
            "domain": "general",
            "language": "en",
            "source_dataset": "msmarco",
            "source_subset": "",
            "version": "v1",
        },
        "stats": stats,
    }

    with output_file.open("w", encoding="utf-8") as fout:
        json.dump(dataset, fout, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MSMARCO benchmark layout in weak-validation mode.")
    parser.add_argument("--input-dir", default=".", help="Directory containing raw MSMARCO TSV files.")
    parser.add_argument("--output-dir", default="msmarco_v1", help="Output benchmark directory.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    collection_file = input_dir / "collection.tsv"
    train_queries_file = input_dir / "queries.train.tsv"
    dev_queries_file = input_dir / "queries.dev.tsv"
    train_qrels_file = input_dir / "qrels.train.tsv"
    dev_qrels_file = input_dir / "qrels.dev.tsv"

    required_files = [
        collection_file,
        train_queries_file,
        dev_queries_file,
        train_qrels_file,
        dev_qrels_file,
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "train")
    ensure_dir(output_dir / "dev")

    log_records: List[dict] = []
    error_counter: Counter = Counter()

    docs_count, doc_ids, skipped_docs = build_docs(
        collection_file,
        output_dir / "docs.jsonl",
        log_records,
        error_counter,
    )

    train_queries_count, train_query_ids, skipped_train_queries = build_queries(
        train_queries_file,
        output_dir / "train" / "queries.jsonl",
        "train",
        log_records,
        error_counter,
    )

    dev_queries_count, dev_query_ids, skipped_dev_queries = build_queries(
        dev_queries_file,
        output_dir / "dev" / "queries.jsonl",
        "dev",
        log_records,
        error_counter,
    )

    train_qrels_count, train_positive_queries, skipped_train_qrels = build_qrels(
        train_qrels_file,
        output_dir / "train" / "qrels.jsonl",
        "train",
        train_query_ids,
        doc_ids,
        log_records,
        error_counter,
    )

    dev_qrels_count, dev_positive_queries, skipped_dev_qrels = build_qrels(
        dev_qrels_file,
        output_dir / "dev" / "qrels.jsonl",
        "dev",
        dev_query_ids,
        doc_ids,
        log_records,
        error_counter,
    )

    train_orphan_queries = count_orphan_queries(
        train_query_ids,
        output_dir / "train" / "qrels.jsonl",
    )
    dev_orphan_queries = count_orphan_queries(
        dev_query_ids,
        output_dir / "dev" / "qrels.jsonl",
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
        "dev": {
            "query_count": dev_queries_count,
            "qrels_count": dev_qrels_count,
            "positive_query_count": dev_positive_queries,
            "orphan_query_count": dev_orphan_queries,
            "skipped_query_count": skipped_dev_queries,
            "skipped_qrels_count": skipped_dev_qrels,
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
