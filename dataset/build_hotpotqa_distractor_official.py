import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


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


def load_samples(
    input_file: Path,
    split: str,
    log_records: List[dict],
    error_counter: Counter,
) -> List[Any]:
    try:
        with input_file.open("r", encoding="utf-8") as fin:
            data = json.load(fin)
    except json.JSONDecodeError as exc:
        log_issue(
            log_records,
            error_counter,
            file=input_file,
            line_num=exc.lineno or 0,
            issue_type="invalid_json_file",
            action="skip_file",
            record_type="sample",
            split=split,
            detail=f"{exc.msg} (line={exc.lineno}, col={exc.colno})",
        )
        return []

    if not isinstance(data, list):
        log_issue(
            log_records,
            error_counter,
            file=input_file,
            line_num=1,
            issue_type="invalid_json_root_type",
            action="skip_file",
            record_type="sample",
            split=split,
            detail=f"type={type(data).__name__}",
        )
        return []

    return data


def clean_sentences(sentences_value: Any) -> List[str]:
    if not isinstance(sentences_value, list):
        return []

    cleaned = []
    for sentence in sentences_value:
        sentence_text = normalize_text(sentence)
        if sentence_text:
            cleaned.append(sentence_text)
    return cleaned


def build_global_docs(
    input_files: List[Tuple[Path, str]],
    output_file: Path,
    log_records: List[dict],
    error_counter: Counter,
) -> Dict[str, int]:
    seen_doc_ids: Set[str] = set()
    doc_payload_by_id: Dict[str, Dict[str, Any]] = {}

    doc_count = 0
    skipped_doc_count = 0
    conflicting_doc_count = 0

    for input_file, split in input_files:
        samples = load_samples(input_file, split, log_records, error_counter)

        for line_num, sample in enumerate(samples, 1):
            if not isinstance(sample, dict):
                continue

            context_value = sample.get("context")
            query_id = normalize_text(sample.get("_id"))

            if not isinstance(context_value, list):
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="invalid_context_format",
                    record_type="doc",
                    record_id=query_id,
                    split=split,
                    detail=f"type={type(context_value).__name__}",
                )
                continue

            for context_idx, context_entry in enumerate(context_value, 1):
                if not isinstance(context_entry, list) or len(context_entry) != 2:
                    skipped_doc_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="invalid_context_entry_format",
                        record_type="doc",
                        record_id=query_id,
                        split=split,
                        detail=f"context_idx={context_idx}",
                    )
                    continue

                title = normalize_text(context_entry[0])
                sentences_value = context_entry[1]

                if not title:
                    skipped_doc_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="missing_doc_title",
                        record_type="doc",
                        record_id=query_id,
                        split=split,
                        detail=f"context_idx={context_idx}",
                    )
                    continue

                if not isinstance(sentences_value, list):
                    skipped_doc_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="invalid_context_sentences_format",
                        record_type="doc",
                        record_id=title,
                        split=split,
                        detail=f"type={type(sentences_value).__name__}",
                    )
                    continue

                cleaned_sentences = clean_sentences(sentences_value)
                if not cleaned_sentences:
                    skipped_doc_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="empty_doc_sentences",
                        record_type="doc",
                        record_id=title,
                        split=split,
                    )
                    continue

                doc_id = title
                new_payload = {
                    "doc_id": doc_id,
                    "title": title,
                    "doc_text": cleaned_sentences,
                }

                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc_payload_by_id[doc_id] = new_payload
                    doc_count += 1
                    continue

                existing_payload = doc_payload_by_id[doc_id]
                if existing_payload["doc_text"] != cleaned_sentences:
                    conflicting_doc_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="conflicting_doc_content_same_title",
                        action="keep_first",
                        record_type="doc",
                        record_id=doc_id,
                        split=split,
                        detail="same title mapped to different sentence lists",
                    )

    with output_file.open("w", encoding="utf-8") as fout:
        for doc_id in sorted(doc_payload_by_id.keys()):
            write_jsonl_record(fout, doc_payload_by_id[doc_id])

    return {
        "doc_count": doc_count,
        "skipped_doc_count": skipped_doc_count,
        "conflicting_doc_count": conflicting_doc_count,
    }


def build_split(
    input_file: Path,
    split: str,
    output_dir: Path,
    global_doc_ids: Set[str],
    log_records: List[dict],
    error_counter: Counter,
) -> Dict[str, int]:
    samples = load_samples(input_file, split, log_records, error_counter)

    seen_query_ids: Set[str] = set()
    seen_qrel_pairs: Set[Tuple[str, str]] = set()
    positive_query_ids: Set[str] = set()

    query_count = 0
    qrels_count = 0
    skipped_query_count = 0
    skipped_qrels_count = 0

    queries_path = output_dir / "queries.jsonl"
    qrels_path = output_dir / "qrels.jsonl"

    with (
        queries_path.open("w", encoding="utf-8") as query_fout,
        qrels_path.open("w", encoding="utf-8") as qrels_fout,
    ):
        for line_num, sample in enumerate(samples, 1):
            if not isinstance(sample, dict):
                skipped_query_count += 1
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="invalid_sample_format",
                    record_type="sample",
                    split=split,
                    detail=f"type={type(sample).__name__}",
                )
                continue

            query_id = normalize_text(sample.get("_id"))
            query_text = normalize_text(sample.get("question"))

            if not query_id:
                skipped_query_count += 1
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
                skipped_query_count += 1
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

            if query_id in seen_query_ids:
                skipped_query_count += 1
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

            seen_query_ids.add(query_id)
            query_count += 1
            write_jsonl_record(
                query_fout,
                {
                    "query_id": query_id,
                    "query_text": query_text,
                },
            )

            supporting_facts_value = sample.get("supporting_facts")
            if not isinstance(supporting_facts_value, list):
                log_issue(
                    log_records,
                    error_counter,
                    file=input_file,
                    line_num=line_num,
                    issue_type="invalid_supporting_facts_format",
                    record_type="qrel",
                    record_id=query_id,
                    split=split,
                    detail=f"type={type(supporting_facts_value).__name__}",
                )
                continue

            supporting_titles: Set[str] = set()
            for fact_idx, fact in enumerate(supporting_facts_value, 1):
                if not isinstance(fact, list) or len(fact) != 2:
                    skipped_qrels_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="invalid_supporting_fact_entry_format",
                        record_type="qrel",
                        record_id=query_id,
                        split=split,
                        detail=f"fact_idx={fact_idx}",
                    )
                    continue

                title = normalize_text(fact[0])
                if not title:
                    skipped_qrels_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="missing_supporting_fact_title",
                        record_type="qrel",
                        record_id=query_id,
                        split=split,
                        detail=f"fact_idx={fact_idx}",
                    )
                    continue

                supporting_titles.add(title)

            wrote_positive_qrel = False
            for title in supporting_titles:
                doc_id = title
                pair = (query_id, doc_id)

                if doc_id not in global_doc_ids:
                    skipped_qrels_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="missing_supporting_doc_in_global_docs",
                        record_type="qrel",
                        record_id=f"{query_id}::{doc_id}",
                        split=split,
                    )
                    continue

                if pair in seen_qrel_pairs:
                    skipped_qrels_count += 1
                    log_issue(
                        log_records,
                        error_counter,
                        file=input_file,
                        line_num=line_num,
                        issue_type="duplicate_qrel_same_relevance",
                        record_type="qrel",
                        record_id=f"{query_id}::{doc_id}",
                        split=split,
                        detail="1",
                    )
                    continue

                seen_qrel_pairs.add(pair)
                qrels_count += 1
                wrote_positive_qrel = True
                write_jsonl_record(
                    qrels_fout,
                    {
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "relevance": 1,
                    },
                )

            if wrote_positive_qrel:
                positive_query_ids.add(query_id)

    orphan_query_count = query_count - len(positive_query_ids)
    return {
        "query_count": query_count,
        "qrels_count": qrels_count,
        "positive_query_count": len(positive_query_ids),
        "orphan_query_count": orphan_query_count,
        "skipped_query_count": skipped_query_count,
        "skipped_qrels_count": skipped_qrels_count,
    }


def write_dataset_json(output_file: Path, stats: Dict) -> None:
    dataset = {
        "dataset_name": "hotpotqa",
        "version": "v1",
        "subset": "distractor",
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
            "source_dataset": "hotpotqa",
            "source_subset": "distractor",
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


def load_global_doc_ids(docs_file: Path) -> Set[str]:
    doc_ids: Set[str] = set()
    with docs_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = normalize_text(record.get("doc_id"))
            if doc_id:
                doc_ids.add(doc_id)
    return doc_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HotpotQA distractor benchmark layout with a global shared document corpus."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing HotpotQA JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="hotpotqa_distractor_v1_global",
        help="Output benchmark directory.",
    )
    parser.add_argument(
        "--train-file",
        default="hotpot_train_v1.1.json",
        help="Train JSON file name.",
    )
    parser.add_argument(
        "--dev-file",
        default="hotpot_dev_distractor_v1.json",
        help="Dev distractor JSON file name.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_input = input_dir / args.train_file
    dev_input = input_dir / args.dev_file

    required_files = [train_input, dev_input]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "train")
    ensure_dir(output_dir / "dev")

    log_records: List[dict] = []
    error_counter: Counter = Counter()

    docs_stats = build_global_docs(
        input_files=[(train_input, "train"), (dev_input, "dev")],
        output_file=output_dir / "docs.jsonl",
        log_records=log_records,
        error_counter=error_counter,
    )

    global_doc_ids = load_global_doc_ids(output_dir / "docs.jsonl")

    train_stats = build_split(
        input_file=train_input,
        split="train",
        output_dir=output_dir / "train",
        global_doc_ids=global_doc_ids,
        log_records=log_records,
        error_counter=error_counter,
    )
    dev_stats = build_split(
        input_file=dev_input,
        split="dev",
        output_dir=output_dir / "dev",
        global_doc_ids=global_doc_ids,
        log_records=log_records,
        error_counter=error_counter,
    )

    stats = {
        "docs": docs_stats,
        "train": train_stats,
        "dev": dev_stats,
        "overall": {
            "doc_count": docs_stats["doc_count"],
            "query_count": train_stats["query_count"] + dev_stats["query_count"],
            "qrels_count": train_stats["qrels_count"] + dev_stats["qrels_count"],
            "positive_query_count": train_stats["positive_query_count"]
            + dev_stats["positive_query_count"],
            "orphan_query_count": train_stats["orphan_query_count"]
            + dev_stats["orphan_query_count"],
            "skipped_doc_count": docs_stats["skipped_doc_count"],
            "conflicting_doc_count": docs_stats["conflicting_doc_count"],
            "skipped_query_count": train_stats["skipped_query_count"]
            + dev_stats["skipped_query_count"],
            "skipped_qrels_count": train_stats["skipped_qrels_count"]
            + dev_stats["skipped_qrels_count"],
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
