import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from index_builder.errors import InputValidationError


@dataclass
class DatasetContext:
    dataset_root: Path
    resolved_dataset_dir: Path
    dataset_meta: Dict[str, Any]
    docs_path: Path
    queries_path: Path


def resolve_subdataset_dir(dataset_root: Path, dataset_name: str) -> Path:
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise InputValidationError(f"--dataset-path does not exist or is not a directory: {dataset_root}")

    candidates = [p for p in dataset_root.iterdir() if p.is_dir()]
    exact = [p for p in candidates if p.name == dataset_name]
    if len(exact) == 1:
        return exact[0]

    lowered = [p for p in candidates if p.name.casefold() == dataset_name.casefold()]
    if not lowered:
        raise InputValidationError(
            f"No sub-dataset directory matched dataset_name={dataset_name!r} under {dataset_root}"
        )
    if len(lowered) > 1:
        names = [p.name for p in lowered]
        raise InputValidationError(
            f"Ambiguous dataset_name={dataset_name!r}, case-insensitive matches={names}"
        )
    return lowered[0]


def load_dataset_context(dataset_root: Path, dataset_name: str) -> DatasetContext:
    resolved_dataset_dir = resolve_subdataset_dir(dataset_root, dataset_name)
    dataset_json_path = resolved_dataset_dir / "dataset.json"
    if not dataset_json_path.exists():
        raise InputValidationError(f"Missing dataset.json in {resolved_dataset_dir}")

    with dataset_json_path.open("r", encoding="utf-8") as fin:
        dataset_meta = json.load(fin)
    if not isinstance(dataset_meta, dict):
        raise InputValidationError("dataset.json must be a JSON object")

    docs_rel = str(dataset_meta.get("docs_file", "docs.jsonl"))
    splits = dataset_meta.get("splits", {})
    if not isinstance(splits, dict) or "train" not in splits or not isinstance(splits["train"], dict):
        raise InputValidationError("dataset.json missing splits.train")
    train_queries_rel = str(splits["train"].get("queries_file", "train/queries.jsonl"))

    docs_path = resolved_dataset_dir / docs_rel
    queries_path = resolved_dataset_dir / train_queries_rel
    if not docs_path.exists():
        raise InputValidationError(f"Missing docs file: {docs_path}")
    if not queries_path.exists():
        raise InputValidationError(f"Missing train queries file: {queries_path}")

    return DatasetContext(
        dataset_root=dataset_root,
        resolved_dataset_dir=resolved_dataset_dir,
        dataset_meta=dataset_meta,
        docs_path=docs_path,
        queries_path=queries_path,
    )
