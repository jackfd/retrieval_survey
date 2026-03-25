import json
import logging
from pathlib import Path

from embed_pipe.domain.errors import InputValidationError
from embed_pipe.domain.models import DatasetContext


class DatasetLoader:
    def resolve_subdataset_dir(self, dataset_root: Path, dataset_name: str) -> Path:
        logger = logging.getLogger("index_builder")
        if not dataset_root.exists() or not dataset_root.is_dir():
            logger.error(
                "event=dataset_validation_failed reason=%s context=%s",
                "--dataset-path does not exist or is not a directory",
                "dataset_root=%s" % dataset_root,
            )
            raise InputValidationError("--dataset-path does not exist or is not a directory: %s" % dataset_root)

        candidates = [path for path in dataset_root.iterdir() if path.is_dir()]
        exact = [path for path in candidates if path.name == dataset_name]
        if len(exact) == 1:
            return exact[0]

        lowered = [path for path in candidates if path.name.casefold() == dataset_name.casefold()]
        if not lowered:
            logger.error(
                "event=dataset_resolution_failed reason=%s context=%s",
                "No sub-dataset directory matched dataset_name",
                "dataset_root=%s dataset_name=%s" % (dataset_root, dataset_name),
            )
            raise InputValidationError(
                "No sub-dataset directory matched dataset_name=%r under %s" % (dataset_name, dataset_root)
            )
        if len(lowered) > 1:
            names = [path.name for path in lowered]
            logger.error(
                "event=dataset_resolution_failed reason=%s context=%s",
                "Ambiguous dataset_name case-insensitive matches",
                "dataset_root=%s dataset_name=%s matches=%s" % (dataset_root, dataset_name, names),
            )
            raise InputValidationError(
                "Ambiguous dataset_name=%r, case-insensitive matches=%s" % (dataset_name, names)
            )
        return lowered[0]

    def load_dataset_context(self, dataset_root: Path, dataset_name: str) -> DatasetContext:
        logger = logging.getLogger("index_builder")
        resolved_dataset_dir = self.resolve_subdataset_dir(dataset_root, dataset_name)
        dataset_json_path = resolved_dataset_dir / "dataset.json"
        if not dataset_json_path.exists():
            logger.error(
                "event=dataset_validation_failed reason=%s context=%s",
                "Missing dataset.json",
                "dataset_dir=%s" % resolved_dataset_dir,
            )
            raise InputValidationError("Missing dataset.json in %s" % resolved_dataset_dir)

        with dataset_json_path.open("r", encoding="utf-8") as fin:
            dataset_meta = json.load(fin)
        if not isinstance(dataset_meta, dict):
            logger.error(
                "event=dataset_validation_failed reason=%s context=%s",
                "dataset.json must be a JSON object",
                "dataset_json=%s" % dataset_json_path,
            )
            raise InputValidationError("dataset.json must be a JSON object")

        docs_rel = str(dataset_meta.get("docs_file", "docs.jsonl"))
        splits = dataset_meta.get("splits", {})
        if not isinstance(splits, dict) or "train" not in splits or not isinstance(splits["train"], dict):
            logger.error(
                "event=dataset_validation_failed reason=%s context=%s",
                "dataset.json missing splits.train",
                "dataset_json=%s" % dataset_json_path,
            )
            raise InputValidationError("dataset.json missing splits.train")
        train_queries_rel = str(splits["train"].get("queries_file", "train/queries.jsonl"))

        docs_path = resolved_dataset_dir / docs_rel
        queries_path = resolved_dataset_dir / train_queries_rel
        if not docs_path.exists():
            logger.error(
                "event=dataset_validation_failed reason=%s context=%s",
                "Missing docs file",
                "dataset_dir=%s docs_path=%s" % (resolved_dataset_dir, docs_path),
            )
            raise InputValidationError("Missing docs file: %s" % docs_path)
        if not queries_path.exists():
            logger.error(
                "event=dataset_validation_failed reason=%s context=%s",
                "Missing train queries file",
                "dataset_dir=%s queries_path=%s" % (resolved_dataset_dir, queries_path),
            )
            raise InputValidationError("Missing train queries file: %s" % queries_path)

        return DatasetContext(
            dataset_root=dataset_root,
            resolved_dataset_dir=resolved_dataset_dir,
            dataset_meta=dataset_meta,
            docs_path=docs_path,
            queries_path=queries_path,
        )
