import json
import logging
from pathlib import Path

from embed_pipe.domain.exceptions import DatasetError
from embed_pipe.domain.models import DatasetContext

logger = logging.getLogger(__name__)


class DatasetLoader:
    def resolve_subdataset_dir(self, dataset_root: Path, dataset_name: str) -> Path:
        if not dataset_root.exists() or not dataset_root.is_dir():
            logger.error(
                "path does not exist or is not a directory dataset_root=%s",
                dataset_root,
            )
            raise DatasetError(
                "path does not exist or is not a directory dataset_root=%s"
                % dataset_root
            )

        candidates = [path for path in dataset_root.iterdir() if path.is_dir()]
        exact = [path for path in candidates if path.name == dataset_name]
        if len(exact) == 1:
            return exact[0]

        logger.error(
            "No sub-dataset directory matched dataset_name dataset_root=%s dataset_name=%s",
            dataset_root,
            dataset_name,
        )
        raise DatasetError(
            "No sub-dataset directory matched dataset_name dataset_root=%s dataset_name=%s"
            % (dataset_root, dataset_name)
        )

    def load_dataset_context(
        self, dataset_root: Path, dataset_name: str
    ) -> DatasetContext:
        resolved_dataset_dir = self.resolve_subdataset_dir(dataset_root, dataset_name)

        dataset_json_path = resolved_dataset_dir / "dataset.json"
        if not dataset_json_path.exists():
            logger.error("Missing dataset.json dataset_dir=%s", resolved_dataset_dir)
            raise DatasetError(
                "Missing dataset.json dataset_dir=%s" % resolved_dataset_dir
            )

        with dataset_json_path.open("r", encoding="utf-8") as fin:
            dataset_meta = json.load(fin)
        if not isinstance(dataset_meta, dict):
            logger.error(
                "dataset.json must be a JSON object dataset_json=%s", dataset_json_path
            )
            raise DatasetError(
                "dataset.json must be a JSON object dataset_json=%s" % dataset_json_path
            )

        docs_rel = str(dataset_meta.get("docs_file", "docs.jsonl"))
        splits = dataset_meta.get("splits", {})
        if (
            not isinstance(splits, dict)
            or "train" not in splits
            or not isinstance(splits["train"], dict)
        ):
            logger.error(
                "dataset.json missing splits.train dataset_json=%s", dataset_json_path
            )
            raise DatasetError(
                "dataset.json missing splits.train dataset_json=%s" % dataset_json_path
            )
        train_queries_rel = str(
            splits["train"].get("queries_file", "train/queries.jsonl")
        )

        docs_path = resolved_dataset_dir / docs_rel
        queries_path = resolved_dataset_dir / train_queries_rel
        if not docs_path.exists():
            logger.error(
                "Missing docs file dataset_dir=%s docs_path=%s",
                resolved_dataset_dir,
                docs_path,
            )
            raise DatasetError(
                "Missing docs file dataset_dir=%s docs_path=%s"
                % (resolved_dataset_dir, docs_path)
            )
        if not queries_path.exists():
            logger.error(
                "Missing train queries file dataset_dir=%s queries_path=%s",
                resolved_dataset_dir,
                queries_path,
            )
            raise DatasetError(
                "Missing train queries file dataset_dir=%s queries_path=%s"
                % (resolved_dataset_dir, queries_path)
            )

        return DatasetContext(
            dataset_root=dataset_root,
            resolved_dataset_dir=resolved_dataset_dir,
            dataset_meta=dataset_meta,
            docs_path=docs_path,
            queries_path=queries_path,
        )
