import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from embed_pipe.infra.jsonl_reader import JsonlReader


class OutputWriter:
    def __init__(self, output_dir: Path, jsonl_reader: JsonlReader | None = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_parquet_path = self.output_dir / "docs.parquet"
        self.queries_parquet_path = self.output_dir / "queries.parquet"
        self.failures_path = self.output_dir / "failures.jsonl"
        self.metadata_path = self.output_dir / "run_metadata.json"
        self._jsonl_reader = jsonl_reader or JsonlReader()

    def load_failed_doc_ids(self) -> Set[str]:
        if not self.failures_path.exists():
            return set()

        doc_ids: List[str] = []
        for _, obj in self._jsonl_reader.read_objects(self.failures_path):
            record_type = str(obj.get("record_type", "")).strip()
            if record_type and record_type != "doc":
                continue
            record_id = str(obj.get("record_id", "")).strip()
            if record_id:
                doc_ids.append(record_id)
                continue

            doc_id = str(obj.get("doc_id", "")).strip()
            if doc_id:
                doc_ids.append(doc_id)
        retry_id_set = set(doc_ids)
        return retry_id_set

    def load_existing_docs(self) -> pd.DataFrame:
        if not self.docs_parquet_path.exists():
            return pd.DataFrame(columns=["doc_id", "chunk_text", "chunk_embedding"])
        return pd.read_parquet(self.docs_parquet_path)

    def merge_docs_with_retry(
        self,
        *,
        new_docs_df: pd.DataFrame,
        existing_docs_df: pd.DataFrame,
        retry_mode: bool,
        retry_id_set: set[str],
    ) -> pd.DataFrame:
        if retry_mode and not existing_docs_df.empty:
            no_retry_df = existing_docs_df[
                ~existing_docs_df["doc_id"].isin(retry_id_set)
            ]
            merged_docs_df = pd.concat([no_retry_df, new_docs_df], ignore_index=True)
        elif retry_mode and existing_docs_df.empty:
            merged_docs_df = new_docs_df
        else:
            merged_docs_df = new_docs_df

        if not merged_docs_df.empty:
            merged_docs_df = merged_docs_df.drop_duplicates(
                subset=["doc_id"], keep="last"
            )
        return merged_docs_df

    def write_docs(self, docs_df: pd.DataFrame) -> None:
        docs_df.to_parquet(self.docs_parquet_path, index=False)

    def write_queries(self, queries_df: pd.DataFrame) -> None:
        queries_df.to_parquet(self.queries_parquet_path, index=False)

    def write_failures(self, failures: List[Dict[str, str]]) -> None:
        with self.failures_path.open("w", encoding="utf-8") as fout:
            for item in failures:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    def write_metadata(self, metadata: Dict[str, object]) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as fout:
            json.dump(metadata, fout, ensure_ascii=False, indent=2)

    def write_all_outputs(
        self,
        *,
        docs_df: pd.DataFrame,
        queries_df: pd.DataFrame,
        failures: List[Dict[str, str]],
        metadata: Dict[str, object],
    ) -> None:
        self.write_docs(docs_df)
        self.write_queries(queries_df)
        self.write_failures(failures)
        self.write_metadata(metadata)
