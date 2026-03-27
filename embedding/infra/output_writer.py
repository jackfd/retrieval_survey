import json
from pathlib import Path
from typing import Dict

import pandas as pd


class OutputWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_parquet_path = self.output_dir / "docs.parquet"
        self.queries_parquet_path = self.output_dir / "queries.parquet"
        self.metadata_path = self.output_dir / "run_metadata.json"

    def load_existing_docs(self) -> pd.DataFrame:
        if not self.docs_parquet_path.exists():
            return pd.DataFrame(columns=["doc_id", "chunk_text", "chunk_embedding"])
        return pd.read_parquet(self.docs_parquet_path)

    def write_docs(self, docs_df: pd.DataFrame) -> None:
        docs_df.to_parquet(self.docs_parquet_path, index=False)

    def write_queries(self, queries_df: pd.DataFrame) -> None:
        queries_df.to_parquet(self.queries_parquet_path, index=False)

    def write_metadata(self, metadata: Dict[str, object]) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as fout:
            json.dump(metadata, fout, ensure_ascii=False, indent=2)

    def write_all_outputs(
        self,
        *,
        docs_df: pd.DataFrame,
        queries_df: pd.DataFrame,
        metadata: Dict[str, object],
    ) -> None:
        self.write_docs(docs_df)
        self.write_queries(queries_df)
        self.write_metadata(metadata)
