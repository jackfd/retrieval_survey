from pathlib import Path

import pandas as pd


class OutputWriter:
    def __init__(self, output_dir: Path, embedding_dim: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dim_label = int(embedding_dim)
        self.docs_parquet_path = self.output_dir / f"docs_dim{dim_label}.parquet"
        self.queries_path = self.output_dir / f"queries_dim{dim_label}.parquet"

    def write_docs(self, docs_df: pd.DataFrame) -> None:
        docs_df.to_parquet(self.docs_parquet_path, index=False)

    def write_queries(self, queries_df: pd.DataFrame) -> None:
        queries_df.to_parquet(self.queries_path, index=False)

    def write_all_outputs(
        self,
        *,
        docs_df: pd.DataFrame,
        queries_df: pd.DataFrame,
    ) -> None:
        self.write_docs(docs_df)
        self.write_queries(queries_df)
