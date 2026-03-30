import shutil
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from embedding.domain.models import DOC_COLUMNS


class OutputWriter:
    def __init__(
        self, output_dir: Path, embedding_dim: int, docs_shard_size: int = 100_000
    ):
        if docs_shard_size <= 0:
            raise ValueError("docs_shard_size must be positive")

        output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.docs_path = output_dir / f"docs_dim{embedding_dim}"
        self.queries_path = output_dir / f"queries_dim{embedding_dim}"
        self.docs_shard_size = docs_shard_size
        self._chunks_buffer: list[dict[str, Any]] = []
        self._docs_part_index = 0
        self._docs_counts = 0

        self._reset_path(self.docs_path)
        self._reset_path(self.queries_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.queries_path.mkdir(parents=True, exist_ok=True)

    def write_doc_chunks(self, chunks: list[dict[str, Any]]) -> None:
        for record in chunks:
            self._chunks_buffer.append(record)
        if len(chunks) > 0:
            self._docs_counts += 1
            if self._docs_counts % self.docs_shard_size == 0:
                self._flush_docs_part()

    def _flush_docs_part(self) -> None:
        part_path = self.docs_path / f"part-{self._docs_part_index:05d}.parquet"
        df = pd.DataFrame(self._chunks_buffer, columns=DOC_COLUMNS)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, part_path)
        del self._chunks_buffer[:]
        self._docs_counts = 0
        self._docs_part_index += 1

    def write_queries(self, queries_df: pd.DataFrame) -> None:
        queries_df.to_parquet(self.queries_path / "queries.parquet", index=False)

    def close(self) -> None:
        if self._chunks_buffer:
            self._flush_docs_part()

    def _reset_path(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
