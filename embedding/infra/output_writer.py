import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from embedding.domain.models import DOC_COLUMNS, QUERY_COLUMNS


class _ParquetShardWriter:
    def __init__(self, base_path: Path, columns: list[str], shard_size: int):
        self.base_path = base_path
        self.columns = columns
        self.shard_size = shard_size
        self._buffer: list[dict[str, Any]] = []
        self._part_index = 0
        self._counts = 0
        self._reset_path(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        self._buffer.extend(records)
        self._counts += 1
        if self._counts % self.shard_size == 0:
            self._flush_part()

    def close(self) -> None:
        if self._buffer:
            self._flush_part()

    def _flush_part(self) -> None:
        part_path = self.base_path / f"part-{self._part_index:05d}.parquet"
        part_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._buffer, columns=self.columns)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, part_path)
        del self._buffer[:]
        self._counts = 0
        self._part_index += 1

    def _reset_path(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


class OutputWriter:
    def __init__(self, output_dir: Path, embedding_dim: int, shard_size: int = 100_000):
        if shard_size <= 0:
            raise ValueError("docs_shard_size must be positive")

        output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_path = output_dir / f"docs_dim{embedding_dim}"
        self.queries_path = output_dir / f"queries_dim{embedding_dim}"
        self._doc_writer = _ParquetShardWriter(self.docs_path, DOC_COLUMNS, shard_size)
        self._query_writer = _ParquetShardWriter(
            self.queries_path, QUERY_COLUMNS, shard_size
        )

    def write_doc_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self._doc_writer.write_records(chunks)

    def write_query_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self._query_writer.write_records(chunks)

    def close(self) -> None:
        self._doc_writer.close()
        self._query_writer.close()
