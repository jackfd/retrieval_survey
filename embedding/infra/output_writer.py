from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class OutputWriter:
    def __init__(self, output_dir: Path, embedding_dim: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_path = self.output_dir / f"docs_dim{embedding_dim}.parquet"
        self.queries_path = self.output_dir / f"queries_dim{embedding_dim}.parquet"
        self._docs_writer: pq.ParquetWriter | None = None
        # 删除已存在的docs文件，确保从空白状态开始写入
        if self.docs_path.exists():
            self.docs_path.unlink()

    def write_docs(self, docs_df: pd.DataFrame) -> None:
        if docs_df.empty:
            return
        # 将pandas DataFrame转换为Arrow Table
        table = pa.Table.from_pandas(docs_df, preserve_index=False)
        if self._docs_writer is None:
            # 创建 ParquetWriter, 数据写入全新的文件中
            self._docs_writer = pq.ParquetWriter(self.docs_path, table.schema)
        self._docs_writer.write_table(table)

    def write_queries(self, queries_df: pd.DataFrame) -> None:
        queries_df.to_parquet(self.queries_path, index=False)

    def close(self) -> None:
        if self._docs_writer is not None:
            self._docs_writer.close()
            self._docs_writer = None
