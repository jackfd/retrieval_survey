from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class RuntimeConfig:
    embedding_dim: int
    normalize_embeddings: bool
    max_length: int
    query_prefix: str
    doc_prefix: str
    instruction_template: str
    batch_size: int
    device: str
    embedding_api_url: str
    http_timeout: float
    http_max_retries: int


@dataclass
class ModelConfig:
    model_name: str
    provider: str
    model_id: str


@dataclass
class BuilderConfig:
    runtime: RuntimeConfig
    model: ModelConfig
    raw_config: Dict[str, Any]


@dataclass
class DatasetContext:
    dataset_root: Path
    resolved_dataset_dir: Path
    dataset_meta: Dict[str, Any]
    docs_path: Path
    queries_path: Path


@dataclass
class ProcessResult:
    output_df: pd.DataFrame
    failures: List[Dict[str, str]]
