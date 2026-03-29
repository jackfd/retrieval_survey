from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

DOC_COLUMNS = [
    "doc_id",
    "chunk_id",
    "chunk_text",
    "chunk_vector",
    "chunk_score",
    "chunk_rank",
]
QUERY_COLUMNS = ["query_id", "query_text", "query_embedding"]
LOG_EVERY_N = 1000
DOC_FLUSH_CHUNK_THRESHOLD = 20000


@dataclass
class ExperimentConfig:
    embedding_dim: int
    max_length: int
    query_prefix: str
    doc_prefix: str
    instruction_template: str


@dataclass
class InferenceConfig:
    batch_size: int
    device: str
    embedding_api_url: str
    http_timeout: float
    http_max_retries: int


@dataclass
class ModelConfig:
    provider: str
    model_id: str


@dataclass
class BuilderConfig:
    experiment: ExperimentConfig
    inference: InferenceConfig
    model: ModelConfig
    raw_config: Dict[str, Any]


@dataclass
class DatasetContext:
    dataset_root: Path
    resolved_dataset_dir: Path
    dataset_meta: Dict[str, Any]
    docs_path: Path
    queries_path: Path
