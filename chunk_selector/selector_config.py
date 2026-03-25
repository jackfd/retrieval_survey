from dataclasses import dataclass


@dataclass
class SelectorConfig:
    cluster_ratio: float = 2.0
    batch_size: int = 64
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    top_keywords: int = 10
    cooccur_window: int = 4
    request_timeout: float = 10.0
    max_retries: int = 2
