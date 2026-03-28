from dataclasses import dataclass


@dataclass
class SelectorConfig:
    batch_size: int = 64
    hard_max_tokens: int = 8092
    target_tokens: int = 3200
    min_independent_tokens: int = 500
    top_n: int = 3
    mmr_lambda: float = 0.7
