class EmbedPipeError(Exception):
    """Base exception for em domain errors."""


class ConfigError(EmbedPipeError):
    """Configuration loading or validation failed."""


class DatasetError(EmbedPipeError):
    """Dataset resolution or validation failed."""


class StrategyBuildError(EmbedPipeError):
    """Embedding strategy construction failed."""


class ProcessingError(EmbedPipeError):
    """Document/query processing failed."""
