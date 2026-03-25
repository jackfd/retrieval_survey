from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    ok: bool
    value: Optional[T] = None
    error_message: str = ""

    @classmethod
    def success(cls, value: Optional[T] = None) -> "Result[T]":
        return cls(ok=True, value=value, error_message="")

    @classmethod
    def failure(cls, error_message: str) -> "Result[T]":
        return cls(ok=False, value=None, error_message=error_message)
