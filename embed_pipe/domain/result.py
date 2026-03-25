from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    ok: bool
    value: Optional[T] = None

    @classmethod
    def success(cls, value: Optional[T] = None) -> "Result[T]":
        return cls(ok=True, value=value)

    @classmethod
    def failure(cls) -> "Result[T]":
        return cls(ok=False, value=None)
