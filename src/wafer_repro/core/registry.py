from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small name-to-object registry used by experiment components."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, value: T | None = None):
        normalized = key.lower()

        def decorator(obj: T) -> T:
            if normalized in self._items:
                raise KeyError(f"{normalized!r} is already registered in {self.name}.")
            self._items[normalized] = obj
            return obj

        if value is not None:
            return decorator(value)
        return decorator

    def get(self, key: str) -> T:
        normalized = key.lower()
        if normalized not in self._items:
            choices = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.name} {key!r}. Available: {choices}")
        return self._items[normalized]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


Factory = Callable[..., T]

