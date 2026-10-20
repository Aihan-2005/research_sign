
from __future__ import annotations
from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, group: str):
        self.group = group
        self._builders: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]) -> None:
        if not name or not callable(builder):
            raise ValueError("Registry entry needs name and callable")
        if name in self._builders:
            raise ValueError(f"Duplicate {self.group}: {name}")
        self._builders[name] = builder

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            builder = self._builders[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {self.group} '{name}'; available: {self.options}"
            ) from exc
        return builder(**kwargs)

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))
