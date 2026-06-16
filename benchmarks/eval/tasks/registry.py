"""Gym-style registry for benchmark tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

from benchmarks.eval.core import Score, TaskInstance


class Task(Protocol):
    name: str
    description: str

    def generate(self, seed: int, **kwargs) -> TaskInstance: ...

    def score(self, answer: str, expected: object, metadata: dict) -> Score: ...


@dataclass(frozen=True)
class TaskSpec:
    """Registered task metadata."""

    id: str
    entry_point: type[Task]
    description: str = ""
    tags: tuple[str, ...] = ()
    default_kwargs: dict = field(default_factory=dict)

    def make(self, **kwargs) -> Task:
        params = {**self.default_kwargs, **kwargs}
        return self.entry_point(**params)


class TaskRegistry:
    """Small registry modeled after Gym's env registry.

    It keeps concrete task specs plus named aliases such as ``official`` and
    ``all``. Callers can ``register()``, ``make()``, ``names()``, or
    ``expand()`` task ids before running a sweep.
    """

    def __init__(self) -> None:
        self._specs: dict[str, TaskSpec] = {}
        self._aliases: dict[str, Callable[["TaskRegistry"], Iterable[str]]] = {}

    def register(
        self,
        id: str,
        *,
        description: str | None = None,
        tags: Iterable[str] = (),
        default_kwargs: dict | None = None,
    ):
        """Register a task class under a CLI id."""

        def decorator(cls: type[Task]) -> type[Task]:
            cls.name = id
            resolved_tags = tuple(tags)
            if not resolved_tags and id.startswith("official_"):
                resolved_tags = ("official",)
            self._specs[id] = TaskSpec(
                id=id,
                entry_point=cls,
                description=description or getattr(cls, "description", ""),
                tags=resolved_tags,
                default_kwargs=default_kwargs or {},
            )
            return cls

        return decorator

    def alias(
        self, name: str, resolver: Callable[["TaskRegistry"], Iterable[str]]
    ) -> None:
        """Register a named expansion like ``official`` or ``all``."""

        self._aliases[name] = resolver

    def make(self, id: str, **kwargs) -> Task:
        spec = self.spec(id)
        return spec.make(**kwargs)

    def spec(self, id: str) -> TaskSpec:
        if id not in self._specs:
            available = ", ".join(self.names())
            raise ValueError(f"unknown task {id!r}. available: {available}")
        return self._specs[id]

    def names(self, *, tags: Iterable[str] | None = None) -> list[str]:
        if tags is None:
            return sorted(self._specs)
        wanted = set(tags)
        return sorted(
            task_id
            for task_id, spec in self._specs.items()
            if wanted.intersection(spec.tags)
        )

    def specs(self) -> dict[str, TaskSpec]:
        return dict(self._specs)

    def expand(self, values: Iterable[str]) -> list[str]:
        expanded: list[str] = []
        for value in values:
            for raw in str(value).split(","):
                name = raw.strip()
                if not name:
                    continue
                if name in self._aliases:
                    expanded.extend(self._aliases[name](self))
                else:
                    expanded.append(name)
        unknown = [name for name in expanded if name not in self._specs]
        if unknown:
            available = ", ".join(self.names())
            raise ValueError(f"unknown task(s): {', '.join(unknown)}. available: {available}")
        return list(dict.fromkeys(expanded))

    def __contains__(self, id: str) -> bool:
        return id in self._specs

    def __len__(self) -> int:
        return len(self._specs)


TASK_REGISTRY = TaskRegistry()


__all__ = ["TASK_REGISTRY", "Task", "TaskRegistry", "TaskSpec"]
