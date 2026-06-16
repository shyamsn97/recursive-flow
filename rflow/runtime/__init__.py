"""Runtimes and backends: where an agent's code runs.

A :class:`~rflow.runtime.runtime.Runtime` is the user-facing object you pass to
:class:`rflow.flow.Flow` (``Flow(runtime=...)``); it owns the working directory
and registered tools, and mints one :class:`ReplBackend` per agent. The default
is :class:`LocalRuntime` (in-process). :class:`DockerRuntime` and the cloud
sandbox runtimes (Modal, E2B, Daytona) run code in isolated containers.

Provider backends are imported lazily via ``__getattr__`` so their optional SDKs
are only required when you actually reference the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rflow.runtime.docker import DockerRepl, DockerRuntime, build_argv
from rflow.runtime.runtime import (
    LocalRuntime,
    RemoteRepl,
    ReplBackend,
    Runtime,
    deserialize,
    parse_response,
    serialize,
)

if TYPE_CHECKING:
    from rflow.runtime.sandbox.daytona import DaytonaRepl, DaytonaRuntime
    from rflow.runtime.sandbox.e2b import E2BRepl, E2BRuntime
    from rflow.runtime.sandbox.modal import ModalRepl, ModalRuntime
    from rflow.runtime.sandbox.remote import RemoteFileRuntime

_LAZY = {
    "RemoteFileRuntime": ("rflow.runtime.sandbox.remote", "RemoteFileRuntime"),
    "ModalRepl": ("rflow.runtime.sandbox.modal", "ModalRepl"),
    "ModalRuntime": ("rflow.runtime.sandbox.modal", "ModalRuntime"),
    "E2BRepl": ("rflow.runtime.sandbox.e2b", "E2BRepl"),
    "E2BRuntime": ("rflow.runtime.sandbox.e2b", "E2BRuntime"),
    "DaytonaRepl": ("rflow.runtime.sandbox.daytona", "DaytonaRepl"),
    "DaytonaRuntime": ("rflow.runtime.sandbox.daytona", "DaytonaRuntime"),
}

__all__ = [
    "DaytonaRepl",
    "DaytonaRuntime",
    "DockerRepl",
    "DockerRuntime",
    "E2BRepl",
    "E2BRuntime",
    "LocalRuntime",
    "ModalRepl",
    "ModalRuntime",
    "RemoteFileRuntime",
    "RemoteRepl",
    "ReplBackend",
    "Runtime",
    "build_argv",
    "deserialize",
    "parse_response",
    "serialize",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)
