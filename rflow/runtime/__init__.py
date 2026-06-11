from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DaytonaRuntime": "rflow.runtime.sandbox.daytona",
    "DockerRuntime": "rflow.runtime.docker",
    "E2BRuntime": "rflow.runtime.sandbox.e2b",
    "LocalRuntime": "rflow.runtime.local",
    "ModalRuntime": "rflow.runtime.sandbox.modal",
    "RemoteFileRuntime": "rflow.runtime.sandbox.remote",
    "Runtime": "rflow.runtime.runtime",
    "ToolDef": "rflow.runtime.runtime",
}

__all__ = [
    "DaytonaRuntime",
    "DockerRuntime",
    "E2BRuntime",
    "LocalRuntime",
    "ModalRuntime",
    "RemoteFileRuntime",
    "Runtime",
    "ToolDef",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
