"""Shared storage backends for workspace session/context data."""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Store(ABC):
    """Backend-neutral object store rooted at one workspace."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return whether ``path`` exists in the store."""

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read UTF-8 text from ``path``."""

    @abstractmethod
    def write_text(self, path: str, value: str) -> None:
        """Write UTF-8 text to ``path``."""

    @abstractmethod
    def append_text(self, path: str, value: str) -> None:
        """Append UTF-8 text to ``path``."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List stored paths under ``prefix``."""

    @abstractmethod
    def remove(self, path: str, *, recursive: bool = False) -> None:
        """Remove ``path`` if it exists."""

    @abstractmethod
    def fork(self, new_location: object) -> Store:
        """Return a deep copy of this store."""

    def read_json(self, path: str) -> Any:
        return json.loads(self.read_text(path))

    def write_json(self, path: str, value: Any) -> None:
        self.write_text(path, json.dumps(value, indent=2))

    def append_jsonl(self, path: str, value: Any) -> None:
        if hasattr(value, "model_dump_json"):
            line = value.model_dump_json()
        else:
            line = json.dumps(value, default=str)
        self.append_text(path, line + "\n")

    def read_jsonl(self, path: str) -> list[Any]:
        if not self.exists(path):
            return []
        return [
            json.loads(line)
            for line in self.read_text(path).splitlines()
            if line.strip()
        ]


class FileStore(Store):
    """Local filesystem implementation of :class:`Store`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, path: str) -> Path:
        return self.root / path

    def exists(self, path: str) -> bool:
        return self.path(path).exists()

    def read_text(self, path: str) -> str:
        return self.path(path).read_text(encoding="utf-8")

    def write_text(self, path: str, value: str) -> None:
        out = self.path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(value, encoding="utf-8")

    def append_text(self, path: str, value: str) -> None:
        out = self.path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(value)

    def list(self, prefix: str = "") -> list[str]:
        base = self.path(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        return sorted(
            str(path.relative_to(self.root))
            for path in base.rglob("*")
            if path.is_file()
        )

    def remove(self, path: str, *, recursive: bool = False) -> None:
        target = self.path(path)
        if not target.exists():
            return
        if target.is_dir():
            if not recursive:
                target.rmdir()
                return
            shutil.rmtree(target)
            return
        target.unlink()

    def fork(self, new_location: object) -> Store:
        dst = Path(new_location).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        if self.root.exists():
            shutil.copytree(self.root, dst)
        else:
            dst.mkdir(parents=True)
        return FileStore(dst)


class MemoryStore(Store):
    """In-memory :class:`Store` useful for tests and future non-file backends."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def exists(self, path: str) -> bool:
        return path in self.values

    def read_text(self, path: str) -> str:
        return self.values[path]

    def write_text(self, path: str, value: str) -> None:
        self.values[path] = value

    def append_text(self, path: str, value: str) -> None:
        self.values[path] = self.values.get(path, "") + value

    def list(self, prefix: str = "") -> list[str]:
        return sorted(path for path in self.values if path.startswith(prefix))

    def remove(self, path: str, *, recursive: bool = False) -> None:
        if recursive:
            prefix = path.rstrip("/") + "/"
            for key in list(self.values):
                if key == path or key.startswith(prefix):
                    del self.values[key]
            return
        self.values.pop(path, None)

    def fork(self, new_location: object) -> Store:
        out = MemoryStore()
        out.values = dict(self.values)
        return out


def resolve_backend(root: Store | str | Path) -> tuple[Store, Path | None]:
    """Resolve a ``Store`` or workspace root path into ``(store, root)``."""
    if isinstance(root, Store):
        return root, getattr(root, "root", None)

    path = Path(root).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return FileStore(path), path


def copy_workspace_paths(
    store: Store,
    new_location: object,
    paths: tuple[str, ...],
) -> Store:
    """Copy a subset of files/directories from a workspace ``Store``.

    For ``FileStore`` backends, copies only the named relative paths into the
    new location and returns a ``FileStore`` rooted there. For other backends,
    falls back to the store's own ``fork`` (a deep copy).
    """
    if not isinstance(store, FileStore):
        return store.fork(new_location)

    dst = Path(new_location).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for rel in paths:
        src = store.path(rel)
        if not src.exists():
            continue
        out = dst / rel
        if src.is_dir():
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(src, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    return FileStore(dst)


__all__ = [
    "FileStore",
    "MemoryStore",
    "Store",
    "copy_workspace_paths",
    "resolve_backend",
]
