"""User-controlled workspace artifacts.

Artifacts are ordinary files under the workspace root, addressed by
workspace-relative paths chosen by the user. The API intentionally does not
force an ``artifacts/`` directory; it only protects coordinator-owned engine
state such as ``session/`` and ``context/``.
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any

from rlmflow.workspace.store import Store, resolve_backend

_RESERVED_ROOTS: tuple[str, ...] = (
    "graph.json",
    "session",
    "context",
    "trace",
    "checkpoint.json",
)


class ArtifactStore:
    """Safe root-relative file/object API for user workspace artifacts."""

    def __init__(self, root: Store | str | Path) -> None:
        self.store, self.root = resolve_backend(root)

    def exists(self, path: str | Path) -> bool:
        """Return whether a user artifact exists at ``path``."""

        return self.store.exists(self._artifact_path(path))

    def read_text(self, path: str | Path) -> str:
        """Read a UTF-8 artifact from a workspace-relative path."""

        return self.store.read_text(self._artifact_path(path))

    def write_text(self, path: str | Path, value: str) -> None:
        """Write a UTF-8 artifact to a workspace-relative path."""

        self.store.write_text(self._artifact_path(path), value)

    def append_text(self, path: str | Path, value: str) -> None:
        """Append UTF-8 text to a workspace-relative artifact."""

        self.store.append_text(self._artifact_path(path), value)

    def read_json(self, path: str | Path) -> Any:
        """Read a JSON artifact from a workspace-relative path."""

        return self.store.read_json(self._artifact_path(path))

    def write_json(self, path: str | Path, value: Any) -> None:
        """Write a JSON artifact to a workspace-relative path."""

        self.store.write_json(self._artifact_path(path), value)

    def list(self, prefix: str | Path = "") -> list[str]:
        """List user artifact files under ``prefix``.

        Engine-owned paths are hidden from the top-level listing and rejected if
        requested directly.
        """

        rel = self._artifact_prefix(prefix)
        return [path for path in self.store.list(rel) if not self._reserved(path)]

    def _artifact_path(self, path: str | Path) -> str:
        rel = self._normalize(path, allow_empty=False)
        if self._reserved(rel):
            raise ValueError(f"{rel!r} is reserved for rlmflow engine state")
        return rel

    def _artifact_prefix(self, prefix: str | Path) -> str:
        rel = self._normalize(prefix, allow_empty=True)
        if rel and self._reserved(rel):
            raise ValueError(f"{rel!r} is reserved for rlmflow engine state")
        return rel

    @staticmethod
    def _normalize(path: str | Path, *, allow_empty: bool) -> str:
        text = path.as_posix() if isinstance(path, Path) else str(path)
        text = text.strip()
        if not text:
            if allow_empty:
                return ""
            raise ValueError("artifact path must be non-empty")
        if text.startswith("/"):
            raise ValueError("artifact path must be relative to the workspace")
        parts = [part for part in text.replace("\\", "/").split("/") if part]
        if any(part == ".." for part in parts):
            raise ValueError("artifact path must not contain '..'")
        rel = posixpath.normpath("/".join(parts))
        if rel == ".":
            if allow_empty:
                return ""
            raise ValueError("artifact path must be non-empty")
        return rel

    @staticmethod
    def _reserved(path: str) -> bool:
        return any(
            path == root or path.startswith(root + "/") for root in _RESERVED_ROOTS
        )


__all__ = ["ArtifactStore"]
