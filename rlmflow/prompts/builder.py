"""Prompt builder: ordered list of named sections, immutable fluent API.

``.section()`` returns a **new** ``PromptBuilder`` — the original is never
modified.  This makes it safe to keep a module-level ``DEFAULT_BUILDER`` and
derive from it.

Usage::

    builder = (
        PromptBuilder()
        .section("role", "You are a helpful agent.", title="Role")
        .section("tools", title="Tools")  # placeholder
    )

    prompt = builder.build(tools="- read_file(path): Read a file.")

    # Derive without mutating the original:
    custom = builder.section("role", "You are a security auditor.", title="Role")

    # Or, to swap only the body of an existing section (keeps title/level/position):
    custom = builder.update("role", "You are a security auditor.")
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from typing import Any

SectionBody = str | Callable[[Any, Any], str]


class Section:
    """One named section of a prompt."""

    __slots__ = ("name", "body", "title", "level")

    def __init__(
        self,
        name: str,
        body: SectionBody = "",
        *,
        title: str | None = None,
        level: int = 2,
    ) -> None:
        self.name = name
        self.body = body
        self.title = title
        self.level = level

    def render(
        self,
        engine: Any = None,
        graph: Any = None,
        body_override: str | None = None,
    ) -> str:
        body = body_override if body_override is not None else self.body
        text = body(engine, graph) if callable(body) else body
        text = textwrap.dedent(text).strip()
        if not text:
            return ""
        if self.title:
            heading = "#" * max(self.level, 1) + " " + self.title
            return heading + "\n\n" + text
        return text


class PromptBuilder:
    """Ordered list of sections with an immutable fluent API.

    ``.section()`` returns a **new** builder — the original is never mutated.
    ``build()`` renders sections top-to-bottom, skipping empties.  Pass
    keyword arguments to ``build()`` to override section bodies for that
    single render.
    """

    def __init__(self) -> None:
        self._sections: list[Section] = []

    def _copy(self) -> PromptBuilder:
        new = PromptBuilder()
        new._sections = list(self._sections)
        return new

    def section(
        self,
        name: str,
        body: SectionBody = "",
        *,
        title: str | None = None,
        level: int = 2,
        before: str | None = None,
        after: str | None = None,
    ) -> PromptBuilder:
        """Add or replace a named section. Returns a new builder."""
        out = self._copy()
        new = Section(name, body, title=title, level=level)

        for i, s in enumerate(out._sections):
            if s.name == name:
                out._sections[i] = new
                return out

        if before:
            for i, s in enumerate(out._sections):
                if s.name == before:
                    out._sections.insert(i, new)
                    return out
        if after:
            for i, s in enumerate(out._sections):
                if s.name == after:
                    out._sections.insert(i + 1, new)
                    return out

        out._sections.append(new)
        return out

    def update(self, name: str, body: SectionBody) -> PromptBuilder:
        """Replace the body of an existing section, preserving title/level/position.

        Raises ``KeyError`` if no section with ``name`` exists — use
        ``.section()`` to add a new one.
        """
        out = self._copy()
        for i, s in enumerate(out._sections):
            if s.name == name:
                out._sections[i] = Section(name, body, title=s.title, level=s.level)
                return out
        raise KeyError(f"no section named {name!r}; use .section() to add it")

    def remove(self, name: str) -> PromptBuilder:
        """Remove a section by name. Returns a new builder."""
        out = self._copy()
        out._sections = [s for s in out._sections if s.name != name]
        return out

    @property
    def names(self) -> list[str]:
        """Section names in current order."""
        return [s.name for s in self._sections]

    def get(self, name: str) -> Section | None:
        for s in self._sections:
            if s.name == name:
                return s
        return None

    def build(self, engine: Any = None, graph: Any = None, **overrides: str) -> str:
        """Render all sections in order, skip empties.

        Callable sections receive ``(engine, graph)``. Keyword arguments
        override section bodies for this call only.
        """
        parts = []
        for s in self._sections:
            override = overrides.get(s.name)
            rendered = s.render(engine, graph, override)
            if rendered.strip():
                parts.append(rendered)
        text = "\n\n".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text + "\n" if text else ""


__all__ = ["PromptBuilder", "Section", "SectionBody"]
