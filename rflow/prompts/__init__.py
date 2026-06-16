"""Prompt content + the immutable :class:`PromptBuilder` that assembles it.

``DEFAULT_BUILDER`` is the live builder rendered by ``Flow.build_system_prompt``;
``SYSTEM_PROMPT`` is its static (agent-independent) render, used as the default
``Flow.system_prompt`` fallback. Derive custom prompts without mutating the
original via ``DEFAULT_BUILDER.update("role", ...)`` / ``.section(...)``.
"""

from __future__ import annotations

from rflow.prompts.builder import PromptBuilder, Section, SectionBody
from rflow.prompts.default import DEFAULT_BUILDER, SYSTEM_PROMPT

__all__ = [
    "DEFAULT_BUILDER",
    "PromptBuilder",
    "SYSTEM_PROMPT",
    "Section",
    "SectionBody",
]
