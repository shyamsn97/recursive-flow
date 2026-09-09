"""System prompt building and chat-message projection."""

from rlmflow.prompts.messages import (
    RenderFn,
    build_messages,
    default_render,
    format_transition_footer,
    render_tools,
    transition_footer,
)
from rlmflow.prompts.prompts import (
    MAX_STATIC_PROMPT_CHARS,
    RLM_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    PromptBuilder,
    PromptProfile,
    SystemPromptFn,
    SystemPromptSource,
    as_system_prompt_fn,
)

__all__ = [
    "MAX_STATIC_PROMPT_CHARS",
    "RLM_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "PromptBuilder",
    "PromptProfile",
    "RenderFn",
    "SystemPromptFn",
    "SystemPromptSource",
    "as_system_prompt_fn",
    "build_messages",
    "default_render",
    "format_transition_footer",
    "render_tools",
    "transition_footer",
]
