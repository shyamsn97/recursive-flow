"""Prompt and control-loop conditions for delegation experiments."""

from __future__ import annotations

from enum import StrEnum

from rlmflow import AgentStart, ErrorOutput, ExecOutput, Flow, UserQuery
from rlmflow.engine.steps import child_returned, complete


class DelegationCondition(StrEnum):
    LOCAL = "local"
    CAPABILITY_ONLY = "capability_only"
    CURRENT_POLICY = "current_policy"

    @classmethod
    def parse(cls, value: str) -> DelegationCondition:
        try:
            return cls(value.replace("-", "_").lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown delegation condition {value!r}; choose {choices}") from exc


def system_prompt_for(condition: DelegationCondition | str) -> None:
    """Capability-only no longer swaps the prompt; it only skips ``PlanQuery``."""
    if isinstance(condition, str):
        DelegationCondition.parse(condition)
    return None


def apply_condition(flow: Flow, condition: DelegationCondition | str) -> None:
    condition = DelegationCondition.parse(condition) if isinstance(condition, str) else condition
    if condition is DelegationCondition.CAPABILITY_ONLY:

        @flow.transitions.on(AgentStart, ErrorOutput, ExecOutput)
        @flow.transitions.on(UserQuery, when=child_returned)
        async def skip_plan(flow: Flow, node):
            return await complete(flow, node)


__all__ = [
    "DelegationCondition",
    "apply_condition",
    "system_prompt_for",
]
