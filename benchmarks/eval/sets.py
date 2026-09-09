"""Named evaluation sets. A set is a question, not a pile of adapters.

Pass the set name to ``--dataset``. ``python -m benchmarks.eval --list-sets``
prints this table. Narrative: ``docs/benchmarks.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.eval.delegation.manifest import SLOTS
from benchmarks.eval.registry import Registry

COMPARE = ("vanilla", "rlmflow-local", "official-rlm")
TASK_GRAPH = tuple(slot.dataset for slot in SLOTS)


@dataclass(frozen=True)
class EvalSet:
    question: str
    datasets: tuple[str, ...]
    runners: tuple[str, ...] = COMPARE


SETS: dict[str, EvalSet] = {
    "smoke": EvalSet(
        "Does the harness run end to end?",
        ("synthetic_needle",),
        ("fake", "vanilla", "rlmflow-local"),
    ),
    "reasoning": EvalSet(
        "Can the agent solve hard problems without long context?",
        ("official_aime_2025", "official_sudoku_extreme"),
    ),
    "long-context": EvalSet(
        "Does the recursive loop beat a vanilla pass on long documents?",
        ("official_sniah", "oolong", "oolong-pairs", "official_codeqa"),
    ),
    "delegation": EvalSet(
        "Does the agent pick the cheapest correct route: local REPL, batched query, or spawn?",
        ("delegation-routing",),
        ("rlmflow-local", "official-rlm"),
    ),
    "task-graph": EvalSet(
        "Frozen 20-problem telemetry across eleven sources.",
        TASK_GRAPH,
        ("rlmflow-local",),
    ),
    "research": EvalSet(
        "Can the agent answer deep-research QA over a fixed document collection?",
        ("official_browsecomp",),
    ),
    "code": EvalSet(
        "Can the agent write code that passes hidden tests?",
        ("official_livecodebench",),
    ),
}

COMPAT_ALIASES: dict[str, tuple[str, ...]] = {
    "needle": ("synthetic_needle",),
    "rlm-core": (
        "official_sniah",
        "official_aime_2025",
        "official_sudoku_extreme",
        "oolong",
        "official_codeqa",
    ),
    "delegation-suite": TASK_GRAPH,
    "delegation-suite-phase1": (
        "delegation_parallelqa",
        "delegation_musique",
        "delegation_twowiki",
        "delegation_dabstep",
        "delegation_natural_plan",
        "delegation_codeqa",
        "delegation_sudoku",
    ),
}


def register_sets(registry: Registry) -> None:
    for name, spec in SETS.items():
        registry.alias(name, spec.datasets)
    for name, datasets in COMPAT_ALIASES.items():
        registry.alias(name, datasets)
    for name in (*SETS, *COMPAT_ALIASES):
        registry.expand([name])


def runners_for(dataset_names: list[str]) -> tuple[str, ...]:
    """Default runners when every token is a named set."""
    specs = [SETS[name] for name in dataset_names if name in SETS]
    if specs and len(specs) == len(dataset_names):
        runners: list[str] = []
        for spec in specs:
            for runner in spec.runners:
                if runner not in runners:
                    runners.append(runner)
        return tuple(runners)
    return ("rlmflow-local",)


def format_sets_help() -> str:
    width = max(len(name) for name in SETS)
    lines = [
        "Named eval sets. Pass a set name to --dataset.",
        "See docs/benchmarks.md.",
        "",
    ]
    for name, spec in SETS.items():
        pad = " " * width
        lines.append(f"{name:<{width}}  {spec.question}")
        lines.append(f"{pad}  datasets: {' '.join(spec.datasets)}")
        lines.append(f"{pad}  runners:  {' '.join(spec.runners)}")
        lines.append("")
    lines.append("Compatibility aliases: " + ", ".join(COMPAT_ALIASES) + ".")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "COMPAT_ALIASES",
    "EvalSet",
    "SETS",
    "TASK_GRAPH",
    "format_sets_help",
    "register_sets",
    "runners_for",
]
