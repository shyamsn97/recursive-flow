"""Pinned subsets of the task-graph adapters for routing and iteration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from benchmarks.eval import dataset
from benchmarks.eval.tasks.arc_agi import ArcAgiTaskGraphDataset
from benchmarks.eval.tasks.dabstep import DABstepTaskGraphDataset
from benchmarks.eval.tasks.delegation_codeqa import DelegationCodeQADataset
from benchmarks.eval.tasks.delegation_sudoku import DelegationSudokuDataset
from benchmarks.eval.tasks.musique import MuSiQueTaskGraphDataset
from benchmarks.eval.tasks.natural_plan import NaturalPlanTaskGraphDataset
from benchmarks.eval.tasks.oolong_pairs import PAPER_CONTEXT_LEN, OolongPairsDataset
from benchmarks.eval.tasks.parallelqa import ParallelQATaskGraphDataset
from benchmarks.eval.tasks.planbench import PlanBenchTaskGraphDataset
from benchmarks.eval.tasks.twowiki import TwoWikiTaskGraphDataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FIVE_TASKS = (
    ("delegation_parallelqa_22", "local_lookup_control"),
    ("delegation_parallelqa_94", "parallel_lookup"),
    ("delegation_planbench_16_logistics_instance_20", "candidate_and_verification"),
    ("delegation_sudoku_18_cross-product", "local_constraint_control"),
    ("delegation_arc_agi_20_2ba387bc", "parallel_analysis_and_synthesis"),
)
TEN_PAIR_IDS = ("4", "16")
TEN_TASKS = (
    ("delegation_parallelqa_11", "local_numeric_control", "local"),
    ("delegation_parallelqa_22", "local_lookup_control", "local"),
    ("delegation_parallelqa_63", "local_arithmetic_control", "local"),
    ("delegation_parallelqa_94", "parallel_lookup", "local"),
    ("delegation_musique_05_4hop1__38130_8966_31714_79432", "multi_hop_retrieval", "subagent"),
    ("delegation_twowiki_07_948c33ea0baf11ebab90acde48001122", "sequential_inference", "subagent"),
    ("delegation_twowiki_08_d594f50208c111ebbd8bac1f6bf848b6", "parallel_evidence", "subagent"),
    ("delegation_planbench_16_logistics_instance_20", "candidate_and_verification", "local"),
    ("delegation_codeqa_17_official_codeqa_00072", "long_context_isolation", "batched_query"),
    ("delegation_sudoku_18_cross-product", "local_constraint_control", "local"),
    ("delegation_arc_agi_19_1ae2feb7", "parallel_analysis_and_synthesis", "local"),
    ("delegation_arc_agi_20_2ba387bc", "parallel_analysis_and_synthesis", "local"),
    ("delegation_dabstep_11_2536", "document_and_recompute", "subagent"),
    ("delegation_natural_plan_13_trip_planning_example_593", "coupled_planning", "subagent"),
    (f"oolong_pairs_{PAPER_CONTEXT_LEN}_04", "batched_semantic_date_filter", "batched_query"),
    (f"oolong_pairs_{PAPER_CONTEXT_LEN}_16", "batched_asymmetric_roles", "batched_query"),
)
REGRESSION_TASKS = (
    ("delegation_parallelqa_63", "authoritative_tool_selection"),
    ("delegation_musique_05_4hop1__38130_8966_31714_79432", "child_source_verification"),
    ("delegation_twowiki_08_d594f50208c111ebbd8bac1f6bf848b6", "structured_result"),
    ("delegation_planbench_16_logistics_instance_20", "candidate_and_verification"),
    ("delegation_arc_agi_20_2ba387bc", "recursive_budget_control"),
)
ROUTING_PAIR_IDS = ("4", "11", "16", "20")
ROUTING_TASKS = (
    ("oolong_pairs_8192_04", "batched_semantic_date_filter", "batched_query"),
    ("oolong_pairs_8192_11", "batched_asymmetric_roles", "batched_query"),
    ("oolong_pairs_8192_16", "batched_complex_classification", "batched_query"),
    ("oolong_pairs_8192_20", "batched_compound_roles", "batched_query"),
    ("delegation_twowiki_08_d594f50208c111ebbd8bac1f6bf848b6", "parallel_evidence", "subagent"),
    ("delegation_musique_05_4hop1__38130_8966_31714_79432", "multi_hop_retrieval", "subagent"),
    ("delegation_parallelqa_63", "local_arithmetic_control", "local"),
    ("delegation_sudoku_18_cross-product", "local_constraint_control", "local"),
)

_SCORERS = (
    ("oolong_pairs_", OolongPairsDataset),
    ("delegation_parallelqa_", ParallelQATaskGraphDataset),
    ("delegation_musique_", MuSiQueTaskGraphDataset),
    ("delegation_twowiki_", TwoWikiTaskGraphDataset),
    ("delegation_planbench_", PlanBenchTaskGraphDataset),
    ("delegation_codeqa_", DelegationCodeQADataset),
    ("delegation_sudoku_", DelegationSudokuDataset),
    ("delegation_arc_agi_", ArcAgiTaskGraphDataset),
    ("delegation_dabstep_", DABstepTaskGraphDataset),
    ("delegation_natural_plan_", NaturalPlanTaskGraphDataset),
)


class _Subset(Dataset):
    selected_tasks: tuple[tuple[str, ...], ...] = ()

    def __init__(self, sources: tuple[Dataset, ...]) -> None:
        self._datasets = sources

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        available: dict[str, Example] = {}
        for source in self._datasets:
            for example in source.examples(split=split, limit=None, seed=seed):
                available[example.id] = example
        missing = [item[0] for item in self.selected_tasks if item[0] not in available]
        if missing:
            raise ValueError(f"delegation subset is missing tasks: {missing}")
        selected = []
        for example_id, role, *rest in self.selected_tasks:
            metadata = {**available[example_id].metadata, "iteration_role": role}
            if rest:
                metadata["expected_route"] = rest[0]
            selected.append(replace(available[example_id], metadata=metadata))
        return selected if limit is None else selected[:limit]

    def score(self, example: Example, prediction: Prediction) -> Score:
        for prefix, source_type in _SCORERS:
            if example.id.startswith(prefix):
                source = next(
                    (item for item in self._datasets if isinstance(item, source_type)),
                    None,
                )
                if source is not None:
                    return source.score(example, prediction)
        raise ValueError(f"no scorer for iteration task {example.id}")


def _pack(
    name: str,
    class_name: str,
    tasks: tuple[tuple[str, ...], ...],
    sources: Callable[[str], tuple[Dataset, ...]],
    tags: list[str],
):
    class Pack(_Subset):
        selected_tasks = tasks

        def __init__(self, data_dir: str = "evals/data") -> None:
            super().__init__(sources(data_dir))

    Pack.__name__ = class_name
    Pack.__qualname__ = class_name
    return dataset(name, tags=tags)(Pack)


DelegationIterationDataset = _pack(
    "delegation-iteration-five",
    "DelegationIterationDataset",
    FIVE_TASKS,
    lambda d: (
        ParallelQATaskGraphDataset(data_dir=d),
        PlanBenchTaskGraphDataset(data_dir=d),
        DelegationSudokuDataset(),
        ArcAgiTaskGraphDataset(data_dir=d),
    ),
    ["delegation", "iteration", "canary"],
)
DelegationIterationTenDataset = _pack(
    "delegation-iteration-ten",
    "DelegationIterationTenDataset",
    TEN_TASKS,
    lambda d: (
        ParallelQATaskGraphDataset(data_dir=d),
        MuSiQueTaskGraphDataset(data_dir=d),
        TwoWikiTaskGraphDataset(data_dir=d),
        PlanBenchTaskGraphDataset(data_dir=d),
        DelegationCodeQADataset(data_dir=d),
        DelegationSudokuDataset(),
        ArcAgiTaskGraphDataset(data_dir=d),
        DABstepTaskGraphDataset(data_dir=d),
        NaturalPlanTaskGraphDataset(data_dir=d),
        OolongPairsDataset(data_dir=d, context_len=PAPER_CONTEXT_LEN, question_ids=TEN_PAIR_IDS),
    ),
    ["delegation", "iteration"],
)
DelegationRegressionDataset = _pack(
    "delegation-regression-five",
    "DelegationRegressionDataset",
    REGRESSION_TASKS,
    lambda d: (
        ParallelQATaskGraphDataset(data_dir=d),
        MuSiQueTaskGraphDataset(data_dir=d),
        TwoWikiTaskGraphDataset(data_dir=d),
        PlanBenchTaskGraphDataset(data_dir=d),
        ArcAgiTaskGraphDataset(data_dir=d),
    ),
    ["delegation", "iteration", "regression"],
)
DelegationRoutingDataset = _pack(
    "delegation-routing",
    "DelegationRoutingDataset",
    ROUTING_TASKS,
    lambda d: (
        OolongPairsDataset(data_dir=d, context_len=8192, question_ids=ROUTING_PAIR_IDS),
        TwoWikiTaskGraphDataset(data_dir=d),
        MuSiQueTaskGraphDataset(data_dir=d),
        ParallelQATaskGraphDataset(data_dir=d),
        DelegationSudokuDataset(),
    ),
    ["delegation", "iteration", "routing"],
)

__all__ = [
    "DelegationIterationDataset",
    "DelegationIterationTenDataset",
    "DelegationRegressionDataset",
    "DelegationRoutingDataset",
    "FIVE_TASKS",
    "REGRESSION_TASKS",
    "ROUTING_PAIR_IDS",
    "ROUTING_TASKS",
    "TEN_PAIR_IDS",
    "TEN_TASKS",
]
