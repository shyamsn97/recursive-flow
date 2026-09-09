from __future__ import annotations

from benchmarks.eval import DATASETS
from benchmarks.eval.run import build_parser, config_from_args, main
from benchmarks.eval.sets import COMPAT_ALIASES, SETS, TASK_GRAPH, format_sets_help


def test_named_sets_expand_to_their_datasets():
    import benchmarks.eval.tasks  # noqa: F401

    for name, spec in SETS.items():
        assert DATASETS.expand([name]) == list(spec.datasets)


def test_compat_aliases_still_expand():
    import benchmarks.eval.tasks  # noqa: F401

    assert DATASETS.expand(["delegation-suite"]) == list(TASK_GRAPH)
    assert DATASETS.expand(["needle"]) == ["synthetic_needle"]
    for name, datasets in COMPAT_ALIASES.items():
        assert DATASETS.expand([name]) == list(datasets)


def test_reasoning_and_long_context_cli_expand():
    reasoning = config_from_args(
        build_parser().parse_args(["--dataset", "reasoning", "--model", "fake"])
    )
    long_context = config_from_args(
        build_parser().parse_args(
            ["--dataset", "long-context", "--runner", "fake", "--model", "fake"]
        )
    )
    delegation = config_from_args(
        build_parser().parse_args(
            ["--dataset", "delegation", "--runner", "fake", "--model", "fake"]
        )
    )

    assert [spec.name for spec in reasoning.datasets] == list(SETS["reasoning"].datasets)
    assert [spec.name for spec in reasoning.runners] == list(SETS["reasoning"].runners)
    assert [spec.name for spec in long_context.datasets] == list(SETS["long-context"].datasets)
    assert [spec.name for spec in delegation.datasets] == ["delegation-routing"]


def test_list_sets_prints_the_table_and_exits(capsys):
    assert main(["--list-sets"]) == 0
    out = capsys.readouterr().out
    assert out == format_sets_help()
    for name in SETS:
        assert name in out
    assert "Compatibility aliases:" in out
