import json

from benchmarks.eval import DATASETS, MODELS, RUNNERS
from benchmarks.eval.loggers.jsonl import load_rows
from benchmarks.eval.metrics import summarize
from benchmarks.eval.run import (
    config_from_args,
    main,
    parse_model_spec,
    parse_params,
    parse_seed_spec,
)
from benchmarks.eval.runners.official_rlm import _parse_result, _render_context
from benchmarks.eval.types import Example, Prediction, Row, Score


def test_seed_spec_parser() -> None:
    assert parse_seed_spec("0:3") == [0, 1, 2]
    assert parse_seed_spec("0:6:2") == [0, 2, 4]
    assert parse_seed_spec("3,7,11") == [3, 7, 11]


def test_component_registries_and_params() -> None:
    assert DATASETS.expand(["smoke"]) == ["synthetic_needle"]
    assert {"synthetic_needle", "oolong"}.issubset(DATASETS.names())
    assert {"fake", "vanilla", "rflow-local", "official-rlm"}.issubset(RUNNERS.names())
    assert {"fake", "openai", "anthropic"}.issubset(MODELS.names())
    assert parse_model_spec("openai:gpt-5-mini").provider == "openai"
    assert parse_model_spec("fake").name == "fake"
    assert parse_params(["synthetic_needle.records=8", "x=true"]) == {
        "synthetic_needle": {"records": 8},
        "_": {"x": True},
    }


def test_official_runner_helpers() -> None:
    official_example = Example(
        id="oolong_00000",
        prompt="question",
        context={"context": "clean context"},
        expected="needle",
    )
    synthetic_example = Example(
        id="synthetic_needle_0000",
        prompt="question",
        context={"haystack": "records"},
        expected="needle",
    )

    assert _render_context(official_example) == official_example.context["context"]
    assert _render_context(synthetic_example).startswith("INPUT haystack:")
    assert _parse_result('noise\n<<<RESULT>>>\n{"response": "needle", "iterations": 2}') == {
        "response": "needle",
        "iterations": 2,
    }


def test_summary() -> None:
    row = Row(
        run_id="run",
        dataset="synthetic_needle",
        example_id="synthetic_needle_0000",
        seed=0,
        runner="vanilla",
        model="fake",
        prediction=Prediction(answer="needle"),
        score=Score(value=1.0, correct=True),
    )
    summary = summarize([row])
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["by_runner"]["vanilla"]["score"] == 1.0
    assert summary["by_dataset"]["synthetic_needle"]["score"] == 1.0


def test_fake_cli_smoke(tmp_path) -> None:
    exit_code = main(
        [
            "--model",
            "fake",
            "--dataset",
            "synthetic_needle",
            "--runner",
            "fake",
            "vanilla",
            "rflow-local",
            "--seeds",
            "0:2",
            "--dataset-param",
            "synthetic_needle.records=8",
            "--dataset-param",
            "synthetic_needle.filler_words=2",
            "--runner-param",
            "rflow-local.max_iters=3",
            "--runner-param",
            "rflow-local.max_depth=1",
            "--runner-param",
            "rflow-local.live_save=false",
            "--out-dir",
            str(tmp_path),
            "--run-id",
            "smoke",
        ]
    )
    assert exit_code == 0
    rows_path = tmp_path / "smoke" / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == 6
    assert [(row["seed"], row["runner"]) for row in rows] == [
        (0, "fake"),
        (0, "vanilla"),
        (0, "rflow-local"),
        (1, "fake"),
        (1, "vanilla"),
        (1, "rflow-local"),
    ]
    assert all(row["score"]["correct"] for row in rows)
    assert (
        tmp_path
        / "smoke"
        / "artifacts"
        / "synthetic_needle"
        / "synthetic_needle_0000"
        / "vanilla"
        / "prediction.json"
    ).exists()
    assert (tmp_path / "smoke" / "summary.json").exists()
    assert (tmp_path / "smoke" / "report.md").exists()
    loaded = load_rows(rows_path)
    assert loaded[0].dataset == "synthetic_needle"


def test_fake_cli_resume_skips_existing_rows(tmp_path) -> None:
    args = [
        "--model",
        "fake",
        "--dataset",
        "synthetic_needle",
        "--runner",
        "fake",
        "vanilla",
            "rflow-local",
        "--seeds",
        "0:2",
        "--dataset-param",
        "synthetic_needle.records=8",
        "--dataset-param",
        "synthetic_needle.filler_words=2",
            "--runner-param",
            "rflow-local.max_iters=3",
            "--runner-param",
            "rflow-local.max_depth=1",
            "--runner-param",
            "rflow-local.live_save=false",
        "--out-dir",
        str(tmp_path),
        "--run-id",
        "resume",
    ]

    assert main(args) == 0
    assert main([*args, "--resume"]) == 0

    results_path = tmp_path / "resume" / "rows.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(rows) == 6


def test_config_from_args(tmp_path) -> None:
    parser = __import__("benchmarks.eval.run", fromlist=["build_parser"]).build_parser()
    args = parser.parse_args(
        [
            "--model",
            "fake",
            "--dataset",
            "smoke",
            "--runner",
            "rflow",
            "--runner-param",
            "rflow-local.max_depth=1",
            "--out-dir",
            str(tmp_path),
        ]
    )
    config = config_from_args(args)
    assert config.datasets[0].name == "synthetic_needle"
    assert config.runners[0].name == "rflow-local"
    assert config.runners[0].params["max_depth"] == 1
