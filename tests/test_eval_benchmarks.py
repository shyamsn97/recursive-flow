from __future__ import annotations

import json
import sys
import types
from argparse import Namespace

from benchmarks.eval.config import RunConfig
from benchmarks.eval.core import EvalResult, TaskInstance
from benchmarks.eval.metrics import summarize
from benchmarks.eval.run import expand_tasks, main, make_run_id, parse_seed_spec
from benchmarks.eval.runners import get_runner, list_runners
from benchmarks.eval.runners.official import _parse_result, _render_context
from benchmarks.eval.store import RunStore
from benchmarks.eval.tasks import OFFICIAL_TASKS, TASK_REGISTRY, get_task, list_tasks
from benchmarks.eval.tasks import long_context
from benchmarks.eval.wandb_logging import WandbLogger


def test_seed_spec_parser() -> None:
    assert parse_seed_spec("0:3") == [0, 1, 2]
    assert parse_seed_spec("0:6:2") == [0, 2, 4]
    assert parse_seed_spec("3,7,11") == [3, 7, 11]


def test_run_id_uses_aliases_and_stays_short() -> None:
    alias_id = make_run_id(
        Namespace(
            model="gpt-5-mini",
            tasks=["official"],
            runners=["rflow", "vanilla", "official"],
        )
    )
    long_id = make_run_id(
        Namespace(
            model="gpt-5-mini",
            tasks=OFFICIAL_TASKS,
            runners=["rflow", "vanilla", "official"],
        )
    )

    assert "_official_" in alias_id
    assert "official_sniah-official_oolong" not in alias_id
    assert len(alias_id.split("/")[-1]) < 255
    assert len(long_id.split("/")[-1]) < 255


def test_task_and_runner_registries() -> None:
    assert "sniah" in list_tasks()
    assert TASK_REGISTRY.expand(["official"]) == OFFICIAL_TASKS
    assert TASK_REGISTRY.make("sniah", records=8, filler_words=2).name == "sniah"
    assert TASK_REGISTRY.spec("official_sniah").tags == ("official",)
    assert set(OFFICIAL_TASKS).issubset(set(list_tasks()))
    assert expand_tasks(["official"]) == OFFICIAL_TASKS
    assert "sniah" in expand_tasks(["all"])
    assert {"fake", "official", "rflow", "vanilla"}.issubset(set(list_runners()))
    for task_name in OFFICIAL_TASKS:
        assert get_task(task_name).name == task_name
    task = get_task("sniah")
    instance = task.generate(7, records=8, filler_words=2)
    assert instance.expected in instance.inputs["haystack"]
    assert task.score(str(instance.expected), instance.expected, instance.metadata).correct
    assert get_runner("fake").run(
        instance,
        client=None,
        model="fake",
        out_dir=None,
        max_iters=1,
        max_depth=0,
        live_save=False,
    ).answer == instance.expected


def test_official_sniah_uses_ruler_plain_config(monkeypatch) -> None:
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {
                "category": "niah_single_1",
                "prompt": "Question: Find the pass key.",
                "extra_info": {"ground_truth": {"answers": ["needle"]}},
            }
        ]

    monkeypatch.setattr(long_context, "_load_dataset", fake_load_dataset)

    task = long_context.OfficialRulerSNIAHTask(max_samples=1)
    instance = task.generate(0)

    assert instance.expected == "needle"
    assert calls[0][1]["config"] == "plain"


def test_official_runner_helpers() -> None:
    official_instance = TaskInstance(
        task_name="official_sniah",
        task_id="official_sniah_0000",
        seed=0,
        prompt="question",
        inputs={"context": "clean context"},
        expected="needle",
    )
    synthetic_instance = TaskInstance(
        task_name="sniah",
        task_id="sniah_0000",
        seed=0,
        prompt="question",
        inputs={"haystack": "records"},
        expected="needle",
    )

    assert _render_context(official_instance) == official_instance.inputs["context"]
    assert _render_context(synthetic_instance).startswith("INPUT haystack:")
    assert _parse_result('noise\n<<<RESULT>>>\n{"response": "needle", "iterations": 2}') == {
        "response": "needle",
        "iterations": 2,
    }


def test_wandb_logger_surfaces_summary_metrics(monkeypatch) -> None:
    logs = []
    summary = {}

    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []

        def add_data(self, *values):
            self.rows.append(values)

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: types.SimpleNamespace(config=kwargs["config"]),
        define_metric=lambda *args, **kwargs: None,
        log=lambda payload: logs.append(payload),
        finish=lambda: None,
        summary=types.SimpleNamespace(update=summary.update),
        Table=FakeTable,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger(
        enabled=True,
        project="proj",
        entity=None,
        run_id="run",
        config={"tasks": ["official_sniah"], "runners": ["vanilla"]},
    )
    row = EvalResult(
        run_id="run",
        task_name="official_sniah",
        task_id="official_sniah_0",
        seed=0,
        runner="vanilla",
        model="fake",
        correct=True,
        score=1.0,
        answer="needle",
        expected="needle",
    )

    logger.log_result(row)
    logger.log_summary(summarize([row]))
    logger.finish()

    assert logs[0]["task"] == "official_sniah"
    assert logs[0]["runner"] == "vanilla"
    assert logs[0]["overall/accuracy"] == 1.0
    assert logs[0]["task_accuracy/official_sniah/vanilla"] == 1.0
    assert summary["overall/accuracy"] == 1.0
    assert summary["by_runner/vanilla/accuracy"] == 1.0
    assert summary["by_task/official_sniah/accuracy"] == 1.0
    assert summary["by_runner_task/vanilla/official_sniah/accuracy"] == 1.0
    assert summary["task_accuracy/official_sniah/vanilla"] == 1.0
    assert summary["tasks_won/vanilla"] == 1
    assert logs[-1]["tables/results"].rows[0][0] == "official_sniah"
    assert logs[-1]["tables/summary"].rows[0][0] == "overall"
    assert logs[-1]["tables/task_accuracy"].rows[0][:3] == (
        "official_sniah",
        "Vanilla",
        1.0,
    )


def test_run_store_roundtrip_and_paths(tmp_path) -> None:
    config = RunConfig(
        run_id="store",
        tasks=("sniah",),
        runners=("fake",),
        seeds=(0,),
        provider="fake",
        model="fake",
        max_iters=1,
        max_depth=0,
        out_dir=tmp_path,
        report_dir=tmp_path / "reports",
        report_name="sniah",
        live_save=False,
        task_params={},
        official_params={},
    )
    store = RunStore(config.root, config=config)
    row = EvalResult(
        run_id="store",
        task_name="sniah",
        task_id="sniah_0000",
        seed=0,
        runner="fake",
        model="fake",
        correct=True,
        score=1.0,
        answer="needle",
        expected="needle",
    )

    store.initialize()
    row.artifacts["result_path"] = str(store.job_result_path("fake", "sniah", "sniah_0000"))
    store.write_job_result(row)
    store.append_result(row)

    assert store.load_results()[0].to_dict() == row.to_dict()
    assert store.has_result("sniah", "fake", 0)
    assert not store.has_result("sniah", "vanilla", 0)
    assert store.artifact_dir("fake", "sniah", "sniah_0000").is_dir()
    assert (config.root / "artifacts" / "fake" / "sniah" / "sniah_0000" / "result.json").exists()


def test_fake_cli_smoke(tmp_path) -> None:
    exit_code = main(
        [
            "--provider",
            "fake",
            "--model",
            "fake",
            "--tasks",
            "sniah",
            "--runners",
            "fake",
            "vanilla",
            "--seeds",
            "0:2",
            "--task-param",
            "records=8",
            "--task-param",
            "filler_words=2",
            "--out-dir",
            str(tmp_path),
            "--report-dir",
            str(tmp_path / "reports"),
            "--run-id",
            "smoke",
            "--no-live-save",
        ]
    )
    assert exit_code == 0
    results_path = tmp_path / "smoke" / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert [(row["seed"], row["runner"]) for row in rows] == [
        (0, "fake"),
        (0, "vanilla"),
        (1, "fake"),
        (1, "vanilla"),
    ]
    assert all(row["correct"] for row in rows)
    assert all(row["artifacts"]["result_path"] for row in rows)
    assert (
        tmp_path / "smoke" / "artifacts" / "vanilla" / "sniah" / "sniah_0000" / "result.json"
    ).exists()
    task_accuracy = json.loads((tmp_path / "smoke" / "task_accuracy.json").read_text())
    assert task_accuracy["sniah"]["fake"]["accuracy"] == 1.0
    assert task_accuracy["sniah"]["vanilla"]["accuracy"] == 1.0
    summary = summarize([type("Row", (), row)() for row in rows])
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["by_runner"]["fake"]["accuracy"] == 1.0
    assert summary["by_task"]["sniah"]["accuracy"] == 1.0
    assert summary["by_runner_task"]["fake/sniah"]["accuracy"] == 1.0
    assert summary["accuracy_by_task"]["sniah"]["vanilla"]["accuracy"] == 1.0
    assert summary["tasks_won"]["counts"]["fake"] == 1
    report_dir = tmp_path / "reports" / "fake" / "sniah"
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "report.md").exists()
    assert (tmp_path / "reports" / "fake" / "index.md").exists()
    problem = json.loads((report_dir / "sniah_0000.json").read_text(encoding="utf-8"))
    assert problem["prompt"]
    assert problem["expected"] in problem["inputs"]["haystack"]
    assert {solution["runner"] for solution in problem["solutions"]} == {"fake", "vanilla"}


def test_fake_cli_resume_skips_existing_rows(tmp_path) -> None:
    args = [
        "--provider",
        "fake",
        "--model",
        "fake",
        "--tasks",
        "sniah",
        "--runners",
        "fake",
        "vanilla",
        "--seeds",
        "0:2",
        "--task-param",
        "records=8",
        "--task-param",
        "filler_words=2",
        "--out-dir",
        str(tmp_path),
        "--report-dir",
        str(tmp_path / "reports"),
        "--run-id",
        "resume",
        "--no-live-save",
    ]

    assert main(args) == 0
    assert main([*args, "--resume"]) == 0

    results_path = tmp_path / "resume" / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    problem = json.loads(
        (tmp_path / "reports" / "fake" / "sniah" / "sniah_0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert problem["prompt"]
