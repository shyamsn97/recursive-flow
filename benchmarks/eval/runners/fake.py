"""Fake runner for harness smoke tests."""

from __future__ import annotations

from benchmarks.eval import runner
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("fake")
class FakeRunner(Runner):
    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        del model, ctx
        return Prediction(answer=str(example.expected), metrics={"iterations": 1})


__all__ = ["FakeRunner"]
