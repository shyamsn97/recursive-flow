"""Clean benchmark harness for rlmflow evaluations."""

from __future__ import annotations

from benchmarks.eval.registry import Registry
from benchmarks.eval.types import Dataset, Logger, Model, Runner

DATASETS = Registry[Dataset]("dataset")
RUNNERS = Registry[Runner]("runner")
MODELS = Registry[Model]("model")
LOGGERS = Registry[Logger]("logger")

dataset = DATASETS.decorator
runner = RUNNERS.decorator
model = MODELS.decorator
logger = LOGGERS.decorator

__all__ = [
    "DATASETS",
    "LOGGERS",
    "MODELS",
    "RUNNERS",
    "Dataset",
    "Logger",
    "Model",
    "Runner",
    "dataset",
    "logger",
    "model",
    "runner",
]
