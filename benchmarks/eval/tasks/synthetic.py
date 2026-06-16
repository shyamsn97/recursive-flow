"""Deterministic synthetic tasks for smoke tests and early runner debugging."""

from __future__ import annotations

import random
import re

from benchmarks.eval.core import Score, TaskInstance
from benchmarks.eval.tasks import register_task


def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().strip("`'\"").strip()


@register_task("sniah")
class SyntheticNeedleTask:
    """Single needle-in-a-haystack retrieval with deterministic generation."""

    name = "sniah"
    description = "Synthetic single-needle retrieval"

    def __init__(self, records: int = 120, filler_words: int = 16) -> None:
        self.records = records
        self.filler_words = filler_words

    def generate(self, seed: int, **kwargs) -> TaskInstance:
        records = int(kwargs.get("records", self.records))
        filler_words = int(kwargs.get("filler_words", self.filler_words))
        rng = random.Random(seed)
        needle_index = rng.randrange(records)
        marker = f"needle-marker-{seed:04d}-{rng.randrange(10**8):08d}"
        expected = f"SECRET-{seed:04d}-{rng.randrange(10**10):010d}"
        vocabulary = [
            "harbor",
            "lantern",
            "quartz",
            "meadow",
            "orbit",
            "cobalt",
            "archive",
            "cedar",
            "signal",
            "vector",
            "delta",
            "ember",
            "garden",
            "matrix",
            "notebook",
            "prairie",
        ]
        blocks: list[str] = []
        for i in range(records):
            local_rng = random.Random((seed * 1_000_003) + i)
            words = [local_rng.choice(vocabulary) for _ in range(filler_words)]
            record_marker = marker if i == needle_index else f"decoy-{seed:04d}-{i:04d}"
            record_secret = expected if i == needle_index else f"DECOY-{seed:04d}-{i:04d}"
            blocks.append(
                "\n".join(
                    [
                        f"RECORD {i:04d}",
                        f"title: synthetic benchmark record {i}",
                        f"marker: {record_marker}",
                        f"body: {' '.join(words)}",
                        f"secret: {record_secret}",
                        "END_RECORD",
                    ]
                )
            )
        rng.shuffle(blocks)
        haystack = "\n\n".join(blocks)
        prompt = (
            "Find the record whose marker is exactly "
            f"`{marker}`. Return only that record's `secret` value, with no "
            "explanation and no extra text. The records are in INPUTS['haystack']."
        )
        return TaskInstance(
            task_name=self.name,
            task_id=f"{self.name}_{seed:04d}",
            seed=seed,
            prompt=prompt,
            inputs={"haystack": haystack},
            expected=expected,
            metadata={
                "records": records,
                "filler_words": filler_words,
                "needle_index": needle_index,
                "marker": marker,
                "context_chars": len(haystack),
            },
        )

    def score(self, answer: str, expected: object, metadata: dict) -> Score:
        normalized = _normalize(answer)
        expected_text = str(expected)
        correct = normalized == expected_text
        return Score(
            correct=correct,
            value=1.0 if correct else 0.0,
            details={
                "normalized_answer": normalized,
                "contains_expected": expected_text in normalized,
            },
        )
