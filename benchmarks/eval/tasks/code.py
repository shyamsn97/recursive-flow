"""Code execution and constraint official tasks.

Task implementations are ported from avilum/minrlm/eval:
https://github.com/avilum/minrlm/tree/master/eval
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.eval.tasks import register_task
from benchmarks.eval.tasks.common import (
    OfficialTask,
    RawTaskInstance,
    _load_dataset,
    _select_index,
)

def _extract_code_block(response: str) -> str:
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response.strip()


def _strip_html(text: str) -> str:
    text = re.sub(r"<pre[^>]*>", "\n```\n", text)
    text = re.sub(r"</pre>", "\n```\n", text)
    text = re.sub(r"<code[^>]*>", "`", text)
    text = re.sub(r"</code>", "`", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _execute_against_tests(code: str, test_cases: list[dict[str, Any]], timeout: int = 10) -> tuple[bool, int, int]:
    passed = 0
    total = len(test_cases)
    for tc in test_cases:
        stdin_input = str(tc.get("input", ""))
        expected_output = str(tc.get("output") or tc.get("expected_output", "")).strip()
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.stdout.strip() == expected_output:
                passed += 1
        except Exception:
            continue
    return passed == total, passed, total


@register_task("official_livecodebench")
class OfficialLiveCodeBenchTask(OfficialTask):
    """LiveCodeBench code generation lite: https://huggingface.co/datasets/livecodebench/code_generation_lite."""

    description = "LiveCodeBench v5/v6 competitive programming problems"
    difficulty = "very hard"
    _HF_BASE = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main"
    _JSONL_FILES = [
        f"{_HF_BASE}/test.jsonl",
        f"{_HF_BASE}/test2.jsonl",
        f"{_HF_BASE}/test3.jsonl",
        f"{_HF_BASE}/test4.jsonl",
        f"{_HF_BASE}/test5.jsonl",
    ]

    def __init__(self, max_samples: int | None = None, **_: Any) -> None:
        self.max_samples = max_samples
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            ds = _load_dataset("json", split="train", data_files=self._JSONL_FILES)
            if self.max_samples:
                ds = ds.select(range(min(self.max_samples, len(ds))))
            self._dataset = ds
        return self._dataset

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        row = self._get_dataset()[_select_index(seed, len(self._get_dataset()))]
        public_raw = row.get("public_test_cases", "[]")
        try:
            public_tests = json.loads(public_raw) if isinstance(public_raw, str) else public_raw
        except Exception:
            public_tests = []
        examples = []
        for i, tc in enumerate((public_tests or [])[:3]):
            examples.append(f"Example {i + 1}:\nInput:\n{str(tc.get('input', '')).strip()}\nOutput:\n{str(tc.get('output') or tc.get('expected_output', '')).strip()}")
        starter = str(row.get("starter_code", "")).strip()
        parts = [
            f"Solve this programming problem in Python.\n\n## {str(row.get('question_title', '')).strip()}\n\n{_strip_html(str(row.get('question_content', '')))}"
        ]
        if examples:
            parts.append("\n\n## Examples\n\n" + "\n\n".join(examples))
        if starter:
            parts.append(f"\n\n## Starter Code\n```python\n{starter}\n```")
        parts.append("\n\nWrite a complete Python solution. For stdin/stdout problems, read from stdin and print to stdout.")
        return RawTaskInstance(
            task="".join(parts),
            context="",
            expected=json.dumps({"test_cases": public_tests, "starter_code": starter}),
            metadata={"question_id": row.get("question_id"), "platform": row.get("platform"), "difficulty": row.get("difficulty")},
        )

    def check(self, response: str, expected: str) -> bool:
        return self.check_partial(response, expected) == 1.0

    def check_partial(self, response: str, expected: str) -> float:
        try:
            test_cases = json.loads(expected).get("test_cases", [])
        except Exception:
            return 0.0
        code = _extract_code_block(response)
        if not code or not test_cases:
            return 0.0
        _, passed, total = _execute_against_tests(code, test_cases)
        return passed / total if total else 0.0


@register_task("official_sudoku_extreme")
class OfficialSudokuExtremeTask(OfficialTask):
    """Sudoku Extreme: https://huggingface.co/datasets/sapientinc/sudoku-extreme."""

    description = "Sudoku Extreme (hard constraint-satisfaction puzzles)"
    difficulty = "hard"

    def __init__(self, data_dir: str = "evals/data", max_samples: int | None = None, min_rating: int | None = None, max_rating: int | None = None, **_: Any) -> None:
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.min_rating = min_rating
        self.max_rating = max_rating
        self._records: list[dict[str, Any]] | None = None

    def _get_records(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        parquet_path = self.data_dir / "sudoku_extreme" / "test.parquet"
        if parquet_path.exists():
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            if self.min_rating is not None:
                df = df[df["rating"] >= self.min_rating]
            if self.max_rating is not None:
                df = df[df["rating"] <= self.max_rating]
            if self.max_samples:
                df = df.head(self.max_samples)
            self._records = df.to_dict("records")
            return self._records
        ds = _load_dataset("sapientinc/sudoku-extreme", split="test")
        if self.max_samples:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        self._records = [dict(row) for row in ds]
        return self._records

    @staticmethod
    def _format_grid(puzzle: str) -> str:
        lines = []
        for r in range(9):
            row = puzzle[r * 9 : (r + 1) * 9]
            cells = [row[c] if row[c] != "." else "_" for c in range(9)]
            lines.append(" ".join(cells[0:3]) + " | " + " ".join(cells[3:6]) + " | " + " ".join(cells[6:9]))
            if r in (2, 5):
                lines.append("------+-------+------")
        return "\n".join(lines)

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        records = self._get_records()
        row = records[_select_index(seed, len(records))]
        puzzle = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        task = (
            "Solve this Sudoku puzzle. Each row, column, and 3x3 box must contain digits 1-9 exactly once.\n\n"
            f"{self._format_grid(puzzle)}\n\nRaw puzzle: {puzzle}\n\n"
            "Return ONLY the completed 81-digit solution string (no spaces, no newlines)."
        )
        return RawTaskInstance(task=task, context="", expected=answer, metadata={"rating": row.get("rating"), "puzzle": puzzle})

    def check(self, response: str, expected: str) -> bool:
        digits = re.sub(r"[^1-9]", "", response)
        return len(digits) >= 81 and digits[:81] == expected

    def check_partial(self, response: str, expected: str) -> float:
        digits = re.sub(r"[^1-9]", "", response)[:81]
        if not digits:
            return 0.0
        return sum(1 for a, b in zip(digits, expected) if a == b) / 81.0


