"""OOLONG benchmark dataset."""

from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset("oolong", tags=["long-context"])
class OolongDataset(Dataset):
    """Load OOLONG examples from Hugging Face or a local `evals/data/oolong` copy."""

    dataset_name = "oolongbench/oolong-synth"

    def __init__(
        self,
        data_dir: str = "evals/data",
        max_context_chars: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_context_chars = max_context_chars
        self.max_context_tokens = max_context_tokens
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        rows = self._load(split)
        if not rows:
            raise ValueError("No OOLONG examples fit the configured limits.")
        count = limit or 1
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        selected = indices[: min(count, len(indices))]
        return [self._example(rows[index], index=index) for index in selected]

    def _load(self, split: str) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        try:
            from datasets import load_dataset, load_from_disk  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "OOLONG requires the eval extra: pip install -e '.[eval]'"
            ) from exc

        local = self.data_dir / "oolong"
        if local.exists():
            ds = load_from_disk(str(local))
            if hasattr(ds, "keys") and split in ds:
                ds = ds[split]
        else:
            ds = load_dataset(self.dataset_name, split=split)

        rows: list[dict[str, Any]] = []
        for row in ds:
            context = str(
                row.get("context_window_text_with_labels")
                or row.get("context_window_text")
                or ""
            )
            context_len = row.get("context_len")
            if self.max_context_tokens is not None:
                if isinstance(context_len, (int, float)):
                    if context_len > self.max_context_tokens:
                        continue
                elif len(context) // 4 > self.max_context_tokens:
                    continue
            if self.max_context_chars is not None and len(context) > self.max_context_chars:
                continue
            rows.append(dict(row))
        self._rows = rows
        return rows

    def _example(self, row: dict[str, Any], *, index: int) -> Example:
        context = str(
            row.get("context_window_text_with_labels")
            or row.get("context_window_text")
            or ""
        )
        answers = _normalize_answers(row.get("answer"))
        answer_type = row.get("answer_type")
        return Example(
            id=f"oolong_{index:05d}",
            prompt=str(row.get("question", "")).strip(),
            context={"context": context},
            expected=answers,
            metadata={
                "dataset": row.get("dataset"),
                "context_len": row.get("context_len"),
                "answer_type": answer_type,
                "context_window_id": row.get("context_window_id"),
            },
        )

    def score(self, example: Example, prediction: Prediction) -> Score:
        answer = prediction.answer.lower()
        best = 0.0
        matched = None
        for expected in example.expected or []:
            candidate = str(expected).strip()
            if not candidate:
                continue
            if _is_numeric(candidate):
                value = _extract_number(answer)
                if value is not None:
                    score = 1.0 if abs(value - float(candidate)) < 1e-6 else 0.0
                    if score > best:
                        best, matched = score, candidate
            elif re.search(r"\b" + re.escape(candidate.lower()) + r"\b", answer):
                best, matched = 1.0, candidate
                break
        return Score(
            value=best,
            correct=best >= 1.0,
            details={"matched": matched, "expected": example.expected},
        )


def _normalize_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            out.extend(_normalize_answers(item))
        return out
    if isinstance(raw, dict):
        values = raw.get("answers") or raw.get("answer") or raw.values()
        return _normalize_answers(list(values) if not isinstance(values, str) else values)
    text = str(raw).strip()
    if not text:
        return []
    # HF rows often store answers as a Python-literal repr, e.g. "['incorrect']"
    # or "[48]". Parse those so we match the value, not the bracketed string.
    if text[0] in "[(" and text[-1] in ")]":
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return _normalize_answers(list(parsed))
        if parsed is not None:
            return [str(parsed).strip()]
    return [part.strip() for part in text.split("||") if part.strip()]


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _extract_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


__all__ = ["OolongDataset"]
