"""Reasoning and instruction-following official tasks.

Task implementations are ported from avilum/minrlm/eval:
https://github.com/avilum/minrlm/tree/master/eval
"""

from __future__ import annotations

import json
import random
import re
import string
from typing import Any

from benchmarks.eval.tasks import register_task
from benchmarks.eval.tasks.common import (
    OfficialTask,
    RawTaskInstance,
    _extract_choice,
    _load_dataset,
    _select_index,
)

class _HFChoiceTask(OfficialTask):
    dataset_name = ""
    dataset_config: str | None = None
    default_split = "train"
    max_letter = "D"

    def __init__(self, max_samples: int | None = None, **kwargs: Any) -> None:
        self.max_samples = max_samples
        self.kwargs = kwargs
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            ds = _load_dataset(self.dataset_name, split=self.default_split, config=self.dataset_config)
            ds = self._filter_dataset(ds)
            if self.max_samples:
                ds = ds.select(range(min(self.max_samples, len(ds))))
            self._dataset = ds
        return self._dataset

    def _filter_dataset(self, ds):
        return ds

    def check(self, response: str, expected: str) -> bool:
        letters = string.ascii_uppercase[: string.ascii_uppercase.index(self.max_letter) + 1]
        return _extract_choice(response, letters) == expected


@register_task("official_aime_2025")
class OfficialAIME2025Task(OfficialTask):
    """AIME 2025: https://huggingface.co/datasets/MathArena/aime_2025."""

    description = "AIME 2025 (30 competition math problems)"
    difficulty = "very hard"
    dataset_name = "MathArena/aime_2025"
    default_split = "train"

    def __init__(self, max_samples: int | None = None, problem_type_filter: str | None = None, **_: Any) -> None:
        self.max_samples = max_samples
        self.problem_type_filter = problem_type_filter
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            ds = _load_dataset(self.dataset_name, split=self.default_split)
            if self.problem_type_filter:
                ds = ds.filter(lambda x: any(self.problem_type_filter.lower() in pt.lower() for pt in x.get("problem_type", [])))
            if self.max_samples:
                ds = ds.select(range(min(self.max_samples, len(ds))))
            self._dataset = ds
        return self._dataset

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        row = self._get_dataset()[_select_index(seed, len(self._get_dataset()))]
        return RawTaskInstance(
            task=(
                "Solve this AIME competition math problem. Return ONLY the final "
                f"integer answer, without any explanation.\n\n{str(row.get('problem', '')).strip()}"
            ),
            context="",
            expected=str(row.get("answer", "")),
            metadata={"problem_idx": row.get("problem_idx"), "problem_types": row.get("problem_type", [])},
        )

    def check(self, response: str, expected: str) -> bool:
        if not expected:
            return False
        return any(int(num) == int(expected) for num in re.findall(r"-?\d+", response.strip()))


@register_task("official_gpqa_diamond")
class OfficialGPQADiamondTask(_HFChoiceTask):
    """GPQA Diamond: https://huggingface.co/datasets/Idavidrein/gpqa."""

    description = "GPQA Diamond (198 grad-level science questions)"
    difficulty = "very hard"
    dataset_name = "Idavidrein/gpqa"
    dataset_config = "gpqa_diamond"
    max_letter = "D"

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        ds = self._get_dataset()
        idx = _select_index(seed, len(ds))
        row = ds[idx]
        correct = row.get("Correct Answer") or row.get("correct_answer", "")
        incorrects = [
            row.get("Incorrect Answer 1") or row.get("incorrect_answer_1", ""),
            row.get("Incorrect Answer 2") or row.get("incorrect_answer_2", ""),
            row.get("Incorrect Answer 3") or row.get("incorrect_answer_3", ""),
        ]
        choices = [correct, *incorrects]
        order = list(range(4))
        random.Random(seed + idx).shuffle(order)
        shuffled = [choices[i] for i in order]
        correct_letter = chr(65 + order.index(0))
        choices_text = "\n".join(f"{chr(65 + i)}) {shuffled[i]}" for i in range(4))
        return RawTaskInstance(
            task=f"{row.get('Question') or row.get('question', '')}\n\nChoices:\n{choices_text}\n\nReturn ONLY the letter (A, B, C, or D).",
            context="",
            expected=correct_letter,
            metadata={"subdomain": row.get("Subdomain") or row.get("subdomain")},
        )


@register_task("official_mmlu_pro")
class OfficialMMLUProTask(_HFChoiceTask):
    """MMLU-Pro: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro."""

    description = "MMLU-Pro (12K hard MC questions, 10 choices)"
    difficulty = "hard"
    dataset_name = "TIGER-Lab/MMLU-Pro"
    default_split = "test"
    max_letter = "J"

    def _filter_dataset(self, ds):
        category = self.kwargs.get("category_filter")
        return ds.filter(lambda x: x.get("category") == category) if category else ds

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        ds = self._get_dataset()
        row = ds[_select_index(seed, len(ds))]
        options = row.get("options", [])
        letters = string.ascii_uppercase[: len(options)]
        choices_text = "\n".join(f"{letters[i]}) {options[i]}" for i in range(len(options)))
        return RawTaskInstance(
            task=f"{str(row.get('question', '')).strip()}\n\nChoices:\n{choices_text}\n\nReturn ONLY the letter ({', '.join(letters)}).",
            context="",
            expected=str(row.get("answer", "")).strip().upper(),
            metadata={"category": row.get("category"), "question_id": row.get("question_id")},
        )


def _relation_check(actual: int, expected: int, relation: str) -> bool:
    relation = relation.lower().strip()
    if relation in ("at least", "no less than", ">="):
        return actual >= expected
    if relation in ("at most", "no more than", "<="):
        return actual <= expected
    if relation in ("exactly", "==", "equal to"):
        return actual == expected
    if relation in ("less than", "<"):
        return actual < expected
    if relation in ("greater than", "more than", ">"):
        return actual > expected
    return actual >= expected


def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def _check_ifeval_instruction(instruction_id: str, kwargs: dict[str, Any], response: str) -> bool:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    lower = response.lower()
    if instruction_id == "keywords:existence":
        return all(str(kw).lower() in lower for kw in kwargs.get("keywords", []))
    if instruction_id == "keywords:frequency":
        return _relation_check(lower.count(str(kwargs.get("keyword", "")).lower()), int(kwargs.get("frequency", 1)), kwargs.get("relation", "at least"))
    if instruction_id == "keywords:forbidden_words":
        return not any(str(w).lower() in lower for w in kwargs.get("forbidden_words", []))
    if instruction_id == "length_constraints:number_words":
        return _relation_check(_count_words(response), int(kwargs.get("num_words", 1)), kwargs.get("relation", "at least"))
    if instruction_id == "length_constraints:number_sentences":
        return _relation_check(_count_sentences(response), int(kwargs.get("num_sentences", 1)), kwargs.get("relation", "at least"))
    if instruction_id == "detectable_format:json_format":
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            json.loads(text)
            return True
        except Exception:
            return False
    if instruction_id == "detectable_format:constrained_response":
        return _count_words(response.strip()) <= 10
    if instruction_id == "combination:two_responses":
        return "******" in response
    if instruction_id == "startend:end_checker":
        phrase = kwargs.get("end_phrase", "")
        return response.rstrip().lower().endswith(str(phrase).lower()) if phrase else True
    if instruction_id == "punctuation:no_comma":
        return "," not in response
    if instruction_id == "change_case:english_capital":
        letters = re.findall(r"[a-zA-Z]", response)
        return all(c.isupper() for c in letters) if letters else False
    if instruction_id == "change_case:english_lowercase":
        letters = re.findall(r"[a-zA-Z]", response)
        return all(c.islower() for c in letters) if letters else False
    return True


@register_task("official_ifeval")
class OfficialIFEvalTask(OfficialTask):
    """IFEval: https://huggingface.co/datasets/google/IFEval."""

    description = "IFEval (541 instruction-following prompts)"
    difficulty = "medium"
    dataset_name = "google/IFEval"
    default_split = "train"

    def __init__(self, max_samples: int | None = None, **_: Any) -> None:
        self.max_samples = max_samples
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            ds = _load_dataset(self.dataset_name, split=self.default_split)
            if self.max_samples:
                ds = ds.select(range(min(self.max_samples, len(ds))))
            self._dataset = ds
        return self._dataset

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        row = self._get_dataset()[_select_index(seed, len(self._get_dataset()))]
        expected = json.dumps({"instruction_id_list": row.get("instruction_id_list", []), "kwargs": row.get("kwargs", [])})
        return RawTaskInstance(
            task=str(row.get("prompt", "")).strip(),
            context="",
            expected=expected,
            metadata={"key": row.get("key"), "instruction_types": row.get("instruction_id_list", [])},
        )

    def check(self, response: str, expected: str) -> bool:
        return self.check_partial(response, expected) == 1.0

    def check_partial(self, response: str, expected: str) -> float:
        try:
            info = json.loads(expected)
        except Exception:
            return 0.0
        ids = info.get("instruction_id_list", [])
        kwargs_list = info.get("kwargs", [])
        if not ids:
            return 1.0
        passed = sum(
            1
            for i, inst_id in enumerate(ids)
            if _check_ifeval_instruction(inst_id, kwargs_list[i] if i < len(kwargs_list) else {}, response)
        )
        return passed / len(ids)

