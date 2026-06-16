"""Official RLM-Bench task adapters.

Ported from avilum/minrlm's eval task layer and adapted to rflow's
``TaskInstance(inputs=...)`` schema:
https://github.com/avilum/minrlm/tree/master/eval
"""

from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.eval.core import Score, TaskInstance


DEFAULT_DATA_DIR = Path("evals/data")


@dataclass(frozen=True)
class RawTaskInstance:
    """minrlm-style task instance before adapting to rflow inputs."""

    task: str
    context: str
    expected: str
    metadata: dict[str, Any] | None = None


class OfficialTask:
    """Base adapter for RLM-Bench tasks."""

    name = "official"
    description = "Official RLM-Bench task"
    difficulty = "medium"

    def generate(self, seed: int = 42, **kwargs) -> TaskInstance:
        raw = self.generate_raw(seed=seed, **kwargs)
        metadata = dict(raw.metadata or {})
        metadata.setdefault("source", "avilum/minrlm/eval")
        inputs = {"context": raw.context} if raw.context else {}
        return TaskInstance(
            task_name=self.name,
            task_id=f"{self.name}_{seed:04d}",
            seed=seed,
            prompt=raw.task,
            inputs=inputs,
            expected=raw.expected,
            metadata=metadata,
        )

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        raise NotImplementedError

    def check(self, response: str, expected: str) -> bool:
        raise NotImplementedError

    def check_partial(self, response: str, expected: str) -> float:
        return 1.0 if self.check(response, expected) else 0.0

    def score(self, answer: str, expected: object, metadata: dict) -> Score:
        expected_text = str(expected)
        value = self.check_partial(answer, expected_text)
        return Score(
            correct=self.check(answer, expected_text),
            value=value,
            details={"source": "avilum/minrlm/eval", "partial_score": value},
        )


def _optional_import_datasets():
    try:
        from datasets import load_dataset, load_from_disk
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
        return load_dataset, load_from_disk
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Missing eval dataset dependencies. Install with `pip install -e .[eval]` "
            "or install `datasets huggingface_hub pandas pyarrow`."
        ) from exc


def _load_dataset(
    dataset_name: str,
    *,
    split: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    local_name: str | None = None,
    config: str | None = None,
    data_files: Any = None,
):
    load_dataset, load_from_disk = _optional_import_datasets()
    local_root = Path(data_dir) / (local_name or dataset_name.split("/")[-1])
    if local_root.exists():
        split_dir = local_root / split
        return load_from_disk(str(split_dir if split_dir.exists() else local_root))
    if data_files is not None:
        return load_dataset("json", data_files=data_files, split="train")
    if config is not None:
        return load_dataset(dataset_name, config, split=split)
    return load_dataset(dataset_name, split=split)


def _select_index(seed: int, length: int) -> int:
    if length <= 0:
        raise ValueError("Empty dataset")
    return random.Random(seed).randrange(length)


def _load_json_records(
    dataset_root: Path, max_samples: int | None = None, split: str | None = None
) -> list[dict[str, Any]]:
    if dataset_root.is_file():
        candidates = [dataset_root]
    else:
        target_root = dataset_root / split if split and (dataset_root / split).is_dir() else dataset_root
        candidates = sorted(
            p
            for p in target_root.iterdir()
            if p.suffix in {".json", ".jsonl"}
            or p.name.endswith(".jsonl.gz")
            or p.name.endswith(".json.gz")
        )
    if not candidates:
        raise FileNotFoundError(f"No JSON/JSONL files found in {dataset_root}")

    path = candidates[0]
    records: list[dict[str, Any]] = []

    def open_stream(p: Path):
        if p.name.endswith(".gz"):
            return gzip.open(p, "rt", encoding="utf-8")
        return p.open("r", encoding="utf-8")

    with open_stream(path) as handle:
        if path.suffix == ".json" and not path.name.endswith(".jsonl"):
            payload = json.load(handle)
            if isinstance(payload, dict):
                payload = payload.get("data") or payload.get("examples") or payload.get("records") or [payload]
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected JSON structure in {path}")
            for item in payload:
                if isinstance(item, dict):
                    records.append(item)
                if max_samples and len(records) >= max_samples:
                    break
        else:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
                if max_samples and len(records) >= max_samples:
                    break
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def _stringify_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for i, item in enumerate(value):
            if isinstance(item, dict):
                title = item.get("path") or item.get("file") or item.get("docid") or item.get("id") or f"item_{i}"
                body = item.get("content") or item.get("text") or item.get("code") or item.get("snippet") or str(item)
                parts.append(f"[{title}]\n{body}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    if isinstance(value, dict):
        for key in ("content", "text", "code", "snippet"):
            if key in value:
                return str(value[key])
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _normalize_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, dict):
        for key in ("answer", "answers", "gold", "target", "label"):
            if key in raw:
                return _normalize_answers(raw[key])
        return []
    text = str(raw).strip()
    if "||" in text:
        return [part.strip() for part in text.split("||") if part.strip()]
    return [text] if text else []


def _extract_identifiers(text: str) -> list[str]:
    identifiers: list[str] = []
    for pattern in (
        r"\bdef\s+([A-Za-z_][\w]*)",
        r"\bfunction\s+([A-Za-z_][\w]*)",
        r"\bclass\s+([A-Za-z_][\w]*)",
    ):
        identifiers.extend(match.group(1) for match in re.finditer(pattern, text or ""))
    return identifiers


def _extract_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_numeric_string(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", text.strip()))


def _extract_choice(response: str, letters: str = "ABCD") -> str | None:
    match = re.search(rf"\b([{re.escape(letters)}])\b", response.upper())
    return match.group(1) if match else None


def _strip_ruler_prompt(prompt: str) -> str:
    if "<|im_start|>user" in prompt:
        user_part = prompt.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
    else:
        user_part = prompt
    return user_part.replace("<|im_start|>assistant", "").strip()


__all__ = [name for name in globals() if not name.startswith("__")]
