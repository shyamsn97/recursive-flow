"""Professional work task adapters.

Task implementations are ported from avilum/minrlm/eval:
https://github.com/avilum/minrlm/tree/master/eval
"""

from __future__ import annotations

import json
import re
from typing import Any

from benchmarks.eval.tasks import register_task
from benchmarks.eval.tasks.common import OfficialTask, RawTaskInstance, _load_dataset, _select_index

@register_task("official_gdpval")
class OfficialGDPVALTask(OfficialTask):
    """GDP Val: https://huggingface.co/datasets/openai/gdpval."""

    description = "GDPVAL (real professional tasks across 44 occupations)"
    difficulty = "very hard"
    dataset_name = "openai/gdpval"
    default_split = "train"

    def __init__(self, max_samples: int | None = None, occupation_filter: str | None = None, **_: Any) -> None:
        self.max_samples = max_samples
        self.occupation_filter = occupation_filter
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            ds = _load_dataset(self.dataset_name, split=self.default_split)
            if self.occupation_filter:
                ds = ds.filter(lambda x: x.get("occupation") == self.occupation_filter)
            if self.max_samples:
                ds = ds.select(range(min(self.max_samples, len(ds))))
            self._dataset = ds
        return self._dataset

    def generate_raw(self, seed: int = 42, **kwargs) -> RawTaskInstance:
        row = self._get_dataset()[_select_index(seed, len(self._get_dataset()))]
        rubric = row.get("rubric_json", "[]")
        criteria: list[str] = []
        try:
            for item in json.loads(rubric) if isinstance(rubric, str) else rubric:
                text = item.get("criterion", "")
                criteria.extend(re.findall(r'"([^"]+)"', text))
                criteria.extend(re.findall(r"'([^']+)'", text))
                criteria.extend(re.findall(r"\$[\d,]+\.?\d*|\d+\.?\d*%|\d{4}-\d{2}-\d{2}|\d+\.\d+", text))
        except Exception:
            pass
        expected = "||".join(dict.fromkeys(c.strip() for c in criteria if len(c.strip()) > 1))
        context = "\n".join(str(uri) for uri in row.get("reference_file_hf_uris", []) or [])
        return RawTaskInstance(
            task=str(row.get("prompt", "")).strip(),
            context=context,
            expected=expected,
            metadata={"task_id": row.get("task_id"), "occupation": row.get("occupation"), "sector": row.get("sector")},
        )

    def check(self, response: str, expected: str) -> bool:
        if not response.strip() or len(response) < 50:
            return False
        lower = response.lower()
        if any(x in lower for x in ("can't access", "cannot access", "please upload", "i don't have", "need the file")):
            return False
        criteria = [c.strip() for c in expected.split("||") if c.strip()]
        if criteria:
            matches = sum(1 for c in criteria if c.lower() in lower)
            return matches >= max(1, int(len(criteria) * 0.2)) and len(response) > 200
        return len(response) > 300 and sum(1 for ind in (": ", "\n- ", "\n* ", "patient", "dr ")) >= 2


