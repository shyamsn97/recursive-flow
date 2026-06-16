"""Official RLM runner from the RLM paper implementation.

Adapted from avilum/minrlm's official eval runner:
https://github.com/avilum/minrlm/blob/master/eval/runners.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from rflow.clients import LLMClient

from benchmarks.eval.core import RunResult, TaskInstance
from benchmarks.eval.runners import register_runner


@register_runner("official")
class OfficialRLMRunner:
    """Run the official `alexzhang13/rlm` package in an isolated venv."""

    name = "official"
    repo = "git+https://github.com/alexzhang13/rlm"

    def __init__(self) -> None:
        self._venv_python: str | None = None

    def run(
        self,
        instance: TaskInstance,
        *,
        client: LLMClient,
        model: str,
        out_dir: Path,
        max_iters: int,
        max_depth: int,
        live_save: bool,
    ) -> RunResult:
        del client, max_depth, live_save
        start = time.perf_counter()
        log_dir = out_dir / "official_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        context_file = _write_temp(_render_context(instance))
        task_file = _write_temp(instance.prompt)
        try:
            python = self._ensure_venv()
            proc = subprocess.run(
                [
                    python,
                    "-c",
                    _official_script(
                        context_file=context_file,
                        task_file=task_file,
                        model=model,
                        log_dir=log_dir,
                        max_iters=max_iters,
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=max(300, max_iters * 60),
                env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
            )
            if proc.returncode != 0:
                return RunResult(
                    answer="",
                    time_seconds=time.perf_counter() - start,
                    error=_short_error(proc.stderr or proc.stdout),
                    metadata={"model": model, "log_dir": str(log_dir), "runner": "official"},
                )
            data = _parse_result(proc.stdout)
            return RunResult(
                answer=str(data.get("response") or ""),
                input_tokens=int(data.get("input_tokens") or 0),
                output_tokens=int(data.get("output_tokens") or 0),
                time_seconds=float(data.get("time_seconds") or (time.perf_counter() - start)),
                iterations=int(data.get("iterations") or 1),
                metadata={
                    "model": model,
                    "log_dir": str(log_dir),
                    "log_file_path": data.get("log_file_path"),
                    "generated_code": data.get("generated_code"),
                    "runner": "official",
                },
            )
        except Exception as exc:  # benchmark rows should record failures
            return RunResult(
                answer="",
                time_seconds=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"model": model, "log_dir": str(log_dir), "runner": "official"},
            )
        finally:
            _unlink(context_file)
            _unlink(task_file)

    def _ensure_venv(self) -> str:
        if self._venv_python:
            return self._venv_python

        venv_dir = Path(tempfile.gettempdir()) / "rflow_eval_official_rlm_venv"
        python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if python.exists() and _can_import_rlm(python):
            self._venv_python = str(python)
            return self._venv_python

        shutil.rmtree(venv_dir, ignore_errors=True)
        if shutil.which("uv"):
            _run_checked(["uv", "venv", str(venv_dir)])
            _run_checked(
                ["uv", "pip", "install", self.repo, "--python", str(python)],
            )
        else:
            _run_checked([sys.executable, "-m", "venv", str(venv_dir)])
            _run_checked(
                [str(python), "-m", "pip", "install", self.repo],
            )
        self._venv_python = str(python)
        return self._venv_python


def _render_context(instance: TaskInstance) -> str:
    if set(instance.inputs) == {"context"}:
        return instance.inputs["context"]
    return "\n\n".join(f"INPUT {key}:\n{value}" for key, value in sorted(instance.inputs.items()))


def _official_script(
    *,
    context_file: str,
    task_file: str,
    model: str,
    log_dir: Path,
    max_iters: int,
) -> str:
    return textwrap.dedent(
        f"""
        import json
        import time

        from rlm import RLM
        from rlm.logger import RLMLogger

        with open({json.dumps(context_file)}, encoding="utf-8") as handle:
            context = handle.read()
        with open({json.dumps(task_file)}, encoding="utf-8") as handle:
            task = handle.read()

        logger = RLMLogger(log_dir={json.dumps(str(log_dir))})
        start = time.time()
        rlm = RLM(
            backend="openai",
            backend_kwargs={{"model_name": {json.dumps(model)}}},
            environment="local",
            max_iterations={max_iters},
            verbose=False,
            logger=logger,
        )
        result = rlm.completion(prompt=context, root_prompt=task)
        elapsed = time.time() - start

        input_tokens = 0
        output_tokens = 0
        usage = getattr(result, "usage_summary", None)
        model_summaries = getattr(usage, "model_usage_summaries", None) if usage else None
        if model_summaries:
            for item in model_summaries.values():
                input_tokens += getattr(item, "total_input_tokens", 0)
                output_tokens += getattr(item, "total_output_tokens", 0)

        generated_code = None
        trajectory = logger.get_trajectory()
        for iteration in (trajectory or {{}}).get("iterations", []):
            code_blocks = iteration.get("code_blocks", [])
            if code_blocks:
                generated_code = code_blocks[0].get("code")
                break

        response = getattr(result, "response", "") or ""
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]

        print("<<<RESULT>>>")
        print(json.dumps({{
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_seconds": elapsed,
            "iterations": getattr(result, "num_iterations", None) or getattr(result, "iterations", None) or 1,
            "generated_code": generated_code,
            "log_file_path": getattr(logger, "log_file_path", None),
        }}))
        """
    )


def _parse_result(stdout: str) -> dict[str, object]:
    marker = "<<<RESULT>>>"
    if marker not in stdout:
        raise ValueError(f"official runner output missing {marker}: {_short_error(stdout)}")
    return json.loads(stdout.split(marker, 1)[1].strip())


def _can_import_rlm(python: Path) -> bool:
    result = subprocess.run([str(python), "-c", "import rlm"], capture_output=True)
    return result.returncode == 0


def _run_checked(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        output = _short_error("\n".join(part for part in (result.stderr, result.stdout) if part))
        raise RuntimeError(f"command failed: {' '.join(command)}\n{output}")


def _write_temp(text: str) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    with handle:
        handle.write(text)
    return handle.name


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _short_error(text: str, limit: int = 1000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = ["OfficialRLMRunner"]
