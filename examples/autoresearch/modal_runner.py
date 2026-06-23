"""Modal launcher for autoresearch train.py trials."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModalConfig:
    app_name: str = "rlmflow-autoresearch"
    gpu: str = "L4"
    parallel: int = 4
    timeout_s: int = 1020
    volume_name: str = "rlmflow-autoresearch-cache"
    python_version: str | None = None


def preflight(config: ModalConfig) -> dict[str, Any]:
    """Fail early if Modal is not installed."""

    _modal()
    return {
        "status": "ok",
        "gpu": config.gpu,
        "app_name": config.app_name,
    }


def submit(
    config: ModalConfig,
    *,
    path: str | Path,
    train_budget_s: int,
    slug: str,
    n: int,
    run_id: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Spawn one detached Modal job for an archived train.py path."""

    modal = _modal()
    app = modal.App(config.app_name)
    cache_volume = modal.Volume.from_name(config.volume_name, create_if_missing=True)

    @app.function(
        image=_image(modal, config),
        gpu=config.gpu,
        timeout=config.timeout_s,
        max_containers=max(1, config.parallel),
        volumes={"/root/.cache/autoresearch": cache_volume},
        serialized=True,
    )
    def run_train(payload: dict[str, Any]) -> dict[str, Any]:
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from time import monotonic

        t0 = monotonic()
        trial = payload["n"]
        slug = payload["slug"]
        print(f"[autoresearch] start trial={trial} slug={slug}", flush=True)

        with tempfile.TemporaryDirectory(prefix="autoresearch-") as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()

            (root / "train.py").write_text(payload["source"])

            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "ARTIFACT_DIR": str(artifact_dir),
                "AUTORESEARCH_RUN_ID": payload["run_id"],
                "AUTORESEARCH_TRIAL": str(trial),
                "AUTORESEARCH_SLUG": slug,
            }
            if payload.get("seed") is not None:
                env["AUTORESEARCH_SEED"] = str(payload["seed"])

            try:
                train = subprocess.run(
                    [sys.executable, "-u", "train.py"],
                    cwd=str(root),
                    env=env,
                    timeout=int(payload["train_budget_s"]),
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[autoresearch] train timed out trial={trial} "
                    f"timeout_s={payload['train_budget_s']}",
                    file=sys.stderr,
                    flush=True,
                )
                return {
                    "status": "timeout",
                    "returncode": -1,
                    "elapsed_s": monotonic() - t0,
                }

            print(
                f"[autoresearch] done trial={trial} slug={slug} "
                f"returncode={train.returncode}",
                flush=True,
            )
            result_path = artifact_dir / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            status = (
                "succeeded"
                if train.returncode == 0 and result.get("val_bpb") is not None
                else "crashed"
            )
            return {
                **result,
                "status": status,
                "returncode": train.returncode,
                "elapsed_s": monotonic() - t0,
                "artifact_names": sorted(p.name for p in artifact_dir.iterdir()),
            }

    payload = {
        "source": Path(path).read_text(),
        "train_budget_s": train_budget_s,
        "slug": slug,
        "n": n,
        "run_id": run_id,
        "seed": seed,
    }
    t0 = time.monotonic()
    with modal.enable_output():
        with app.run(detach=True):
            call = run_train.spawn(payload)

    return {
        "status": "submitted",
        "val_bpb": None,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "elapsed_s": time.monotonic() - t0,
        "job_id": (
            getattr(call, "object_id", None)
            or getattr(call, "function_call_id", None)
            or f"{run_id}:{n}:{slug}"
        ),
        "gpu": config.gpu,
    }


def collect(config: ModalConfig, *, job_id: str, timeout_s: float = 0) -> dict[str, Any]:
    """Fetch one submitted job result if it is ready."""

    modal = _modal()
    try:
        function_call = modal.FunctionCall.from_id(job_id)
    except AttributeError:
        from modal.functions import FunctionCall  # pyright: ignore[reportMissingImports]

        function_call = FunctionCall.from_id(job_id)

    try:
        result = function_call.get(timeout=timeout_s)
    except TimeoutError:
        return {"status": "submitted", "job_id": job_id}
    return {
        **(result or {}),
        "job_id": job_id,
        "gpu": config.gpu,
    }


def classify_result(returncode: int, stderr: str, val_bpb: float | None) -> str:
    lower = (stderr or "").lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return "oom"
    if returncode != 0:
        return "crashed"
    if val_bpb is None:
        return "crashed"
    return "succeeded"


def _modal():
    try:
        import modal  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Modal autoresearch requires the modal extra: pip install -e '.[modal]'"
        ) from exc
    return modal


def _image(modal, config: ModalConfig):
    python_version = config.python_version or (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
            add_python=python_version,
        )
        .apt_install("git")
        .run_commands(
            "python -m pip install --upgrade pip",
            "python -m pip install --extra-index-url "
            "https://download.pytorch.org/whl/cu128 torch==2.9.1",
            "python -m pip install 'requests>=2.32.0'",
        )
        .env({"PYTHONUNBUFFERED": "1"})
    )
