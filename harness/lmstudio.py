"""Model lifecycle control via the `lms` CLI.

Unified memory will not hold two of these models at once, so a stage loads its model
once at the start and unloads it at the end — never per run, which would let
load time dominate wall clock and pollute the §5 timings.

`§3.1` requires exactly one model loaded. This module is how that state is
*established*; `harness/environment.py` asserts it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass

LMS = "lms"


class LMStudioError(RuntimeError):
    pass


def _run(args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    if shutil.which(LMS) is None:
        raise LMStudioError(
            "the `lms` CLI is not on PATH; install it from LM Studio "
            "(Developer > Command Line Tools)"
        )
    return subprocess.run(
        [LMS, *args], capture_output=True, text=True, timeout=timeout
    )


@dataclass(frozen=True)
class LoadedModel:
    identifier: str
    model_key: str
    context_length: int | None

    @classmethod
    def from_json(cls, payload: dict) -> LoadedModel:
        return cls(
            identifier=payload.get("identifier", ""),
            model_key=payload.get("modelKey") or payload.get("path", ""),
            context_length=payload.get("contextLength"),
        )


def list_loaded() -> list[LoadedModel]:
    """Models currently loaded. Note this is not `/v1/models`, which lists
    everything downloaded rather than everything resident."""
    completed = _run(["ps", "--json"], timeout=60)
    if completed.returncode != 0:
        raise LMStudioError(f"lms ps failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise LMStudioError(f"could not parse lms ps output: {exc}") from exc
    return [LoadedModel.from_json(item) for item in payload]


def unload_all() -> None:
    completed = _run(["unload", "--all"], timeout=120)
    if completed.returncode != 0:
        raise LMStudioError(f"lms unload --all failed: {completed.stderr.strip()}")


def load(
    model_key: str,
    context_length: int | None = None,
    identifier: str | None = None,
    gpu: str | None = None,
    timeout: float = 600.0,
) -> LoadedModel:
    """Load one model, failing loudly rather than leaving ambiguous state."""
    args = ["load", model_key, "--yes"]
    if context_length is not None:
        args += ["--context-length", str(context_length)]
    if identifier is not None:
        args += ["--identifier", identifier]
    if gpu is not None:
        args += ["--gpu", gpu]

    completed = _run(args, timeout=timeout)
    if completed.returncode != 0:
        raise LMStudioError(
            f"lms load {model_key} failed: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )

    wanted = identifier or model_key
    for model in list_loaded():
        if wanted in (model.identifier, model.model_key):
            return model
    raise LMStudioError(f"{model_key} reported loaded but is not in lms ps")


def estimate(model_key: str, context_length: int | None = None) -> str:
    """Resource estimate without loading. Useful as a cross-check on the §2.2
    admissibility calculation."""
    args = ["load", model_key, "--estimate-only", "--yes"]
    if context_length is not None:
        args += ["--context-length", str(context_length)]
    completed = _run(args, timeout=120)
    return (completed.stdout or completed.stderr).strip()


@contextmanager
def loaded(
    model_key: str,
    context_length: int | None = None,
    identifier: str | None = None,
    gpu: str | None = None,
):
    """Bracket a stage: load once, run everything, unload once.

    Unloads anything already resident first, so a stage never runs against a
    model it did not choose, and always unloads on the way out — including on
    failure, so an aborted stage does not strand a model in memory.
    """
    unload_all()
    model = load(model_key, context_length=context_length, identifier=identifier, gpu=gpu)
    try:
        yield model
    finally:
        unload_all()
