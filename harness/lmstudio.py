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
class Artefact:
    """One downloaded model, identified by the field that does not move.

    `model_key` is **not** a stable identifier. LM Studio derives it from the
    set of models currently installed, appending `@<quant>` only where one is
    needed to disambiguate: with a single Bonsai GGUF present the key is
    `ternary-bonsai-8b`, and installing a second one silently renames it. A key
    recorded in a result set therefore need not denote the same artefact later.

    `path` (LM Studio's `indexedModelIdentifier`) is the publisher/repo/file
    triple, is unique, and does not change when unrelated models are installed.
    It is what §2.1 already records as `model_repo` + `quant_file`, so the
    harness identifies models by path throughout and treats `model_key` as
    nothing more than the handle `lms load` happens to accept.
    """

    model_key: str
    path: str
    size_bytes: int | None = None
    architecture: str | None = None
    quantization: str | None = None
    max_context_length: int | None = None

    @classmethod
    def from_json(cls, payload: dict) -> Artefact:
        quant = payload.get("quantization")
        return cls(
            model_key=payload.get("modelKey", ""),
            path=payload.get("indexedModelIdentifier") or payload.get("path", ""),
            size_bytes=payload.get("sizeBytes"),
            architecture=payload.get("architecture"),
            quantization=quant.get("name") if isinstance(quant, dict) else quant,
            max_context_length=payload.get("maxContextLength"),
        )


@dataclass(frozen=True)
class LoadedModel:
    identifier: str
    model_key: str
    path: str
    context_length: int | None

    @classmethod
    def from_json(cls, payload: dict) -> LoadedModel:
        return cls(
            identifier=payload.get("identifier", ""),
            model_key=payload.get("modelKey", ""),
            path=payload.get("indexedModelIdentifier") or payload.get("path", ""),
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


def list_downloaded() -> list[Artefact]:
    """Every model on disk, whether or not it is resident."""
    completed = _run(["ls", "--json"], timeout=60)
    if completed.returncode != 0:
        raise LMStudioError(f"lms ls failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise LMStudioError(f"could not parse lms ls output: {exc}") from exc
    return [Artefact.from_json(item) for item in payload]


def resolve(model: str) -> Artefact:
    """Map a stable path (or an exact key) to the artefact it denotes.

    Matching is exact and must be unique. `lms load` itself matches the key as a
    *substring* and, under `--yes`, loads the first of several matches after a
    warning on stdout that a caller checking only the exit status never sees:
    `lms load lfm2.5-2.6b` matches four artefacts and loads an MLX build. For a
    measuring instrument that is the worst available failure mode, because the
    run succeeds and the results are attributed to the wrong artefact.
    """
    artefacts = list_downloaded()

    for field in ("path", "model_key"):
        matches = [a for a in artefacts if getattr(a, field) == model]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise LMStudioError(
                f"{model!r} is ambiguous: it matches {len(matches)} downloaded "
                f"models on {field}. Identify the model by its path instead: "
                + ", ".join(sorted(a.path for a in matches))
            )

    raise LMStudioError(
        f"no downloaded model matches {model!r} exactly. Identify models by "
        f"path, not by key. Available: "
        + ", ".join(sorted(a.path for a in artefacts))
    )


def unload_all() -> None:
    completed = _run(["unload", "--all"], timeout=120)
    if completed.returncode != 0:
        raise LMStudioError(f"lms unload --all failed: {completed.stderr.strip()}")


def load(
    model: str,
    context_length: int | None = None,
    identifier: str | None = None,
    gpu: str | None = None,
    timeout: float = 600.0,
) -> LoadedModel:
    """Load one model by path (or exact key), failing loudly on any ambiguity.

    Verification is on `path`, not on the key or the identifier: the point is to
    confirm that the artefact now resident is the one asked for, and only the
    path carries that meaning.
    """
    artefact = resolve(model)

    args = ["load", artefact.model_key, "--yes"]
    if context_length is not None:
        args += ["--context-length", str(context_length)]
    if identifier is not None:
        args += ["--identifier", identifier]
    if gpu is not None:
        args += ["--gpu", gpu]

    completed = _run(args, timeout=timeout)
    if completed.returncode != 0:
        raise LMStudioError(
            f"lms load {artefact.path} failed: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )

    resident = list_loaded()
    for loaded_model in resident:
        if loaded_model.path == artefact.path:
            return loaded_model
    raise LMStudioError(
        f"{artefact.path} reported loaded, but lms ps shows "
        + (", ".join(m.path for m in resident) or "nothing resident")
    )


def estimate(model: str, context_length: int | None = None) -> str:
    """Resource estimate without loading. Useful as a cross-check on the §2.2
    admissibility calculation."""
    args = ["load", resolve(model).model_key, "--estimate-only", "--yes"]
    if context_length is not None:
        args += ["--context-length", str(context_length)]
    completed = _run(args, timeout=120)
    return (completed.stdout or completed.stderr).strip()


@contextmanager
def loaded(
    model: str,
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
    loaded_model = load(
        model, context_length=context_length, identifier=identifier, gpu=gpu
    )
    try:
        yield loaded_model
    finally:
        unload_all()
