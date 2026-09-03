"""Fixture preparation and grading.

A driver is any callable ``(task, sandbox) -> RunOutcome`` (doc/benchmark.md §4.1).
Grading is identical for every driver: it inspects final fixture state and the
final answer, never the transcript's internal structure.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .execution import Executor
from .paths import ensure_runs_root
from .sandbox import Sandbox, tree_hashes
from .scoring import progress_score
from .types import Ctx, RunOutcome, Task, load_expected

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

Driver = Callable[[Task, Sandbox], RunOutcome]


@dataclass
class Graded:
    task_id: str
    suite: str
    passed: bool
    progress: int
    steps: int
    tool_calls: int
    invalid_calls: int
    path_errors: int
    termination_reason: str
    wall_clock_s: float
    answer: str
    # Names of the sub-conditions that did NOT hold, for a task that declares
    # a breakdown; None where the task declares none. Diagnostic only.
    condition_failures: tuple[str, ...] | None = None
    metrics: dict | None = None
    transcript: list[dict] | None = None


# The work directory holding a run's fixture copy, while the run is in
# progress. Read and traverse only: a write beside `root/` must fail rather
# than silently succeed (§4.6). Restored to `WORKDIR_MUTABLE` before cleanup,
# which needs to unlink `root/` out of it.
WORKDIR_SEALED = 0o555
WORKDIR_MUTABLE = 0o755


@contextmanager
def prepared(task: Task, executor: Executor | None = None):
    """Yield a sandbox on a fresh fixture copy, plus that copy's baseline hashes.

    The baseline is taken *after* any fixture variant is applied, so a
    variant's own files never register as a change made by the agent. It is a
    hash map rather than a second copy of the tree: `Ctx.baseline` explains
    why a copy inside the work directory was the wrong shape.

    The work directory around `root/` is sealed for the duration of the run.
    The tool container mounts the whole runs root and executes as this same
    uid, so without it a model that resolves `data/rowcount.txt` against the
    wrong parent creates a directory beside the fixture, writes into it,
    reads it back to confirm, and reports success — while grading, which reads
    `root/`, correctly sees nothing. Two `v7` W07 runs were decided that way
    (`findings.md`, 2026-09-03). Sealed, the same mistake returns
    `Permission denied`, which is a result the agent can act on.
    """
    workdir = Path(
        tempfile.mkdtemp(dir=ensure_runs_root(), prefix=f"llmbench-{task.id}-")
    )
    try:
        root = workdir / "root"
        shutil.copytree(FIXTURES / task.fixture, root)
        if task.variant is not None:
            task.variant(root)

        workdir.chmod(WORKDIR_SEALED)
        yield Sandbox(root, task.fixture, executor), tree_hashes(root)
    finally:
        workdir.chmod(WORKDIR_MUTABLE)
        shutil.rmtree(workdir, ignore_errors=True)


def condition_failures(task: Task, ctx: Ctx) -> tuple[str, ...] | None:
    """Which of a task's declared sub-conditions failed (§10.1).

    A multi-condition `check` returns one boolean, so a recorded failure does
    not say *which* requirement was missed — W07 cannot be told apart from
    "never wrote the file" without reading the transcript. This records the
    breakdown alongside the verdict. It never influences pass/fail: `check`
    remains the single authority, and a task with no `conditions` records
    `null`.
    """
    if task.conditions is None:
        return None
    return tuple(name for name, held in task.conditions(ctx).items() if not held)


def grade(
    task: Task, sandbox: Sandbox, baseline: dict[str, str], outcome: RunOutcome
) -> tuple[bool, int, Ctx]:
    ctx = Ctx(
        root=sandbox.root,
        baseline=baseline,
        answer=outcome.answer,
        calls=outcome.calls,
        expected=load_expected(task.id),
        path_errors=sandbox.path_errors,
        executor=sandbox.executor,
    )
    try:
        passed = bool(task.check(ctx))
    except Exception:
        # A broken assertion must not be reported as a model failure; it will be
        # caught by the oracle gate (§8).
        raise
    return passed, progress_score(task, outcome, ctx, passed), ctx


def run_task(task: Task, driver: Driver, *, executor: Executor | None = None) -> Graded:
    """Grade one run. `executor` decides where commands run (§4.6); omitted,
    they run on the host, which is what the offline tests and one-off scripts
    want. `Driver` itself is unchanged -- a driver that needs the executor
    reads `sandbox.executor`."""
    started = time.monotonic()
    with prepared(task, executor) as (sandbox, baseline):
        outcome = driver(task, sandbox)
        outcome.path_errors = sandbox.path_errors
        passed, progress, ctx = grade(task, sandbox, baseline, outcome)
        failures = condition_failures(task, ctx)
        elapsed = time.monotonic() - started

    return Graded(
        task_id=task.id,
        suite=task.suite,
        passed=passed,
        progress=progress,
        steps=outcome.steps,
        tool_calls=len(outcome.calls),
        invalid_calls=outcome.invalid_calls,
        path_errors=outcome.path_errors,
        termination_reason=outcome.termination_reason,
        wall_clock_s=round(elapsed, 3),
        answer=outcome.answer,
        metrics=outcome.metrics,
        transcript=outcome.transcript,
        condition_failures=failures,
    )
