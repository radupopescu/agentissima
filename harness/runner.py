"""Fixture preparation and grading.

A driver is any callable ``(task, sandbox) -> RunOutcome`` (benchmark.md §4.1).
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

from .sandbox import Sandbox
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


@contextmanager
def prepared(task: Task):
    """Yield a sandbox on a fresh fixture copy, plus a pristine reference copy.

    The pristine snapshot is taken *after* any fixture variant is applied, so a
    variant's own files never register as a change made by the agent.
    """
    workdir = Path(tempfile.mkdtemp(prefix=f"llmbench-{task.id}-"))
    try:
        root = workdir / "root"
        shutil.copytree(FIXTURES / task.fixture, root)
        if task.variant is not None:
            task.variant(root)

        pristine = workdir / "pristine"
        shutil.copytree(root, pristine)

        yield Sandbox(root, task.fixture), pristine
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def grade(task: Task, sandbox: Sandbox, pristine: Path, outcome: RunOutcome) -> tuple[bool, int, Ctx]:
    ctx = Ctx(
        root=sandbox.root,
        pristine=pristine,
        answer=outcome.answer,
        calls=outcome.calls,
        expected=load_expected(task.id),
        path_errors=sandbox.path_errors,
    )
    try:
        passed = bool(task.check(ctx))
    except Exception:
        # A broken assertion must not be reported as a model failure; it will be
        # caught by the oracle gate (§8).
        raise
    return passed, progress_score(task, outcome, ctx, passed), ctx


def run_task(task: Task, driver: Driver) -> Graded:
    started = time.monotonic()
    with prepared(task) as (sandbox, pristine):
        outcome = driver(task, sandbox)
        outcome.path_errors = sandbox.path_errors
        passed, progress, _ = grade(task, sandbox, pristine, outcome)
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
    )
