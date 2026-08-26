"""Shared types for tasks and runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .tools import ToolCall

EXPECTED_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "expected"


def load_expected(task_id: str) -> dict:
    """Read a task's expected values, emitted by the fixture generator (§6)."""
    return json.loads((EXPECTED_DIR / f"{task_id}.json").read_text(encoding="utf-8"))


@dataclass
class Ctx:
    """Everything an assertion may inspect.

    Assertions read final fixture state and the final answer only — never the
    transcript's internal structure — so that they remain valid under any
    driver (§4.1).
    """

    root: Path
    pristine: Path
    answer: str
    calls: list[ToolCall]
    expected: dict
    path_errors: int

    def read(self, relative: str) -> str | None:
        path = self.root / relative
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None


@dataclass(frozen=True)
class Task:
    id: str
    suite: str  # "W" or "T"
    category: str
    fixture: str  # "workspace" or "testrepo"
    prompt: str
    min_context: int
    target_paths: tuple[str, ...]
    check: Callable[[Ctx], bool]
    shape: Callable[[Ctx], bool]
    extra_rules: str | None = None
    variant: Callable[[Path], None] | None = None


@dataclass
class RunOutcome:
    """The result of one driver executing one task."""

    task_id: str
    root: Path
    answer: str
    calls: list[ToolCall] = field(default_factory=list)
    termination_reason: str = "final_answer"
    steps: int = 0
    path_errors: int = 0
    # Populated by `native`; left None by drivers that cannot produce §5
    # metrics. A null is never replaced by an estimate (§5.3).
    metrics: dict | None = None

    @property
    def invalid_calls(self) -> int:
        return sum(1 for call in self.calls if not call.valid)
