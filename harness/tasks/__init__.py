"""Task registry for both suites."""

from __future__ import annotations

from ..types import Task
from . import repo, workspace

ALL_TASKS: list[Task] = workspace.build() + repo.build()
BY_ID: dict[str, Task] = {task.id: task for task in ALL_TASKS}

SUITE_W = [task for task in ALL_TASKS if task.suite == "W"]
SUITE_T = [task for task in ALL_TASKS if task.suite == "T"]

__all__ = ["ALL_TASKS", "BY_ID", "SUITE_W", "SUITE_T", "Task"]
