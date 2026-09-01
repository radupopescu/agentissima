"""Progress score, per doc/benchmark.md §7.4.

0 no valid tool call · 1 valid tool call · 2 read or searched the correct target
· 3 final answer of the right shape · 4 passed.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from .types import Ctx, RunOutcome, Task


# Tools whose *results* can surface a target the call did not name. `native`
# has one; `pi` names its equivalents differently (§4.1). Listing pi's names
# here changes nothing for `native`, which never emits them, and keeps
# progress level 2 reachable the same way under both drivers.
SEARCH_TOOLS = frozenset({"search_files", "grep", "find"})


def _normalise(path: str) -> str:
    return str(PurePosixPath(path.strip().lstrip("./"))).rstrip("/")


def touched_target(outcome: RunOutcome, targets: tuple[str, ...]) -> bool:
    """True when a valid call named a target path, or a search surfaced one."""
    wanted = [_normalise(target) for target in targets if target.strip()]

    for call in outcome.calls:
        if not call.valid:
            continue

        for reference in call.referenced_paths:
            referenced = _normalise(reference)
            for target in wanted:
                if referenced == target:
                    return True
                if referenced.startswith(target + "/"):
                    return True
                if PurePosixPath(referenced).name == PurePosixPath(target).name:
                    return True

        # A search over a wider scope counts when its results surface a target.
        if call.name in SEARCH_TOOLS:
            for target in wanted:
                if target in call.result:
                    return True

    return False


def progress_score(task: Task, outcome: RunOutcome, ctx: Ctx, passed: bool) -> int:
    if passed:
        return 4
    if any(call.valid for call in outcome.calls):
        if task.shape(ctx):
            return 3
        if touched_target(outcome, task.target_paths):
            return 2
        return 1
    return 0
