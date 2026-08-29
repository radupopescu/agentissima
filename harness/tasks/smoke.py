"""Stage 0 — the tool-calling gate (doc/benchmark.md §9 Stage 0).

Three trivial, single-tool-call tasks. Stage 0 checks that the tool-calling
plumbing works, not that the model can reason — so each `check` looks only
for a valid call of the right tool against the right target, never at
whether the final answer is any good.

Deliberately independent of Suite W/T content: every target here is
`README.md`'s fixed text in `fixtures/build_workspace.py`, not anything the
rng assigns, so Stage 0 never moves when a fixture is regenerated.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from ..types import Ctx, Task

SUITE = "0"
FIXTURE = "workspace"
MIN_CONTEXT = 8192

# The line in README.md that S03 searches for. Fixed text (`README_MD` in
# fixtures/build_workspace.py), not derived from the rng seed.
SEARCH_LITERAL = "expense and headcount registers"


def _normalise(path: str) -> str:
    return str(PurePosixPath(path.strip())).rstrip("/")


def _valid_calls(ctx: Ctx, name: str):
    return [call for call in ctx.calls if call.valid and call.name == name]


def _check_s01(ctx: Ctx) -> bool:
    for call in _valid_calls(ctx, "list_files"):
        path = (call.arguments or {}).get("path", "")
        if _normalise(path) in ("", "."):
            return True
    return False


def _check_s02(ctx: Ctx) -> bool:
    for call in _valid_calls(ctx, "read_file"):
        path = (call.arguments or {}).get("path", "")
        if _normalise(path) == "README.md":
            return True
    return False


def _check_s03(ctx: Ctx) -> bool:
    for call in _valid_calls(ctx, "search_files"):
        if "README.md" in call.result:
            return True
    return False


def _has_answer(ctx: Ctx) -> bool:
    return bool(ctx.answer.strip())


def build() -> list[Task]:
    common = dict(suite=SUITE, fixture=FIXTURE, min_context=MIN_CONTEXT)
    return [
        Task(
            id="S01",
            category="tool-call",
            prompt="List the files and directories directly inside the working directory.",
            target_paths=(),
            check=_check_s01,
            shape=_has_answer,
            **common,
        ),
        Task(
            id="S02",
            category="tool-call",
            prompt="Read README.md and say what the workspace contains, in one sentence.",
            target_paths=("README.md",),
            check=_check_s02,
            shape=_has_answer,
            **common,
        ),
        Task(
            id="S03",
            category="tool-call",
            prompt=(
                f'Search the working directory for the exact text "{SEARCH_LITERAL}" '
                "and name the file it appears in."
            ),
            target_paths=("README.md",),
            check=_check_s03,
            shape=_has_answer,
            **common,
        ),
    ]


STAGE0_TASKS: list[Task] = build()
