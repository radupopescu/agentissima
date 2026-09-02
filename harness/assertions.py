"""Assertion helpers shared by both task suites.

Every helper works on final fixture state or final answer text, keeping the
assertions driver-independent (doc/benchmark.md §4.1).
"""

from __future__ import annotations

import ast
import csv
import io
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .execution import ExecutionError
from .sandbox import tree_hashes
from .types import Ctx

# US spellings checked by W07 and T07. Matched as whole words, case-insensitively.
US_SPELLING = re.compile(
    r"\b\w*(?:organiz|summariz|analyz|categoriz|recogniz|prioritiz|normaliz)\w*\b",
    re.IGNORECASE,
)


def contains_number(text: str, value: str | int | Decimal) -> bool:
    """True when ``value`` appears as a standalone number in ``text``.

    Tolerates thousands separators and a currency symbol, and treats 85, 85.0
    and 85.00 as the same number.
    """
    try:
        target = Decimal(str(value))
    except InvalidOperation:
        return False

    for candidate in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            if Decimal(candidate.replace(",", "")) == target:
                return True
        except InvalidOperation:
            continue
    return False


def mentions(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def no_us_spelling(text: str) -> bool:
    return US_SPELLING.search(text) is None


def changed_paths(ctx: Ctx) -> set[str]:
    """Relative paths that differ from the fixture as the run started."""
    before = ctx.baseline
    after = tree_hashes(ctx.root)
    return {
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    }


def unchanged_under(ctx: Ctx, prefix: str) -> bool:
    """True when nothing under ``prefix`` was created, modified or deleted."""
    return not any(path.startswith(prefix) for path in changed_paths(ctx))


def only_changed(ctx: Ctx, allowed: set[str]) -> bool:
    return changed_paths(ctx) <= allowed


def filenames_in(text: str, pattern: str) -> set[str]:
    """Basenames matching ``pattern`` mentioned anywhere in ``text``."""
    return {Path(match).name for match in re.findall(pattern, text)}


def parse_csv(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse CSV text into (header, rows), or None when it will not parse."""
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return None
    rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if not rows:
        return None
    return [cell.strip() for cell in rows[0]], [[cell.strip() for cell in r] for r in rows[1:]]


def function_has_docstring(source: str, name: str) -> bool:
    """True when ``source`` parses and its ``name`` function has a docstring."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) is not None
    return False


def pytest_passes(ctx: Ctx, target: str = "", ignore: str = "") -> bool:
    """Run pytest against the run's fixture and report whether it exited 0.

    Takes `Ctx`, not a bare path, because grading must execute where the run
    executed (§4.6): under a container executor the fixture's tests have to run
    in the container, not on the host. `Ctx` is the right carrier because
    assertions already receive it and nothing else in grading shells out.
    """
    command = "pytest -q"
    if ignore:
        command += f" --ignore={ignore}"
    if target:
        command += f" {target}"

    try:
        result = ctx.executor.run(command, cwd=ctx.root, timeout_s=120.0)
    except ExecutionError:
        return False
    return not result.timed_out and result.exit_code == 0
