"""The tool sandbox, per doc/benchmark.md §4.6.

Every tool returns a string. Errors — including path violations, missing files
and disallowed commands — are returned as ordinary result strings so the model
can recover from them (§4.5). Nothing here raises into the agent loop.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path

from .execution import TIMEOUT_EXIT_CODE, ExecutionError, Executor, HostExecutor

TRUNCATE_LIMIT = 4000

# Command segments are split on these so that every segment's leading token can
# be checked against the allowlist. This permits pipes and globs while
# preventing `cat x | sh` from smuggling a disallowed command past the check.
# `&` (background) is included alongside `&&`: without it, `wc -l a.txt &
# sleep 5` checks only "wc" and runs "sleep" — not in any allowlist — anyway.
# The `(?<![>&])` exclusion keeps `2>&1`/`1>&2`-style redirection whole: a `&`
# immediately after `>` is part of a redirect target, not a background
# operator. Missing this split a real command in two, refusing it with a
# nonsensical "command not permitted: 1" — found in LFM-G8's Stage 2A data.
SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;]|(?<![>&])&")

# Command substitution can smuggle anything at all; refuse it outright.
SUBSTITUTION = re.compile(r"\$\(|`")

ALLOWLISTS = {
    "workspace": ("ls", "cat", "grep", "find", "head", "tail", "wc", "python"),
    "testrepo": ("ls", "cat", "grep", "find", "head", "tail", "wc", "python", "pytest"),
}

# Generated artefacts are excluded from every tree comparison; running the test
# suite must not, by itself, count as modifying the repository.
IGNORED = ("__pycache__", ".pytest_cache", ".ruff_cache")


def _escapes_sandbox(token: str) -> bool:
    """True if this argument, used as a path, would reach outside the sandbox.

    run_command has no structured path parameter to normalise — the leading-`/`
    root-anchoring the other tools apply doesn't happen here, because `command`
    is handed to a real shell against a real `cwd`. An absolute path or a `..`
    segment in an argument reaches the real filesystem directly."""
    if token.startswith("/") or token.startswith("~"):
        return True
    return any(part == ".." for part in token.split("/"))


def truncate(text: str, limit: int = TRUNCATE_LIMIT) -> str:
    """Apply the §4.6 truncation rule."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated, {len(text) - limit} more characters]"


def _is_ignored(relative: Path) -> bool:
    return any(part in IGNORED or part.endswith(".pyc") for part in relative.parts)


def tree_hashes(root: Path) -> dict[str, str]:
    """Map relative path -> content hash, skipping generated artefacts."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_ignored(relative):
            continue
        out[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


class Sandbox:
    """A rooted, non-networked working directory exposing the five tools."""

    def __init__(
        self, root: Path, fixture: str, executor: Executor | None = None
    ) -> None:
        self.root = Path(root).resolve()
        self.fixture = fixture
        self.allowlist = ALLOWLISTS[fixture]
        self.path_errors = 0
        # Where `run_command` actually runs (§4.6). Defaults to the host so a
        # bare `Sandbox(root, fixture)` still works in tests and one-off
        # scripts; stages and gates pass the container executor explicitly.
        self.executor: Executor = executor if executor is not None else HostExecutor()

    # --- path handling ------------------------------------------------------

    def _resolve(self, path: str) -> Path | str:
        """Resolve a path inside the root, or return an error string.

        A leading ``/`` is root-anchored *within the sandbox*, as under chroot:
        the root is the model's entire visible filesystem, so "relative to the
        root" and "absolute from the root" denote the same location. Without
        this, ``root / "/x"`` discards the root (pathlib semantics), escapes to
        the real filesystem, and is refused with a message saying the path is
        outside the working directory — which is false, and unactionable.

        ``..`` traversal is still refused, including after a leading slash.
        """
        if not isinstance(path, str):
            return f"error: path must be a string, got {type(path).__name__}"
        candidate = (self.root / path.lstrip("/")).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            self.path_errors += 1
            return "error: path outside working directory"
        return candidate

    # --- tools --------------------------------------------------------------

    def read_file(self, path: str) -> str:
        resolved = self._resolve(path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.exists():
            self.path_errors += 1
            return f"error: no such file: {path}"
        if resolved.is_dir():
            self.path_errors += 1
            return f"error: {path} is a directory, not a file"
        try:
            return resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"error: {path} is not valid UTF-8 text"

    def write_file(self, path: str, content: str) -> str:
        resolved = self._resolve(path)
        if isinstance(resolved, str):
            return resolved
        if not isinstance(content, str):
            return f"error: content must be a string, got {type(content).__name__}"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return "ok"

    def list_files(self, path: str) -> str:
        resolved = self._resolve(path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.exists():
            self.path_errors += 1
            return f"error: no such directory: {path}"
        if not resolved.is_dir():
            self.path_errors += 1
            return f"error: {path} is a file, not a directory"
        entries = []
        for child in sorted(resolved.iterdir()):
            if _is_ignored(child.relative_to(self.root)):
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
        return "\n".join(entries) if entries else "(empty directory)"

    def search_files(self, pattern: str, path: str = ".") -> str:
        resolved = self._resolve(path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.exists():
            self.path_errors += 1
            return f"error: no such path: {path}"
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"error: invalid regular expression: {exc}"

        targets = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        matches = []
        for target in targets:
            if not target.is_file():
                continue
            relative = target.relative_to(self.root)
            if _is_ignored(relative):
                continue
            try:
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{relative}:{number}:{line.strip()}")
        return "\n".join(matches) if matches else "(no matches)"

    def run_command(self, command: str, timeout_s: float = 30.0) -> str:
        if not isinstance(command, str):
            return f"error: command must be a string, got {type(command).__name__}"
        if SUBSTITUTION.search(command):
            return "exit=127 command substitution is not permitted"

        for segment in SEGMENT_SPLIT.split(command):
            segment = segment.strip()
            if not segment:
                continue
            try:
                tokens = shlex.split(segment)
            except ValueError as exc:
                return f"exit=127 could not parse command: {exc}"
            if not tokens:
                continue
            head = Path(tokens[0]).name
            if head not in self.allowlist:
                return f"exit=127 command not permitted: {tokens[0]}"
            for token in tokens:
                if _escapes_sandbox(token):
                    return f"exit=127 path outside working directory: {token}"

        # Validation above is host-side and stays here: the allowlist, the
        # segment split and `_escapes_sandbox` are what §4.5 measures, and they
        # must be identical whichever executor runs the command.
        try:
            result = self.executor.run(command, cwd=self.root, timeout_s=timeout_s)
        except ExecutionError as exc:
            return f"exit=127 could not start command: {exc}"

        if result.timed_out:
            return f"exit={TIMEOUT_EXIT_CODE} command timed out after {timeout_s:g}s"
        return f"exit={result.exit_code}\n{result.output}"
