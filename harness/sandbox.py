"""The tool sandbox, per benchmark.md §4.6.

Every tool returns a string. Errors — including path violations, missing files
and disallowed commands — are returned as ordinary result strings so the model
can recover from them (§4.5). Nothing here raises into the agent loop.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path

TRUNCATE_LIMIT = 4000

# Command segments are split on these so that every segment's leading token can
# be checked against the allowlist. This permits pipes and globs while
# preventing `cat x | sh` from smuggling a disallowed command past the check.
SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;]")

# Command substitution can smuggle anything at all; refuse it outright.
SUBSTITUTION = re.compile(r"\$\(|`")

ALLOWLISTS = {
    "workspace": ("ls", "cat", "grep", "find", "head", "tail", "wc", "python"),
    "testrepo": ("ls", "cat", "grep", "find", "head", "tail", "wc", "python", "pytest"),
}

# Generated artefacts are excluded from every tree comparison; running the test
# suite must not, by itself, count as modifying the repository.
IGNORED = ("__pycache__", ".pytest_cache", ".ruff_cache")


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

    def __init__(self, root: Path, fixture: str) -> None:
        self.root = Path(root).resolve()
        self.fixture = fixture
        self.allowlist = ALLOWLISTS[fixture]
        self.path_errors = 0

    # --- path handling ------------------------------------------------------

    def _resolve(self, path: str) -> Path | str:
        """Resolve a relative path inside the root, or return an error string."""
        if not isinstance(path, str):
            return f"error: path must be a string, got {type(path).__name__}"
        candidate = (self.root / path).resolve()
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

    def run_command(self, command: str) -> str:
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

        env = dict(os.environ)
        venv_bin = Path(__file__).resolve().parent.parent / ".venv" / "bin"
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "exit=124 command timed out after 30s"

        output = (completed.stdout or "") + (completed.stderr or "")
        return f"exit={completed.returncode}\n{output}"
