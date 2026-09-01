"""Where a command runs, per doc/benchmark.md §4.6.

The agent's tools and *grading* both execute commands — T03 and T09 run
`pytest` to decide their verdict — so both go through one boundary rather than
each reimplementing subprocess handling. `HostExecutor` runs on the machine the
harness runs on; `ContainerExecutor` (harness/container.py) runs in the pinned
Linux tool image.

**`cwd` is always a host path.** Mapping it into a container is that executor's
private business, so `Ctx.root`, `Sandbox.root` and every assertion keep
working with host paths and nothing above this boundary changes shape.

There is deliberately no ambient or context-local "current executor": an
implicit one would let a stage silently run half its work on the host, which is
exactly the failure this boundary exists to make impossible.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

REPO_ROOT = Path(__file__).resolve().parent.parent

# §4.6: a command that overruns is reported as exit 124, matching `timeout(1)`
# and the string `Sandbox.run_command` has always returned.
TIMEOUT_EXIT_CODE = 124


class ExecutionError(RuntimeError):
    """The executor could not run the command at all.

    Distinct from a command that ran and failed: that is an ordinary
    `CommandResult` with a non-zero exit code, which §4.5 requires be handed
    back to the model as a tool result rather than raised.
    """


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    # Combined stdout and stderr, in the order `run_command` has always
    # concatenated them.
    output: str
    timed_out: bool = False


@runtime_checkable
class Executor(Protocol):
    """Runs one command in the fixture environment."""

    def run(self, command: str, *, cwd: Path, timeout_s: float) -> CommandResult:
        """Run `command` through a shell. The `run_command` tool and
        `pytest_passes` both take this path."""
        ...

    def spawn(
        self, argv: list[str], *, cwd: Path, timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run `argv` directly, with no shell. External agents (`pi`, later
        `opencode`) take this path: they stream JSON on stdout and must not be
        passed through a shell."""
        ...

    @property
    def provenance(self) -> dict:
        """What produced this execution environment, recorded verbatim in
        `environment.json` (§3)."""
        ...


def _tool_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a host-side command runs under.

    The `.venv/bin` prepend is what makes `python` and `pytest` — both on the
    §4.6 allowlist — resolve to this project's interpreter rather than whatever
    the operator's shell happens to have. It is host-specific by nature; the
    container image puts them on `PATH` directly and needs none of this.
    """
    env = dict(os.environ)
    venv_bin = REPO_ROOT / ".venv" / "bin"
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


@dataclass(frozen=True)
class HostExecutor:
    """Runs commands directly on the machine the harness runs on.

    This is what the benchmark used before §4.6's container, and it remains the
    executor the offline tests run against. It confines nothing: containment is
    the container's job.
    """

    def run(self, command: str, *, cwd: Path, timeout_s: float) -> CommandResult:
        # `start_new_session=True` puts the shell and everything it spawns in
        # their own process group. A plain timeout kills only the shell, so a
        # `grep -r` it forked keeps running, orphaned -- once observed still
        # scanning the disk an hour after being "killed" (findings.md). Killing
        # the whole group is what actually stops the work.
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=_tool_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ExecutionError(f"could not start command: {exc}") from exc

        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            process.wait()
            return CommandResult(TIMEOUT_EXIT_CODE, "", timed_out=True)

        return CommandResult(process.returncode, (stdout or "") + (stderr or ""))

    def spawn(
        self, argv: list[str], *, cwd: Path, timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=_tool_env(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ExecutionError(f"could not start {argv[0]!r}: {exc}") from exc

        timed_out = False
        try:
            stdout, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(process)
            stdout, _ = process.communicate()

        return CommandResult(process.returncode, stdout or "", timed_out=timed_out)

    @property
    def provenance(self) -> dict:
        return {"mode": "host"}


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass
class RecordingExecutor:
    """Test double: records what it was asked to run and executes nothing.

    Lets a test assert that a command was refused *before* reaching execution,
    which is stronger than inferring it from the returned error string (§4.5:
    the refusal is the measurement, so it must not depend on the command being
    harmless to run).
    """

    calls: list[tuple[str, Path]] = None  # type: ignore[assignment]
    result: CommandResult = CommandResult(0, "")

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def run(self, command: str, *, cwd: Path, timeout_s: float) -> CommandResult:
        self.calls.append((command, cwd))
        return self.result

    def spawn(
        self, argv: list[str], *, cwd: Path, timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        self.calls.append((" ".join(argv), cwd))
        return self.result

    @property
    def provenance(self) -> dict:
        return {"mode": "recording"}
