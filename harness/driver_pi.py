"""The `pi` external coding agent driver (doc/benchmark.md §4.1).

Unlike `NativeDriver`, pi operates on the real filesystem through its own
read/write/edit/bash/find/grep/ls tools -- never through `harness/sandbox.py`.
Containment therefore comes from two independent layers: pi's own
`pi-permission-system` extension (denies `read`/`write`/`edit`/`find`/`grep`/
`ls` outside the working directory) and a per-run macOS Seatbelt profile this
driver wraps every invocation in (denies *writes* anywhere but the fixture
copy, the isolated pi config directory, and an isolated temp directory --
kernel-enforced, not string matching). `bash` reads outside the working
directory are not blocked by either layer; this was a deliberate, recorded
trade-off (see doc/findings.md) rather than an oversight -- denying `bash`
outright would stop pi self-verifying Suite T's T03/T09 with pytest.

pi's own agent loop, retries and termination handling are opaque to this
driver (§4.1's deliberate confound): `RunOutcome.calls` stays empty (pi's
tool calls never go through `harness/tools.py`, so §4.5's no-repair
accounting doesn't apply to them) and metrics are only whatever `--mode json`
exposes (§5.3: metric availability is per driver, nulls are never estimated).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .execution import ExecutionError, Executor, HostExecutor
from .sandbox import Sandbox
from .tools import ToolCall
from .types import RunOutcome, Task

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "setup" / "pi_config"

# The pi-permission-system extension is loaded by path (--extension),
# independent of PI_CODING_AGENT_DIR, so the isolated config directory above
# never needs its own npm install of it. This does mean the driver depends on
# the user's personal global pi installation having the package present.
PERMISSION_EXTENSION = (
    Path.home() / ".pi" / "agent" / "npm" / "node_modules" / "pi-permission-system"
)

WALL_CLOCK_LIMIT_S = 600.0  # matches NativeDriver's default

# §11: any change to the invocation, the seatbelt profile, or the containment
# story bumps this.
#
# "2": `--append-system-prompt` now delivers `task.extra_rules` (W07/T07 were
# graded against rules pi was never sent, while pi auto-loaded the fixture's
# contradicting AGENTS.md), the four discovery flags below isolate ambient
# machine state, and `RunOutcome.calls` is reconstructed from pi's event
# stream so the progress score works.
DRIVER_VERSION = "2"

# Ambient state must not leak into a run. `PI_CODING_AGENT_DIR` already
# isolates pi's *global* discovery slot, but project-local discovery resolves
# against `cwd` — the fixture copy (`dist/core/skills.js` resolves
# `<cwd>/.pi/skills`). Nothing loads from the current fixtures, so this is
# structural rather than a fix for a live defect.
#
# `--no-extensions` still honours explicit `-e` paths, so the permission
# extension survives it.
#
# Deliberately NOT pinned: `--tools`/`--exclude-tools`, `--thinking`,
# `--system-prompt`. Those are pi's identity as a harness; freezing them
# yields "pi as configured in August 2026", which decays as pi improves and
# undoes the reason for using it (§4.1). Recording pi's version instead makes
# that drift detectable.
ISOLATION_FLAGS = (
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-approve",
)

# Context files are NOT disabled. pi auto-loads the fixture's adversarial
# AGENTS.md from the working directory root, which is what a production
# harness does and what makes W07/T07's instruction conflict live rather than
# contrived. It does mean those two tasks are not cross-driver comparable:
# `native` exposes AGENTS.md only if the model chooses to read it.


def pi_version(binary: str = "pi") -> str | None:
    """pi's own version, distinct from this wrapper's `DRIVER_VERSION`."""
    try:
        completed = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _sb_string(value: str) -> str:
    """Escape a path for use inside a Seatbelt (Scheme) string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _seatbelt_profile(writable: list[Path]) -> str:
    """A minimal write-confinement profile: allow everything by default, then
    deny all writes except under the given paths. Verified directly (see
    doc/findings.md): a write attempted outside these paths from inside pi's
    `bash` tool fails with "Operation not permitted" -- real, kernel-enforced
    containment, unlike pi-permission-system's own unparsed bash string
    patterns."""
    lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
    for path in writable:
        lines.append(f'(allow file-write* (subpath "{_sb_string(str(path))}"))')
    # Apple's per-user temp tree: some Node/npm internals write here even
    # with TMPDIR overridden. Not the user's real files -- low value target.
    lines.append('(allow file-write* (regex #"^/private/var/folders/"))')
    lines.append('(allow file-write* (subpath "/dev"))')
    return "\n".join(lines) + "\n"


@dataclass
class PiDriver:
    """Callable with the `Driver` signature: (task, sandbox) -> RunOutcome."""

    model: str
    provider: str = "lmstudio"
    wall_clock_limit_s: float = WALL_CLOCK_LIMIT_S
    config_dir: Path = CONFIG_DIR
    permission_extension: Path = PERMISSION_EXTENSION
    pi_binary: str = "pi"
    # Where pi itself runs (§4.6). Defaults to the host, which is the
    # pre-container behaviour; stages pass the container executor.
    executor: Executor = field(default_factory=HostExecutor)

    def __call__(self, task: Task, sandbox: Sandbox) -> RunOutcome:
        workdir = sandbox.root.parent
        tmp_dir = workdir / "pi-tmp"
        tmp_dir.mkdir(exist_ok=True)

        profile_path = workdir / "pi-seatbelt.sb"
        profile_path.write_text(
            _seatbelt_profile([sandbox.root, self.config_dir, tmp_dir]),
            encoding="utf-8",
        )

        env = {
            "PI_CODING_AGENT_DIR": str(self.config_dir),
            "TMPDIR": str(tmp_dir),
        }

        argv = [
            "sandbox-exec", "-f", str(profile_path),
            self.pi_binary, "-p", task.prompt,
            "--provider", self.provider,
            "--model", self.model,
            "--mode", "json",
            "--no-session",
            "--extension", str(self.permission_extension),
            *ISOLATION_FLAGS,
        ]

        # Delivering the task's rules is part of the task definition (§4.3),
        # not a per-driver choice. `--append-system-prompt` appends to pi's own
        # prompt rather than replacing it, so pi stays pi: verified in
        # `dist/core/system-prompt.js`, which orders the assembled prompt as
        # base -> appended text -> <project_context> -> skills.
        if task.extra_rules:
            argv += ["--append-system-prompt", task.extra_rules]

        try:
            result = self.executor.spawn(
                argv, cwd=sandbox.root, timeout_s=self.wall_clock_limit_s, env=env
            )
        except ExecutionError:
            return RunOutcome(
                task_id=task.id,
                root=sandbox.root,
                answer="",
                termination_reason="server_error",
            )

        stdout, timed_out = result.output, result.timed_out

        return _outcome_from_output(task, sandbox, stdout, timed_out=timed_out)


def _outcome_from_output(
    task: Task, sandbox: Sandbox, stdout: str, *, timed_out: bool
) -> RunOutcome:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A non-JSON banner/warning line pi prints before the stream
            # starts (e.g. "Warning: Model ... not found ..."), not a
            # protocol error.
            continue

    steps = sum(1 for event in events if event.get("type") == "turn_start")
    agent_end = next((e for e in reversed(events) if e.get("type") == "agent_end"), None)

    answer = ""
    transcript = None
    if agent_end is not None:
        transcript = agent_end.get("messages")
        answer = _final_answer_text(transcript)

    if timed_out:
        termination = "timeout"
    elif agent_end is None:
        # pi exited (crashed, or produced no parseable stream) without ever
        # settling -- an infrastructure fault, not a model mistake, the same
        # distinction NativeDriver draws for a mid-stream APIError.
        termination = "server_error"
    else:
        termination = "final_answer" if answer.strip() else "empty_answer"

    return RunOutcome(
        task_id=task.id,
        root=sandbox.root,
        answer=answer,
        calls=_calls_from_transcript(transcript),
        termination_reason=termination,
        steps=steps,
        metrics=_metrics_from_events(events),
        transcript=transcript,
    )


# pi names a path argument `path` on every path-taking tool (read, write,
# edit, ls, and optionally grep/find). `bash` has no structured path
# parameter, so it contributes none — the same limitation `run_command` has
# under `native` (§4.6).
_PATH_KEYS = ("path", "file_path")


def _calls_from_transcript(transcript: list[dict] | None) -> list[ToolCall]:
    """Reconstruct the tool calls pi made, from its own message log.

    Without this `RunOutcome.calls` is empty and `progress_score` collapses to
    0-or-4 for every pi run, disabling §1.2's stated mechanism for
    discriminating when pass rates are near zero.

    `valid` is always True and `error` always None: pi validates, repairs and
    retries internally, so a malformed call never reaches this log. §4.5's
    no-repair accounting is therefore not measurable under pi and must not be
    inferred — `invalid_calls` is 0 because nothing was observed, not because
    nothing went wrong.
    """
    calls: list[ToolCall] = []
    if not transcript:
        return calls

    results: dict[str, str] = {}
    for message in transcript:
        if message.get("role") != "toolResult":
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                results.setdefault(message.get("toolCallId") or "", block.get("text", ""))

    for message in transcript:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            arguments = block.get("arguments")
            arguments = arguments if isinstance(arguments, dict) else {}
            referenced = tuple(
                str(arguments[key]) for key in _PATH_KEYS
                if isinstance(arguments.get(key), str)
            )
            calls.append(
                ToolCall(
                    name=block.get("name") or "",
                    raw_arguments=json.dumps(arguments, sort_keys=True),
                    arguments=arguments,
                    valid=True,
                    result=results.get(block.get("id") or "", ""),
                    referenced_paths=referenced,
                )
            )
    return calls


def _final_answer_text(transcript: list[dict] | None) -> str:
    if not transcript:
        return ""
    for message in reversed(transcript):
        if message.get("role") != "assistant":
            continue
        return "".join(
            block.get("text", "")
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _metrics_from_events(events: list[dict]) -> dict:
    """Whatever `--mode json` exposes -- per-turn token usage on each
    assistant `message_end`. Everything else (TTFT, generation throughput)
    stays null: pi's stream carries no inter-token timing to derive it from,
    and §5.3 forbids estimating it."""
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    turns = 0
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        prompt_tokens += usage.get("input") or 0
        completion_tokens += usage.get("output") or 0
        reasoning_tokens += usage.get("reasoning") or 0
        turns += 1

    return {
        "final_finish_reason": None,
        "ttft_s": None,
        "ttft_turn1_s": None,
        "ttft_median_later_s": None,
        "gen_tps": None,
        "prompt_tps": None,
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": completion_tokens or None,
        "reasoning_tokens": reasoning_tokens or None,
        "total_tokens": (prompt_tokens + completion_tokens) or None,
        "turns": turns or None,
    }
