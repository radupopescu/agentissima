"""The `pi` external coding agent driver (doc/benchmark.md §4.1).

Unlike `NativeDriver`, pi operates on the real filesystem through its own
read/write/edit/bash/find/grep/ls tools -- never through `harness/sandbox.py`.
Containment is therefore the §4.6 container, which pi runs *inside*: it can see
the run's fixture copy and nothing else on the host, reads included.

This replaced a macOS Seatbelt profile that confined writes but permitted all
reads, and pi's own `pi-permission-system`, which cannot inspect `bash` because
a shell command is an opaque string. The extension is still loaded -- the
container is the real boundary now, but removing it would change pi's
observable tool behaviour on top of everything else, and defence in depth costs
nothing.

pi's own agent loop, retries and termination handling are opaque to this
driver (§4.1's deliberate confound). Tool calls are reconstructed from pi's
message log rather than observed as they happen (§5.3), so `invalid_calls`
is not measurable here: pi repairs internally and a malformed call never
reaches the log.
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
# Inside the container these are fixed paths: the image installs the extension
# globally and creates the agent directory (setup/docker/Dockerfile).
CONTAINER_CONFIG_DIR = "/pi-config"
CONTAINER_PERMISSION_EXTENSION = (
    "/usr/local/lib/node_modules/pi-permission-system"
)

WALL_CLOCK_LIMIT_S = 600.0  # matches NativeDriver's default

# §11: any change to the invocation, the seatbelt profile, or the containment
# story bumps this.
#
# "4": a run's message log is assembled from the `message_end` events as they
# stream, instead of only from the terminal `agent_end`. A killed process
# never emits `agent_end`, so every timed-out run recorded an empty
# transcript, `tool_calls: 0` and `progress_score: 0` however far it had got
# — nine runs in `v6`, one of them 59 turns and 288k prompt tokens deep.
#
# "3": pi now runs inside the §4.6 container instead of on the host under a
# Seatbelt profile. Its tool surface is the pinned Linux image, and reads
# outside the fixture are impossible rather than merely discouraged.
#
# "2": `--append-system-prompt` now delivers `task.extra_rules` (W07/T07 were
# graded against rules pi was never sent, while pi auto-loaded the fixture's
# contradicting AGENTS.md), the four discovery flags below isolate ambient
# machine state, and `RunOutcome.calls` is reconstructed from pi's event
# stream so the progress score works.
DRIVER_VERSION = "4"

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


def pi_version(executor=None) -> str | None:
    """pi's version, distinct from this wrapper's `DRIVER_VERSION`.

    Reported from wherever pi will actually run (§4.6): with an executor, from
    inside the tool image; without one, from the host. Reporting the host's
    version for a containerised run would record a version that never ran.
    """
    if executor is not None:
        from pathlib import Path as _Path

        from .paths import ensure_runs_root

        try:
            result = executor.spawn(
                ["pi", "--version"], cwd=_Path(ensure_runs_root()), timeout_s=60
            )
        except Exception:
            return None
        return result.output.strip() or None

    try:
        completed = subprocess.run(
            ["pi", "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


@dataclass
class PiDriver:
    """Callable with the `Driver` signature: (task, sandbox) -> RunOutcome."""

    model: str
    provider: str = "lmstudio"
    wall_clock_limit_s: float = WALL_CLOCK_LIMIT_S
    config_dir: Path = CONFIG_DIR
    pi_binary: str = "pi"
    # Where pi itself runs (§4.6). Defaults to the host, which is the
    # pre-container behaviour; stages pass the container executor.
    executor: Executor = field(default_factory=HostExecutor)

    def __call__(self, task: Task, sandbox: Sandbox) -> RunOutcome:
        # `CONTAINER_CONFIG_DIR` is a container-local directory the image
        # creates; the two authored config files are mounted into it by
        # `container_session`. Deliberately not a bind mount of
        # `setup/pi_config`, which also holds a cached macOS `fd` that pi
        # would find ahead of PATH and fail to execute.
        env = {"PI_CODING_AGENT_DIR": CONTAINER_CONFIG_DIR, "TMPDIR": "/tmp"}

        argv = [
            self.pi_binary, "-p", task.prompt,
            "--provider", self.provider,
            "--model", self.model,
            "--mode", "json",
            "--no-session",
            "--extension", CONTAINER_PERMISSION_EXTENSION,
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

    # `agent_end` carries the whole log and is authoritative when it arrives.
    # It never arrives for a run the wall clock kills, which is why the
    # streamed reconstruction exists (see DRIVER_VERSION "4").
    transcript = agent_end.get("messages") if agent_end is not None else None
    if not transcript:
        transcript = _transcript_from_events(events) or None

    # The answer is read only from a run that settled. A timed-out run's last
    # assistant message is a mid-investigation remark, not an answer, and
    # grading it as one would credit work the model never concluded.
    answer = _final_answer_text(transcript) if agent_end is not None else ""

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


def _transcript_from_events(events: list[dict]) -> list[dict]:
    """The message log, assembled from the stream rather than from `agent_end`.

    `message_end` fires once per settled message and carries the whole message
    object — read from pi 0.84.4's `dist/core/agent-session.js`, where the same
    event drives session persistence for `user`, `assistant` and `toolResult`
    roles alike. So the list this returns is the same shape `agent_end` would
    have given, short by any message still in flight when the process died.

    `custom` messages (extension output) are excluded: they are not part of the
    model's conversation and `agent_end` does not carry them.
    """
    transcript: list[dict] = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") not in ("user", "assistant", "toolResult"):
            continue
        transcript.append(message)
    return transcript


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
