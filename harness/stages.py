"""Stage runner (doc/benchmark.md §9).

Loads a model once, runs a set of tasks across repetitions, and writes §10.1
JSONL records — the load/unload discipline of §9.0, generalised so most of
§9's stages are a different task list, repetition count and context, not new
code: `run_stage()` backs Stage 0/2A/2B/3 and Stage 5B's compaction variant.
Stage 1 is raw inference rather than task-based, so it has its own inner
loop (`run_stage1()`), but shares the same load/capture/resume preamble via
`_model_session()`.

Session identity is deterministic (`<config_id>-<context_length>`, no
timestamp), so re-running the same stage command lands in the same session
directory and `results.existing_keys()` on that stage's own raw file does the
actual resume work — no separate flag. The one risk that creates — resuming
into a session captured under a different harness revision or
`task_set_version` and silently pooling incomparable runs — is guarded by
`_check_task_set_version_matches` before anything new is written.

A task whose `min_context` exceeds the stage's `context_length` is skipped
entirely — no runs, no records — rather than scored as a failure; it isn't
solvable at this context by construction, and counting it as a fail would
understate a configuration that is otherwise capable at a context it hasn't
been given (§9 Stage 4's "larger context is not assumed to be better" cuts
both ways: a task also should not be penalised for a context it wasn't run
at).

`run_stages(config_id, stage_names, driver=...)` is the one general sequencer:
it runs `stage_names` in order under either driver (§4.1), stopping at the
first stage that doesn't complete or fails its gate — no further stages
attempted for that `(config, driver)` pair. It's a `STEPS` registry
(`stage0`/`stage1`/`stage2a`/`stage2b`) plus that loop, not several
near-duplicate functions for different stage/driver combinations; adding a
later stage is a registry entry, not a new function. Stage 0/1 always run
`native` regardless of `driver` — Stage 0 specifically tests *our* loop, and
Stage 1 has no tool use at all, so a driver distinction is meaningless there.

Stage 3 and beyond are deliberately outside `run_stages()` and called
independently. Stage 4's trigger ("only where Stage 3 showed failures
attributable to context limits") is a judgement about failure *cause*, not a
mechanical threshold like Stage 0/2A's gates, and §9.2 calls each gate "an
explicit go/no-go decision point, not a formality" — chaining past Stage 3
automatically would be exactly that. Stage 5B is a separate, standalone
experiment (`run_stage5b_compact()`) that never feeds the controlled
comparison, so it isn't part of the registry either.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import environment, lmstudio, results
from .admissibility import UNSUPPORTED, classify_declared
from .client import LMStudioClient
from .container import container_session
from .execution import Executor
from .driver_native import NativeDriver
from .driver_pi import PiDriver
from .metrics import MemorySampler, SwapWindow, find_inference_pid, nonce_prefix, turn_metrics
from .runner import Driver, run_task
from .tasks import SUITE_T, SUITE_W
from .tasks.smoke import STAGE0_TASKS
from .types import Task
from .version import TASK_SET_VERSION

RESULTS_DIR = Path("results")

# §4.6: both drivers share one network policy. `pi` must reach LM Studio on the
# host, and `native`'s allowlist includes `python`, which can open sockets --
# so giving only one of them egress would introduce an asymmetry rather than
# remove one. Restricting both to LM Studio alone is a separate change.
CONTAINER_NETWORK = "bridge"
STAGE_IDENTIFIER = "bench"
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "prompts"

# §9 Stage 0: "fewer than 2 of 3 valid tool calls" for 3 tasks × 3 repetitions
# (9 runs) is read as an aggregate rate, not a per-task count — see
# doc/benchmark.md §9 Stage 0 for the reasoning.
STAGE0_REPETITIONS = 3
STAGE0_GATE_RATE = 2 / 3

# 3 repetitions per task for every agent stage past Stage 0 (§9 Stage
# 2A/2B; Stage 3 doesn't restate it, but nothing suggests a different
# methodology there, so the same figure is used and documented in
# benchmark.md).
AGENT_REPETITIONS = 3

# §9 Stage 2A: "passes ≥3 of 10 on Suite W or has a mean progress score
# ≥2.5" over 3 repetitions per task. Neither half of that OR says how 3
# per-task repetitions collapse into one task-level pass/fail, so a task
# counts toward "3 of 10" on a strict majority of its repetitions (≥2 of 3) —
# the standard resolution of a repeated binary trial, and well-defined here
# since 3 is odd. `mean progress score` is the mean over every included run
# (all repetitions of every task that was not min_context-skipped), not a
# per-task mean of means — with uniform repetitions per task the two
# coincide, and this is simpler when they don't. See doc/benchmark.md §9
# Stage 2A for the recorded reasoning.
STAGE2A_MIN_PASSES = 3
STAGE2A_MIN_MEAN_PROGRESS = 2.5

# §9 Stage 1: 8K/16K tiers, 5 repetitions, a repetition counts only if
# completion_tokens >= 128 or is retried once with the alternate prompt.
STAGE1_CONTEXT = {"8k": 8192, "16k": 16384}
STAGE1_REPETITIONS = 5
STAGE1_MIN_COMPLETION_TOKENS = 128

# §9 Stage 5B's context-compaction experiment.
STAGE5B_REPETITIONS = AGENT_REPETITIONS


class SessionMismatchError(RuntimeError):
    """A resumed session's raw records were written under a different
    `task_set_version` than the one running now. Refuses to pool them
    silently."""


class UnsupportedContextError(RuntimeError):
    """Raised by `_model_session` before any load is attempted: the
    configuration's advertised maximum context is below what was requested
    (§2.2's `unsupported`)."""


@dataclass
class StageOutcome:
    status: str  # "completed" | "unsupported" | "oversized"
    session_dir: Path | None = None
    raw_path: Path | None = None
    records: list[dict] = field(default_factory=list)
    detail: str | None = None
    # Task IDs excluded because task.min_context > context_length (not run,
    # not scored as a failure).
    skipped_min_context: list[str] = field(default_factory=list)


@dataclass
class Stage0Outcome:
    tool_capable: bool
    valid_runs: int
    total_runs: int
    stage: StageOutcome


@dataclass
class Stage2AOutcome:
    proceeds: bool
    tasks_passed: int
    tasks_total: int
    mean_progress: float
    stage: StageOutcome


@dataclass
class StagesRunOutcome:
    """The result of one `run_stages()` call: every stage attempted, in
    order, plus which one (if any) stopped the sequence."""

    config_id: str
    driver: str
    stage_names: list[str]
    results: dict[str, Any] = field(default_factory=dict)
    stopped_at: str | None = None


def _check_task_set_version_matches(raw_path: Path) -> None:
    """§11: a `task_set_version` bump is what marks results incomparable —
    covering a fixture, task, prompt, tool schema, sandbox limit or driver
    version change. Comparing the full `environment_sha256` instead was tried
    and rejected: it also covers `free_memory_bytes` and
    `swap_used_bytes_start`, both instant-in-time machine readings that drift
    between any two captures regardless of comparability, so it never
    actually matched twice and defeated resume outright — caught by a live
    verification run where a second capture of the same session, seconds
    later, changed only those two fields and nothing else."""
    for record in results.read_records(raw_path):
        if record["task_set_version"] != TASK_SET_VERSION:
            raise SessionMismatchError(
                f"{raw_path} already holds records written under "
                f"task_set_version {record['task_set_version']!r}, but this "
                f"run is {TASK_SET_VERSION!r}. Resuming would pool "
                "incomparable results into one file. Move or remove the old "
                "raw file if the change was intended."
            )


@dataclass
class ModelSession:
    """One load/unload bracket (§9.0): everything a stage needs once the
    model is resident, shared by every stage runner via `_model_session()`."""

    session_id: str
    session_dir: Path
    raw_path: Path
    transcripts_dir: Path
    env: environment.SessionEnvironment
    client: LMStudioClient
    overhead: float
    pid: int | None
    existing: set[tuple]
    # Where the stage's tool commands run (§4.6). One container per stage.
    executor: Executor


@contextmanager
def _model_session(
    config_id: str,
    stage_name: str,
    context_length: int,
    *,
    results_dir: Path,
    configs_dir: Path | None,
    verify_hash: bool,
    driver: str = "native",
) -> Iterator[ModelSession]:
    """Load `config_id` once, capture the environment, and yield everything a
    stage loop needs. Raises `UnsupportedContextError` before any load is
    attempted, or lets `lmstudio.ModelOversizedError` propagate from the load
    itself — both are for the caller to turn into a `StageOutcome`, since
    Stage 0/2A/etc. and Stage 1 want that outcome shaped slightly
    differently.
    """
    resolved = environment.load_resolved(config_id, configs_dir)
    if classify_declared(resolved["advertised_max_context"], context_length) == UNSUPPORTED:
        raise UnsupportedContextError(
            f"{config_id} advertises a maximum context of "
            f"{resolved['advertised_max_context']}, below the requested "
            f"{context_length}"
        )

    session_id = f"{config_id}-{context_length}"
    session_dir = Path(results_dir) / session_id
    raw_path = session_dir / "raw" / f"{stage_name}.jsonl"
    transcripts_dir = session_dir / "transcripts"

    # The container is entered *outside* the model load: it starts in 0.3 s and
    # a failure here must not waste the minutes a load costs.
    with container_session(network=CONTAINER_NETWORK) as executor, lmstudio.loaded(
        resolved["model_path"],
        context_length=context_length,
        identifier=STAGE_IDENTIFIER,
    ):
        env = environment.capture(
            config_id,
            context_length,
            driver=driver,
            out_dir=results_dir,
            configs_dir=configs_dir,
            verify_hash=verify_hash,
            session_id=session_id,
            executor=executor,
        )

        existing = results.existing_keys(raw_path)
        if existing:
            _check_task_set_version_matches(raw_path)

        client = LMStudioClient(model=STAGE_IDENTIFIER)
        overhead = client.measure_overhead()
        # After overhead calibration: MLX allocates lazily and is not yet
        # identifiable immediately after load (§5.2 defect note).
        found = find_inference_pid()
        pid = found[0] if found else None

        yield ModelSession(
            session_id=session_id,
            session_dir=session_dir,
            raw_path=raw_path,
            transcripts_dir=transcripts_dir,
            env=env,
            client=client,
            overhead=overhead,
            pid=pid,
            existing=existing,
            executor=executor,
        )


def _record_for(
    task: Task,
    repetition: int,
    *,
    config_id: str,
    suite: str,
    session_id: str,
    environment_sha256: str,
    context_length: int,
    driver: NativeDriver,
    driver_label: str,
    pid: int | None,
    transcripts_dir: Path,
    executor: Executor | None = None,
) -> dict:
    # The driver belongs in the identifier, not just in the stage's file name.
    # Without it two drivers' runs of the same task share a `run_id` and so a
    # transcript path, and the later stage silently overwrites the earlier
    # one's transcripts — which is exactly what happened to the `v4` `native`
    # Suite W/T transcripts (see implementation-plan.md's defect table).
    run_id = f"{config_id}-{driver_label}-{suite}-{task.id}-r{repetition}"

    sampler = MemorySampler(pid).start() if pid is not None else None
    with SwapWindow() as swap:
        graded = run_task(task, driver, executor=executor)
    peak = sampler.stop() if sampler is not None else None

    transcript_path = None
    if graded.transcript is not None:
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        tpath = transcripts_dir / f"{run_id}.json"
        tpath.write_text(
            json.dumps(graded.transcript, indent=2, default=str), encoding="utf-8"
        )
        transcript_path = str(tpath)

    metrics = graded.metrics or {}
    return {
        "run_id": run_id,
        "session_id": session_id,
        "config_id": config_id,
        "driver": driver_label,
        "suite": suite,
        "task_id": task.id,
        "repetition": repetition,
        "environment_sha256": environment_sha256,
        "context_length": context_length,
        "task_set_version": TASK_SET_VERSION,
        "ttft_s": metrics.get("ttft_s"),
        "gen_tps": metrics.get("gen_tps"),
        "prompt_tps": metrics.get("prompt_tps"),
        "ttft_turn1_s": metrics.get("ttft_turn1_s"),
        "ttft_median_later_s": metrics.get("ttft_median_later_s"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "total_tokens": metrics.get("total_tokens"),
        "peak_memory_bytes": peak,
        "swap_delta_bytes": swap.delta_bytes,
        "swap_flag": swap.flagged,
        "steps": graded.steps,
        "tool_calls": graded.tool_calls,
        "invalid_calls": graded.invalid_calls,
        "path_errors": graded.path_errors,
        "termination_reason": graded.termination_reason,
        "passed": graded.passed,
        "progress_score": graded.progress,
        # Which named sub-conditions failed, for a task that declares a
        # breakdown; null otherwise. Diagnostic only -- `passed` is the
        # verdict (§10.1).
        "condition_failures": (
            list(graded.condition_failures)
            if graded.condition_failures is not None else None
        ),
        # Deciding this needs every repetition of a task, which may not all
        # exist yet under resume; left to `harness/report.py` (§9.1).
        "flaky": None,
        "wall_clock_s": graded.wall_clock_s,
        "transcript_path": transcript_path,
    }


def _driver_factory(
    driver: str, *, history_mode: str = "full"
) -> Callable[[ModelSession], Driver]:
    """The one place that branches on driver name. Stage 0/1 never call this
    (native-only, per §4.1 -- Stage 0 specifically tests *our* loop, Stage 1
    has no tool use at all); Stage 2A/2B/3 use it to build whichever driver
    `run_stage()` should run this session under.

    `history_mode` only means anything for `native` (Stage 5B's compaction
    experiment); `pi` manages its own history and ignores it."""
    if driver == "native":
        return lambda session: NativeDriver(
            client=session.client, overhead_s=session.overhead, history_mode=history_mode
        )
    if driver == "pi":
        return lambda session: PiDriver(
            model=STAGE_IDENTIFIER, executor=session.executor
        )
    raise ValueError(f"unknown driver: {driver!r}")


def run_stage(
    config_id: str,
    tasks: list[Task],
    *,
    stage_name: str,
    suite: str,
    context_length: int,
    repetitions: int,
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
    verify_hash: bool = True,
    history_mode: str = "full",
    driver_label: str = "native",
    driver_factory: Callable[[ModelSession], Driver] | None = None,
) -> StageOutcome:
    """Run `tasks` × `repetitions` against `config_id` at `context_length`,
    loading the model once (§9.0) and writing §10.1 records as they complete.

    `driver_factory` builds the driver from the loaded `ModelSession`;
    defaults to `native` so every existing call site is unchanged. Records
    from different drivers are never pooled (§4.1) -- callers running under
    `pi` must also give a driver-specific `stage_name`, so records land in
    their own raw file (the same pattern Stage 5B already uses for its
    compaction experiment).
    """
    try:
        with _model_session(
            config_id, stage_name, context_length,
            results_dir=results_dir, configs_dir=configs_dir, verify_hash=verify_hash,
            driver=driver_label,
        ) as session:
            make_driver = driver_factory or _driver_factory("native", history_mode=history_mode)
            driver = make_driver(session)

            skipped_min_context = [
                task.id for task in tasks if task.min_context > context_length
            ]

            for task in tasks:
                if task.min_context > context_length:
                    continue
                for repetition in range(1, repetitions + 1):
                    key = (config_id, suite, task.id, repetition)
                    if key in session.existing:
                        continue
                    record = _record_for(
                        task,
                        repetition,
                        config_id=config_id,
                        suite=suite,
                        session_id=session.session_id,
                        environment_sha256=session.env.sha256,
                        context_length=context_length,
                        driver=driver,
                        driver_label=driver_label,
                        pid=session.pid,
                        transcripts_dir=session.transcripts_dir,
                        executor=session.executor,
                    )
                    results.append_record(session.raw_path, record)
    except UnsupportedContextError as exc:
        return StageOutcome(status="unsupported", detail=str(exc))
    except lmstudio.ModelOversizedError as exc:
        return StageOutcome(status="oversized", detail=str(exc))

    return StageOutcome(
        status="completed",
        session_dir=session.session_dir,
        raw_path=session.raw_path,
        records=results.read_records(session.raw_path),
        skipped_min_context=skipped_min_context,
    )


def _evaluate_stage0(stage: StageOutcome) -> Stage0Outcome:
    """The Stage 0 gate's pure arithmetic — separated so `harness/report.py`
    can recompute it from raw JSONL (`stage0_gate`) without a live run,
    exactly as `_evaluate_stage2a` does for Stage 2A."""
    if stage.status != "completed":
        return Stage0Outcome(tool_capable=False, valid_runs=0, total_runs=0, stage=stage)

    total = len(stage.records)
    valid = sum(1 for r in stage.records if r["tool_calls"] - r["invalid_calls"] > 0)
    tool_capable = total > 0 and (valid / total) >= STAGE0_GATE_RATE
    return Stage0Outcome(
        tool_capable=tool_capable, valid_runs=valid, total_runs=total, stage=stage
    )


def stage0_gate(records: list[dict]) -> Stage0Outcome:
    """The Stage 0 gate over records already on disk (`harness/report.py`)."""
    return _evaluate_stage0(StageOutcome(status="completed", records=records))


def run_stage0(
    config_id: str,
    context_length: int = 8192,
    *,
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
) -> Stage0Outcome:
    stage = run_stage(
        config_id,
        STAGE0_TASKS,
        stage_name="stage0",
        suite="0",
        context_length=context_length,
        repetitions=STAGE0_REPETITIONS,
        results_dir=results_dir,
        configs_dir=configs_dir,
    )
    return _evaluate_stage0(stage)


def _task_passed_by_majority(records: list[dict]) -> bool:
    passes = sum(1 for r in records if r["passed"])
    return passes * 2 > len(records)


def _evaluate_stage2a(stage: StageOutcome) -> Stage2AOutcome:
    """The gate's pure arithmetic, over whatever records `run_stage` wrote —
    separated from running the stage so it can be exercised directly against
    synthetic records, without driving all ten Suite W tasks through a model
    or a fake client."""
    if stage.status != "completed":
        return Stage2AOutcome(
            proceeds=False, tasks_passed=0, tasks_total=0, mean_progress=0.0, stage=stage
        )

    by_task: dict[str, list[dict]] = {}
    for record in stage.records:
        by_task.setdefault(record["task_id"], []).append(record)

    tasks_total = len(by_task)
    tasks_passed = sum(1 for recs in by_task.values() if _task_passed_by_majority(recs))
    mean_progress = (
        sum(r["progress_score"] for r in stage.records) / len(stage.records)
        if stage.records
        else 0.0
    )

    proceeds = tasks_passed >= STAGE2A_MIN_PASSES or mean_progress >= STAGE2A_MIN_MEAN_PROGRESS
    return Stage2AOutcome(
        proceeds=proceeds,
        tasks_passed=tasks_passed,
        tasks_total=tasks_total,
        mean_progress=mean_progress,
        stage=stage,
    )


def stage2a_gate(records: list[dict]) -> Stage2AOutcome:
    """The Stage 2A gate over records already on disk (`harness/report.py`)."""
    return _evaluate_stage2a(StageOutcome(status="completed", records=records))


def run_stage2a(
    config_id: str,
    context_length: int = 8192,
    *,
    driver: str = "native",
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
) -> Stage2AOutcome:
    stage = run_stage(
        config_id,
        SUITE_W,
        stage_name="stage2a" if driver == "native" else f"stage2a-{driver}",
        suite="W",
        context_length=context_length,
        repetitions=AGENT_REPETITIONS,
        results_dir=results_dir,
        configs_dir=configs_dir,
        driver_label=driver,
        driver_factory=_driver_factory(driver),
    )
    return _evaluate_stage2a(stage)


def run_stage2b(
    config_id: str,
    context_length: int = 8192,
    *,
    driver: str = "native",
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
) -> StageOutcome:
    """Survivors of the Stage 2A gate only (§9 Stage 2B) — this function
    doesn't check that itself; `run_stages()` (§5) only calls it once Stage
    2A has proceeded under the same driver."""
    return run_stage(
        config_id,
        SUITE_T,
        stage_name="stage2b" if driver == "native" else f"stage2b-{driver}",
        suite="T",
        context_length=context_length,
        repetitions=AGENT_REPETITIONS,
        results_dir=results_dir,
        configs_dir=configs_dir,
        driver_label=driver,
        driver_factory=_driver_factory(driver),
    )


def run_stage3(
    config_id: str,
    *,
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
) -> dict[str, StageOutcome]:
    """Both suites at 16K (§9 Stage 3), for configurations already above the
    floor at 8K. Not part of `run_stages()` (see the module docstring) —
    called independently once Stage 2A has proceeded."""
    outcomes = {}
    for suite_name, tasks in (("W", SUITE_W), ("T", SUITE_T)):
        outcomes[suite_name] = run_stage(
            config_id,
            tasks,
            stage_name="stage3",
            suite=suite_name,
            context_length=16384,
            repetitions=AGENT_REPETITIONS,
            results_dir=results_dir,
            configs_dir=configs_dir,
        )
    return outcomes


def _stage1_record(
    tier: str,
    repetition: int,
    *,
    config_id: str,
    session_id: str,
    environment_sha256: str,
    context_length: int,
    client: LMStudioClient,
    overhead: float,
    pid: int | None,
    primary_prompt: str,
    alternate_prompt: str,
) -> dict:
    # Stage 1 bypasses the drivers entirely (raw inference, no agent loop), but
    # carries the driver in `run_id` for the same uniqueness contract as the
    # agent stages.
    run_id = f"{config_id}-native-1-{tier}-r{repetition}"

    def _attempt(prompt_text: str):
        messages = [{"role": "user", "content": nonce_prefix() + "\n\n" + prompt_text}]
        sampler = MemorySampler(pid).start() if pid is not None else None
        started = time.monotonic()
        with SwapWindow() as swap:
            turn = client.stream_turn(messages, clock=time.monotonic)
        elapsed = time.monotonic() - started
        peak = sampler.stop() if sampler is not None else None
        return turn, peak, swap, elapsed

    turn, peak, swap, elapsed = _attempt(primary_prompt)
    if (turn.completion_tokens or 0) < STAGE1_MIN_COMPLETION_TOKENS:
        turn, peak, swap, elapsed = _attempt(alternate_prompt)

    tm = turn_metrics(turn, overhead)
    return {
        "run_id": run_id,
        "session_id": session_id,
        "config_id": config_id,
        "driver": "native",
        "suite": "1",
        "task_id": tier,
        "repetition": repetition,
        "environment_sha256": environment_sha256,
        "context_length": context_length,
        "task_set_version": TASK_SET_VERSION,
        "ttft_s": tm.ttft_s,
        "gen_tps": tm.gen_tps,
        "prompt_tps": tm.prompt_tps,
        "ttft_turn1_s": tm.ttft_s,
        "ttft_median_later_s": None,
        "prompt_tokens": tm.prompt_tokens,
        "completion_tokens": tm.completion_tokens,
        "total_tokens": (tm.prompt_tokens or 0) + (tm.completion_tokens or 0),
        "peak_memory_bytes": peak,
        "swap_delta_bytes": swap.delta_bytes,
        "swap_flag": swap.flagged,
        "steps": 1,
        "tool_calls": 0,
        "invalid_calls": 0,
        "path_errors": 0,
        # Not one of native's §4.8 termination reasons — a raw completion has
        # no agent loop, so the model's own finish_reason is what's meaningful.
        "termination_reason": turn.finish_reason or "unknown",
        "passed": None,
        "progress_score": None,
        "condition_failures": None,
        "flaky": None,
        "wall_clock_s": round(elapsed, 3),
        "transcript_path": None,
    }


def run_stage1(
    config_id: str,
    tier: str,
    *,
    repetitions: int = STAGE1_REPETITIONS,
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
    verify_hash: bool = True,
) -> StageOutcome:
    context_length = STAGE1_CONTEXT[tier]
    primary = (PROMPTS_DIR / f"{tier}_primary.txt").read_text(encoding="utf-8")
    alternate = (PROMPTS_DIR / f"{tier}_alternate.txt").read_text(encoding="utf-8")

    try:
        with _model_session(
            config_id, "stage1", context_length,
            results_dir=results_dir, configs_dir=configs_dir, verify_hash=verify_hash,
        ) as session:
            for repetition in range(1, repetitions + 1):
                key = (config_id, "1", tier, repetition)
                if key in session.existing:
                    continue
                record = _stage1_record(
                    tier,
                    repetition,
                    config_id=config_id,
                    session_id=session.session_id,
                    environment_sha256=session.env.sha256,
                    context_length=context_length,
                    client=session.client,
                    overhead=session.overhead,
                    pid=session.pid,
                    primary_prompt=primary,
                    alternate_prompt=alternate,
                )
                results.append_record(session.raw_path, record)
    except UnsupportedContextError as exc:
        return StageOutcome(status="unsupported", detail=str(exc))
    except lmstudio.ModelOversizedError as exc:
        return StageOutcome(status="oversized", detail=str(exc))

    return StageOutcome(
        status="completed",
        session_dir=session.session_dir,
        raw_path=session.raw_path,
        records=results.read_records(session.raw_path),
    )


def run_stage5b_compact(
    config_id: str,
    context_length: int = 8192,
    *,
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
) -> dict[str, StageOutcome]:
    """Stage 5B's context-compaction experiment (§9 Stage 5B): the same
    Suite W/T tasks, run through `NativeDriver(history_mode="compact")`.
    Written with `driver="native-compact"` to its own raw files
    (`stage5b-compact-w.jsonl` / `-t.jsonl`) so it is never pooled with the
    controlled comparison, even though it shares the Stage 2A/2B session."""
    outcomes = {}
    for suite_name, tasks in (("W", SUITE_W), ("T", SUITE_T)):
        outcomes[suite_name] = run_stage(
            config_id,
            tasks,
            stage_name=f"stage5b-compact-{suite_name.lower()}",
            suite=suite_name,
            context_length=context_length,
            repetitions=STAGE5B_REPETITIONS,
            results_dir=results_dir,
            configs_dir=configs_dir,
            history_mode="compact",
            driver_label="native-compact",
        )
    return outcomes


def _step_stage0(
    config_id: str, driver: str, results_dir: Path, configs_dir: Path | None
) -> Stage0Outcome:
    return run_stage0(config_id, results_dir=results_dir, configs_dir=configs_dir)


def _step_stage1(
    config_id: str, driver: str, results_dir: Path, configs_dir: Path | None
) -> dict[str, StageOutcome]:
    return {
        tier: run_stage1(config_id, tier, results_dir=results_dir, configs_dir=configs_dir)
        for tier in STAGE1_CONTEXT
    }


def _step_stage2a(
    config_id: str, driver: str, results_dir: Path, configs_dir: Path | None
) -> Stage2AOutcome:
    return run_stage2a(
        config_id, driver=driver, results_dir=results_dir, configs_dir=configs_dir
    )


def _step_stage2b(
    config_id: str, driver: str, results_dir: Path, configs_dir: Path | None
) -> StageOutcome:
    return run_stage2b(
        config_id, driver=driver, results_dir=results_dir, configs_dir=configs_dir
    )


@dataclass
class StageStep:
    name: str
    run: Callable[[str, str, Path, Path | None], Any]
    # outcome -> whether the sequence should attempt the next stage.
    proceeds: Callable[[Any], bool]


# Stage 0/1 ignore `driver` (always native, per the module docstring); Stage
# 2A/2B honour it. Adding a later stage (Stage 3, say) is a new entry here,
# not a new function alongside `run_stages()`.
STEPS: dict[str, StageStep] = {
    "stage0": StageStep(
        "stage0", _step_stage0,
        lambda o: o.stage.status == "completed" and o.tool_capable,
    ),
    "stage1": StageStep(
        "stage1", _step_stage1,
        lambda o: True,  # a separate measurement, not a gate -- never blocks
    ),
    "stage2a": StageStep(
        "stage2a", _step_stage2a,
        lambda o: o.stage.status == "completed" and o.proceeds,
    ),
    "stage2b": StageStep(
        "stage2b", _step_stage2b,
        lambda o: o.status == "completed",
    ),
}


def run_stages(
    config_id: str,
    stage_names: list[str],
    *,
    driver: str = "native",
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path | None = None,
) -> StagesRunOutcome:
    """Run `stage_names` in order for `config_id` under `driver`, stopping at
    the first stage that doesn't complete or fails its gate (§9.2: "an
    explicit go/no-go decision point, not a formality") — no further stages
    attempted for this `(config_id, driver)` pair. See the module docstring
    for the `STEPS` registry this walks and why Stage 0/1 stay native
    regardless of `driver`.
    """
    results_by_stage: dict[str, Any] = {}
    stopped_at = None
    for name in stage_names:
        step = STEPS[name]
        result = step.run(config_id, driver, results_dir, configs_dir)
        results_by_stage[name] = result
        if not step.proceeds(result):
            stopped_at = name
            break
    return StagesRunOutcome(
        config_id=config_id,
        driver=driver,
        stage_names=stage_names,
        results=results_by_stage,
        stopped_at=stopped_at,
    )


def _describe_step_result(name: str, result: Any) -> str:
    """One-line summary of a `run_stages()` step result, for the `run` CLI
    subcommand. `result`'s shape depends on `name` (see `STEPS`)."""
    if name == "stage0":
        return f"status={result.stage.status} tool_capable={result.tool_capable}"
    if name == "stage1":
        return ", ".join(f"{tier}={stage.status}" for tier, stage in result.items())
    if name == "stage2a":
        return (
            f"status={result.stage.status} proceeds={result.proceeds} "
            f"({result.tasks_passed}/{result.tasks_total})"
        )
    if name == "stage2b":
        return f"status={result.status}"
    return str(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.stages")
    sub = parser.add_subparsers(dest="command", required=True)

    stage0 = sub.add_parser("stage0", help="run the §9 Stage 0 tool-calling gate")
    stage0.add_argument("config_id", help="§2 configuration ID, e.g. LFM-M8")
    stage0.add_argument("--context", type=int, default=8192)

    stage1 = sub.add_parser("stage1", help="run §9 Stage 1: raw inference at 8K and 16K")
    stage1.add_argument("config_id", help="§2 configuration ID, e.g. LFM-M8")

    stage2a = sub.add_parser("stage2a", help="run §9 Stage 2A: Suite W at 8K")
    stage2a.add_argument("config_id", help="§2 configuration ID, e.g. LFM-M8")
    stage2a.add_argument("--context", type=int, default=8192)
    stage2a.add_argument("--driver", choices=("native", "pi"), default="pi")

    stage2b = sub.add_parser(
        "stage2b", help="run §9 Stage 2B: Suite T at 8K, for Stage 2A survivors"
    )
    stage2b.add_argument("config_id", help="§2 configuration ID, e.g. LFM-M8")
    stage2b.add_argument("--context", type=int, default=8192)
    stage2b.add_argument("--driver", choices=("native", "pi"), default="pi")

    stage5b = sub.add_parser(
        "stage5b-compact", help="run §9 Stage 5B's context-compaction experiment"
    )
    stage5b.add_argument("config_id", help="§2 configuration ID, e.g. LFM-M8")
    stage5b.add_argument("--context", type=int, default=8192)

    run = sub.add_parser("run", help="run an ordered list of stages via run_stages()")
    run.add_argument("config_id", help="§2 configuration ID, e.g. LFM-M8")
    run.add_argument(
        "--stages", required=True,
        help="comma-separated stage names, e.g. stage0,stage1,stage2a,stage2b",
    )
    run.add_argument("--driver", choices=("native", "pi"), default="pi")

    args = parser.parse_args(argv)

    if args.command == "stage0":
        outcome = run_stage0(args.config_id, context_length=args.context)
        print(f"status: {outcome.stage.status}")
        if outcome.stage.status == "completed":
            print(f"tool-capable: {outcome.tool_capable} ({outcome.valid_runs}/{outcome.total_runs} valid)")
            print(f"raw records: {outcome.stage.raw_path}")
            return 0
        print(outcome.stage.detail)
        return 1

    if args.command == "stage1":
        exit_code = 0
        for tier in STAGE1_CONTEXT:
            outcome = run_stage1(args.config_id, tier)
            print(f"{tier}: status={outcome.status}")
            if outcome.status == "completed":
                print(f"  raw records: {outcome.raw_path}")
            else:
                print(f"  {outcome.detail}")
                exit_code = 1
        return exit_code

    if args.command == "stage2a":
        outcome = run_stage2a(args.config_id, context_length=args.context, driver=args.driver)
        print(f"status: {outcome.stage.status}")
        if outcome.stage.status == "completed":
            print(
                f"proceeds to 2B: {outcome.proceeds} "
                f"({outcome.tasks_passed}/{outcome.tasks_total} tasks passed, "
                f"mean progress {outcome.mean_progress:.2f})"
            )
            if outcome.stage.skipped_min_context:
                print(f"skipped (min_context): {', '.join(outcome.stage.skipped_min_context)}")
            print(f"raw records: {outcome.stage.raw_path}")
            return 0
        print(outcome.stage.detail)
        return 1

    if args.command == "stage2b":
        outcome = run_stage2b(args.config_id, context_length=args.context, driver=args.driver)
        print(f"status: {outcome.status}")
        if outcome.status == "completed":
            if outcome.skipped_min_context:
                print(f"skipped (min_context): {', '.join(outcome.skipped_min_context)}")
            print(f"raw records: {outcome.raw_path}")
            return 0
        print(outcome.detail)
        return 1

    if args.command == "stage5b-compact":
        outcomes = run_stage5b_compact(args.config_id, context_length=args.context)
        exit_code = 0
        for suite_name, outcome in outcomes.items():
            print(f"suite {suite_name}: status={outcome.status}")
            if outcome.status == "completed":
                print(f"  raw records: {outcome.raw_path}")
            else:
                print(f"  {outcome.detail}")
                exit_code = 1
        return exit_code

    if args.command == "run":
        stage_names = [name.strip() for name in args.stages.split(",") if name.strip()]
        outcome = run_stages(args.config_id, stage_names, driver=args.driver)
        print(f"config: {outcome.config_id}  driver: {outcome.driver}")
        for name in outcome.stage_names:
            result = outcome.results.get(name)
            if result is None:
                print(f"{name}: not attempted")
                continue
            print(f"{name}: {_describe_step_result(name, result)}")
        if outcome.stopped_at is not None:
            print(f"stopped at: {outcome.stopped_at}")
            return 1
        print("completed every requested stage")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
