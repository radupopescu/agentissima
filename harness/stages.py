"""Stage runner (doc/benchmark.md §9).

Loads a model once, runs a set of tasks across repetitions, and writes §10.1
JSONL records — the load/unload discipline of §9.0, generalised so Stage
2A/2B/3/4 are just a different task list and repetition count later, not new
code.

Session identity is deterministic (`<config_id>-<context_length>`, no
timestamp), so re-running the same stage command lands in the same session
directory and `results.existing_keys()` on that stage's own raw file does the
actual resume work — no separate flag. The one risk that creates — resuming
into a session captured under a different harness revision or
`task_set_version` and silently pooling incomparable runs — is guarded by
`_check_environment_matches` before anything new is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import environment, lmstudio, results
from .admissibility import UNSUPPORTED, classify_declared
from .client import LMStudioClient
from .driver_native import NativeDriver
from .metrics import MemorySampler, SwapWindow, find_inference_pid
from .runner import run_task
from .tasks.smoke import STAGE0_TASKS
from .types import Task
from .version import TASK_SET_VERSION

RESULTS_DIR = Path("results")
STAGE_IDENTIFIER = "bench"

# §9 Stage 0: "fewer than 2 of 3 valid tool calls" for 3 tasks × 3 repetitions
# (9 runs) is read as an aggregate rate, not a per-task count — see
# doc/benchmark.md §9 Stage 0 for the reasoning.
STAGE0_REPETITIONS = 3
STAGE0_GATE_RATE = 2 / 3


class SessionMismatchError(RuntimeError):
    """A resumed session's raw records were written under a different
    `task_set_version` than the one running now. Refuses to pool them
    silently."""


@dataclass
class StageOutcome:
    status: str  # "completed" | "unsupported" | "oversized"
    session_dir: Path | None = None
    raw_path: Path | None = None
    records: list[dict] = field(default_factory=list)
    detail: str | None = None


@dataclass
class Stage0Outcome:
    tool_capable: bool
    valid_runs: int
    total_runs: int
    stage: StageOutcome


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
    pid: int | None,
    transcripts_dir: Path,
) -> dict:
    run_id = f"{config_id}-{suite}-{task.id}-r{repetition}"

    sampler = MemorySampler(pid).start() if pid is not None else None
    with SwapWindow() as swap:
        graded = run_task(task, driver)
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
        "driver": "native",
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
        # Deciding this needs every repetition of a task, which may not all
        # exist yet under resume; left to `harness/report.py` (§9.1).
        "flaky": None,
        "wall_clock_s": graded.wall_clock_s,
        "transcript_path": transcript_path,
    }


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
) -> StageOutcome:
    """Run `tasks` × `repetitions` against `config_id` at `context_length`,
    loading the model once (§9.0) and writing §10.1 records as they complete.
    """
    resolved = environment.load_resolved(config_id, configs_dir)
    if classify_declared(resolved["advertised_max_context"], context_length) == UNSUPPORTED:
        return StageOutcome(
            status="unsupported",
            detail=(
                f"{config_id} advertises a maximum context of "
                f"{resolved['advertised_max_context']}, below the requested "
                f"{context_length}"
            ),
        )

    session_id = f"{config_id}-{context_length}"
    session_dir = Path(results_dir) / session_id
    raw_path = session_dir / "raw" / f"{stage_name}.jsonl"
    transcripts_dir = session_dir / "transcripts"

    try:
        with lmstudio.loaded(
            resolved["model_path"],
            context_length=context_length,
            identifier=STAGE_IDENTIFIER,
        ):
            env = environment.capture(
                config_id,
                context_length,
                driver="native",
                out_dir=results_dir,
                configs_dir=configs_dir,
                verify_hash=verify_hash,
                session_id=session_id,
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
            driver = NativeDriver(client=client, overhead_s=overhead)

            for task in tasks:
                for repetition in range(1, repetitions + 1):
                    key = (config_id, suite, task.id, repetition)
                    if key in existing:
                        continue
                    record = _record_for(
                        task,
                        repetition,
                        config_id=config_id,
                        suite=suite,
                        session_id=session_id,
                        environment_sha256=env.sha256,
                        context_length=context_length,
                        driver=driver,
                        pid=pid,
                        transcripts_dir=transcripts_dir,
                    )
                    results.append_record(raw_path, record)
    except lmstudio.ModelOversizedError as exc:
        return StageOutcome(status="oversized", detail=str(exc))

    return StageOutcome(
        status="completed",
        session_dir=session_dir,
        raw_path=raw_path,
        records=results.read_records(raw_path),
    )


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
    if stage.status != "completed":
        return Stage0Outcome(tool_capable=False, valid_runs=0, total_runs=0, stage=stage)

    total = len(stage.records)
    valid = sum(1 for r in stage.records if r["tool_calls"] - r["invalid_calls"] > 0)
    tool_capable = total > 0 and (valid / total) >= STAGE0_GATE_RATE
    return Stage0Outcome(
        tool_capable=tool_capable, valid_runs=valid, total_runs=total, stage=stage
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.stages")
    sub = parser.add_subparsers(dest="command", required=True)

    stage0 = sub.add_parser("stage0", help="run the §9 Stage 0 tool-calling gate")
    stage0.add_argument("config_id", help="§2 configuration ID, e.g. LFM-M8")
    stage0.add_argument("--context", type=int, default=8192)

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

    return 2


if __name__ == "__main__":
    sys.exit(main())
