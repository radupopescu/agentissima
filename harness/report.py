"""Reporting (doc/benchmark.md §10.1-§10.3).

Everything here reads only `results/<session>/raw/*.jsonl` and recomputes
from it — `way-of-working.md`'s "reports regenerate from JSONL only,"
including `flaky` (§9.1), which `harness/stages.py` deliberately leaves
`null` at write time because deciding it needs every repetition of a task.

**Every file under a session's `raw/` is scanned and pooled, unconditionally
— there is no name filter.** A run that must never be reported (e.g. one
made under a since-fixed environment bug) does not belong under `raw/` at
all; move it to a sibling directory such as `archive/` instead of renaming
it within `raw/`. This was found the hard way: a LFM-GQ4 Stage 2A run made
before an LM Studio bug was fixed was renamed to `stage2a-<description
>.jsonl` still inside `raw/`, and silently got pooled back into the very
report it was meant to be excluded from, doubling the run count.

Not built here: §10.4's "reporting the three questions" is conclusions
written by a person once real multi-configuration data exists — this module
produces the tables that conclusion is drawn from, not the conclusion.

§4.1: "records from different drivers are never pooled, averaged, or
compared cell-by-cell as though equivalent." Every summary below that reads
agent-stage records (Suite W/T) is scoped to one `driver` at a time —
`suite_summary`, `_verdict`, `degenerate_rate`, `server_error_rate`. This
also keeps Stage 5B's `native-compact` experiment (§9 Stage 5B) out of the
main comparison automatically: it is a distinct `driver` value that
`_drivers_present` never selects, so it stays excluded from every one of
these tables without needing its own filter. Stage 0/1 are the one
exception — they always run `native` regardless of which driver a verdict
describes (§4.1), so `_verdict` reads them unscoped by driver.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from . import stages
from .results import read_records

RESULTS_DIR = Path("results")

# §4.2: Stage 5B's recommended-default sampling pass is triggered when
# degenerate behaviour exceeds this rate. None of the four §4.8 outcomes
# below are themselves "finish_reason anomalies" in the raw API sense —
# that value isn't part of the §10.1 schema — so this is a proxy from what
# *is* stored: a run that ended in one of these ways got stuck or produced
# nothing coherent, which is the behaviour §4.2 is describing.
DEGENERATE_TERMINATIONS = frozenset(
    {"loop_detected", "empty_answer", "malformed_calls", "timeout"}
)
DEGENERATE_THRESHOLD = 0.20


# --- loading -----------------------------------------------------------------


def load_all_records(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Every record from every session's `raw/*.jsonl`."""
    records: list[dict] = []
    for raw_path in sorted(Path(results_dir).glob("*/raw/*.jsonl")):
        records.extend(read_records(raw_path))
    return records


def load_stage_records(stage_name: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Records from one stage's file across every session, e.g. `"stage2a"`."""
    records: list[dict] = []
    for raw_path in sorted(Path(results_dir).glob(f"*/raw/{stage_name}.jsonl")):
        records.extend(read_records(raw_path))
    return records


# Drivers reported in the controlled comparison (§4.1). Stage 5B's
# "native-compact" is deliberately not one of these -- see the module
# docstring.
COMPARISON_DRIVERS = ("native", "pi")


def _drivers_present(all_records: list[dict], config_id: str) -> list[str]:
    """Which comparison drivers actually have any record for `config_id`,
    in `COMPARISON_DRIVERS` order -- so a driver never run against this
    configuration (typically `pi`, before it has been) gets no row at all,
    rather than a row of all-dashes."""
    seen = {r["driver"] for r in all_records if r["config_id"] == config_id}
    return [driver for driver in COMPARISON_DRIVERS if driver in seen]


# --- §9.1: flaky ---------------------------------------------------------


def annotate_flaky(records: list[dict]) -> list[dict]:
    """Copies of `records` with `flaky` resolved: not unanimous `passed`
    across every repetition sharing `(config_id, suite, task_id)`. Records
    with `passed=None` (Stage 1's raw inference has no assertion) are
    returned unchanged — flakiness isn't defined for them."""
    groups: dict[tuple, list[bool]] = {}
    for record in records:
        if record["passed"] is None:
            continue
        key = (record["config_id"], record["suite"], record["task_id"])
        groups.setdefault(key, []).append(record["passed"])

    flaky_by_key = {key: len(set(values)) > 1 for key, values in groups.items()}

    annotated = []
    for record in records:
        copy = dict(record)
        if record["passed"] is not None:
            key = (record["config_id"], record["suite"], record["task_id"])
            copy["flaky"] = flaky_by_key[key]
        annotated.append(copy)
    return annotated


# --- §4.2: the Stage 5B recommended-sampling trigger --------------------


def degenerate_rate(records: list[dict]) -> float:
    """Fraction of agent-stage records (Stage 1 has no termination_reason in
    the §4.8 sense and is excluded) ending in a degenerate signature."""
    agent_records = [r for r in records if r["suite"] != "1"]
    if not agent_records:
        return 0.0
    degenerate = sum(1 for r in agent_records if r["termination_reason"] in DEGENERATE_TERMINATIONS)
    return degenerate / len(agent_records)


def is_degenerate_triggered(records: list[dict]) -> bool:
    """Whether §4.2's recommended-default sampling pass is warranted for this
    configuration. This only detects the condition — running the pass itself
    is an operator action (`harness/stages.py` has no automatic trigger)."""
    return degenerate_rate(records) > DEGENERATE_THRESHOLD


def server_error_rate(records: list[dict]) -> float:
    """Fraction of agent-stage runs ending in `server_error` (§4.8) — the
    backend failing mid-stream. Tracked separately from `degenerate_rate`:
    a live run hit this on LFM-GQ4's llama.cpp backend on a long tool-call
    argument, and it is an infrastructure fault, not a decoding degeneracy —
    recommended-default sampling (§4.2's trigger) could plausibly fix a
    repetition loop; it could not fix a server crash, so the two must not be
    conflated into one rate."""
    agent_records = [r for r in records if r["suite"] != "1"]
    if not agent_records:
        return 0.0
    errors = sum(1 for r in agent_records if r["termination_reason"] == "server_error")
    return errors / len(agent_records)


# --- §10.2: headline metric -----------------------------------------------


@dataclass
class SuiteSummary:
    config_id: str
    suite: str
    driver: str
    total_runs: int
    passed_runs: int
    total_wall_clock_s: float
    successful_tasks_per_hour: float


def suite_summary(
    records: list[dict], config_id: str, suite: str, *, driver: str = "native"
) -> SuiteSummary:
    """§10.2's headline metric: successful tasks per hour of wall clock. The
    denominator is every run in the suite's stage, not only the passing
    ones — a configuration that fails fast should not be penalised relative
    to one that fails slowly by the same count; both cost real wall clock.
    Scoped to one `driver` (§4.1) — never pooled across drivers."""
    scoped = [
        r for r in records
        if r["config_id"] == config_id and r["suite"] == suite and r["driver"] == driver
    ]
    total_runs = len(scoped)
    passed_runs = sum(1 for r in scoped if r["passed"])
    total_wall_clock_s = sum(r["wall_clock_s"] or 0.0 for r in scoped)
    hours = total_wall_clock_s / 3600
    rate = passed_runs / hours if hours > 0 else 0.0
    return SuiteSummary(config_id, suite, driver, total_runs, passed_runs, total_wall_clock_s, rate)


# --- §10.3: throughput (Stage 1 only, per §5.4) ---------------------------


@dataclass
class ThroughputSummary:
    config_id: str
    ttft_median_s: float | None
    gen_tps_median: float | None
    prompt_tps_median: float | None
    peak_memory_bytes: int | None
    any_swap: bool


def throughput_summary(
    stage1_records: list[dict], config_id: str, context_length: int = 8192
) -> ThroughputSummary:
    """TTFT/gen tok/s/prompt tok/s/peak RAM from Stage 1's nonce-prefixed raw
    inference, never from Suite W/T — §5.4 is explicit that Phase 2 numbers
    "must not be compared across configurations" once the prompt cache is
    warm past turn 1. First repetition discarded, swap-flagged runs excluded
    from the medians but still checked for `any_swap`, matching §9 Stage 1
    and §10.3."""
    scoped = [
        r for r in stage1_records
        if r["config_id"] == config_id and r["context_length"] == context_length
    ]
    after_first = [r for r in scoped if r["repetition"] != 1]
    any_swap = any(r["swap_flag"] for r in after_first)
    clean = [r for r in after_first if not r["swap_flag"]]

    def median_of(field: str) -> float | None:
        values = [r[field] for r in clean if r[field] is not None]
        return statistics.median(values) if values else None

    peak_values = [r["peak_memory_bytes"] for r in after_first if r["peak_memory_bytes"] is not None]

    return ThroughputSummary(
        config_id=config_id,
        ttft_median_s=median_of("ttft_s"),
        gen_tps_median=median_of("gen_tps"),
        prompt_tps_median=median_of("prompt_tps"),
        peak_memory_bytes=max(peak_values) if peak_values else None,
        any_swap=any_swap,
    )


# --- §10.3: the final table ------------------------------------------------


@dataclass
class FinalRow:
    config_id: str
    driver: str
    verdict: str
    suite_w: SuiteSummary
    suite_t: SuiteSummary
    throughput: ThroughputSummary


def _verdict(config_id: str, all_records: list[dict], *, driver: str = "native") -> str:
    """A mechanical stage-progression status, not the qualitative judgement
    `benchmark.md`'s prose implies — that's §10.4's job, written by a person
    once real data exists across configurations.

    Stage 0 is read unscoped by `driver`: it always runs `native` (§4.1), so
    a `pi` verdict still needs it to know whether this configuration is
    tool-capable at all. Stage 2A/2B/3 are scoped to `driver`."""
    stage0_records = [r for r in all_records if r["config_id"] == config_id and r["suite"] == "0"]
    if not stage0_records:
        return "not run"
    if not stages.stage0_gate(stage0_records).tool_capable:
        return "excluded: not tool-capable (Stage 0)"

    stage2a_records = [
        r for r in all_records if r["config_id"] == config_id and r["suite"] == "W"
        and r["context_length"] == 8192 and r["driver"] == driver
    ]
    if not stage2a_records:
        return "passed Stage 0 only"
    if not stages.stage2a_gate(stage2a_records).proceeds:
        return "excluded: failed Stage 2A gate"

    stage2b_records = [
        r for r in all_records if r["config_id"] == config_id and r["suite"] == "T"
        and r["context_length"] == 8192 and r["driver"] == driver
    ]
    if not stage2b_records:
        return "passed Stage 2A gate"

    # suite in ("W", "T") only — Stage 1's 16K raw-inference records also carry
    # context_length == 16384 under suite "1" and are not Stage 3 evidence.
    stage3_records = [
        r for r in all_records if r["config_id"] == config_id
        and r["suite"] in ("W", "T") and r["context_length"] == 16384 and r["driver"] == driver
    ]
    if stage3_records:
        return "proceeded to Stage 3"
    return "proceeded to Stage 2B"


def final_table(config_ids: list[str], results_dir: Path = RESULTS_DIR) -> list[FinalRow]:
    all_records = load_all_records(results_dir)
    stage1_records = [r for r in all_records if r["suite"] == "1"]

    rows = []
    for config_id in config_ids:
        drivers = _drivers_present(all_records, config_id) or ["native"]
        for driver in drivers:
            rows.append(
                FinalRow(
                    config_id=config_id,
                    driver=driver,
                    verdict=_verdict(config_id, all_records, driver=driver),
                    suite_w=suite_summary(all_records, config_id, "W", driver=driver),
                    suite_t=suite_summary(all_records, config_id, "T", driver=driver),
                    # Stage 1 is always native (§4.1) — not driver-scoped.
                    throughput=throughput_summary(stage1_records, config_id),
                )
            )
    return rows


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return "-"
    return format(value, spec) if spec else str(value)


def render_markdown(rows: list[FinalRow]) -> str:
    # Driver is its own column, never folded into Configuration (§4.1) — a
    # reader must be able to tell at a glance which rows are and are not
    # comparable to each other.
    header = (
        "| Configuration | Driver | Suite W | Suite T | TTFT | Gen tok/s | Prompt tok/s "
        "| Peak RAM | Swap | Verdict |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    lines = [header]
    for row in rows:
        w = f"{row.suite_w.passed_runs}/{row.suite_w.total_runs}" if row.suite_w.total_runs else "-"
        t = f"{row.suite_t.passed_runs}/{row.suite_t.total_runs}" if row.suite_t.total_runs else "-"
        tp = row.throughput
        peak = f"{tp.peak_memory_bytes / 1024**3:.2f} GiB" if tp.peak_memory_bytes else "-"
        swap = "yes" if tp.any_swap else "no"
        lines.append(
            f"| {row.config_id} | {row.driver} | {w} | {t} | {_fmt(tp.ttft_median_s, '.3f')} "
            f"| {_fmt(tp.gen_tps_median, '.1f')} | {_fmt(tp.prompt_tps_median, '.1f')} "
            f"| {peak} | {swap} | {row.verdict} |\n"
        )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.report")
    parser.add_argument("--out", type=Path, default=None, help="write markdown to this path too")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    session_dirs = sorted(p for p in args.results_dir.glob("*") if p.is_dir())
    config_ids = sorted({p.name.rsplit("-", 1)[0] for p in session_dirs})

    if not config_ids:
        print(f"no sessions found under {args.results_dir}")
        return 1

    rows = final_table(config_ids, results_dir=args.results_dir)
    table = render_markdown(rows)
    print(table)

    all_records = load_all_records(args.results_dir)
    for config_id in config_ids:
        for driver in _drivers_present(all_records, config_id):
            agent_records = [
                r for r in all_records if r["config_id"] == config_id
                and r["suite"] != "1" and r["driver"] == driver
            ]
            if not agent_records:
                continue
            if is_degenerate_triggered(agent_records):
                rate = degenerate_rate(agent_records)
                print(
                    f"{config_id} ({driver}): degenerate rate {rate:.0%} exceeds the §4.2 "
                    f"threshold ({DEGENERATE_THRESHOLD:.0%}) — Stage 5B's recommended-default "
                    "sampling pass is warranted"
                )
            error_rate = server_error_rate(agent_records)
            if error_rate > 0:
                print(f"{config_id} ({driver}): {error_rate:.0%} of agent runs ended in server_error")

    if args.out:
        args.out.write_text(table, encoding="utf-8")
        print(f"\nwritten to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
