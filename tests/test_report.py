"""Reporting (doc/benchmark.md §10.1-§10.3): flaky grouping, the headline
metric, the Stage 5B degenerate-rate detector, and the final table's verdict.

Pure computation over synthetic records — nothing here touches a real model
or `results/` on disk (the loaders that scan `results/` are exercised
manually per the plan's live-verification step, not here)."""

from __future__ import annotations

import pytest

from harness import report


# --- §9.1: flaky --------------------------------------------------------


def test_flaky_true_when_passed_is_not_unanimous():
    records = [
        {"config_id": "C", "suite": "W", "task_id": "W01", "passed": True},
        {"config_id": "C", "suite": "W", "task_id": "W01", "passed": False},
        {"config_id": "C", "suite": "W", "task_id": "W01", "passed": True},
    ]
    annotated = report.annotate_flaky(records)
    assert all(r["flaky"] is True for r in annotated)


def test_flaky_false_when_unanimous():
    records = [{"config_id": "C", "suite": "W", "task_id": "W01", "passed": True}] * 3
    annotated = report.annotate_flaky(records)
    assert all(r["flaky"] is False for r in annotated)


def test_different_tasks_are_grouped_separately():
    records = [
        {"config_id": "C", "suite": "W", "task_id": "W01", "passed": True},
        {"config_id": "C", "suite": "W", "task_id": "W01", "passed": True},
        {"config_id": "C", "suite": "W", "task_id": "W02", "passed": True},
        {"config_id": "C", "suite": "W", "task_id": "W02", "passed": False},
    ]
    annotated = report.annotate_flaky(records)
    w01 = [r for r in annotated if r["task_id"] == "W01"]
    w02 = [r for r in annotated if r["task_id"] == "W02"]
    assert all(r["flaky"] is False for r in w01)
    assert all(r["flaky"] is True for r in w02)


def test_passed_none_records_are_left_alone():
    record = {"config_id": "C", "suite": "1", "task_id": "8k", "passed": None, "flaky": None}
    annotated = report.annotate_flaky([record])
    assert annotated[0]["flaky"] is None


# --- §10.2: headline metric -----------------------------------------------


def test_suite_summary_counts_passes_and_computes_rate():
    records = [
        {"config_id": "C", "suite": "W", "driver": "native", "passed": True, "wall_clock_s": 60.0},
        {"config_id": "C", "suite": "W", "driver": "native", "passed": False, "wall_clock_s": 60.0},
        {"config_id": "C", "suite": "T", "driver": "native", "passed": True, "wall_clock_s": 999.0},
    ]
    summary = report.suite_summary(records, "C", "W")
    assert summary.total_runs == 2
    assert summary.passed_runs == 1
    assert summary.total_wall_clock_s == 120.0
    assert summary.successful_tasks_per_hour == pytest.approx(1 / (120.0 / 3600))


def test_suite_summary_a_slow_failure_scores_no_better_than_a_fast_one():
    fast_fail = report.suite_summary(
        [{"config_id": "C", "suite": "W", "driver": "native", "passed": False, "wall_clock_s": 10.0}],
        "C", "W",
    )
    slow_fail = report.suite_summary(
        [{"config_id": "C", "suite": "W", "driver": "native", "passed": False, "wall_clock_s": 500.0}],
        "C", "W",
    )
    assert fast_fail.successful_tasks_per_hour == 0.0
    assert slow_fail.successful_tasks_per_hour == 0.0


def test_suite_summary_with_no_runs_is_zero_not_an_error():
    summary = report.suite_summary([], "C", "W")
    assert summary.total_runs == 0
    assert summary.successful_tasks_per_hour == 0.0


def test_suite_summary_never_pools_a_different_driver():
    """§4.1: records from different drivers are never pooled."""
    records = [
        {"config_id": "C", "suite": "W", "driver": "native", "passed": True, "wall_clock_s": 60.0},
        {"config_id": "C", "suite": "W", "driver": "pi", "passed": True, "wall_clock_s": 60.0},
    ]
    native = report.suite_summary(records, "C", "W", driver="native")
    pi = report.suite_summary(records, "C", "W", driver="pi")
    assert native.total_runs == 1
    assert pi.total_runs == 1


# --- §4.2: degenerate-rate detector -----------------------------------------


def _agent_records(degenerate: int, normal: int) -> list[dict]:
    records = [{"suite": "W", "termination_reason": "loop_detected"} for _ in range(degenerate)]
    records += [{"suite": "W", "termination_reason": "final_answer"} for _ in range(normal)]
    return records


def test_degenerate_rate_at_exactly_the_threshold_does_not_trigger():
    records = _agent_records(degenerate=2, normal=8)  # exactly 20%
    assert report.degenerate_rate(records) == pytest.approx(0.20)
    assert report.is_degenerate_triggered(records) is False


def test_degenerate_rate_above_the_threshold_triggers():
    records = _agent_records(degenerate=3, normal=7)  # 30%
    assert report.is_degenerate_triggered(records) is True


def test_stage1_records_are_excluded_from_degenerate_rate():
    records = [{"suite": "1", "termination_reason": "length"} for _ in range(10)]
    assert report.degenerate_rate(records) == 0.0


def test_degenerate_rate_with_no_records_is_zero():
    assert report.degenerate_rate([]) == 0.0


# --- §4.8 server_error: tracked separately from degenerate_rate ------------


def test_server_error_rate_is_not_counted_as_degenerate():
    """A backend crash is an infrastructure fault, not a decoding
    degeneracy — recommended-default sampling could plausibly fix a
    repetition loop, but not a server crash, so the two rates must not be
    conflated."""
    records = [{"suite": "W", "termination_reason": "server_error"}] * 5
    assert report.degenerate_rate(records) == 0.0
    assert report.server_error_rate(records) == 1.0


def test_server_error_rate_excludes_stage1():
    records = [{"suite": "1", "termination_reason": "server_error"}] * 5
    assert report.server_error_rate(records) == 0.0


# --- §10.3: final table verdict ---------------------------------------------


def _stage0_records(config_id: str, valid_count: int, total: int = 9) -> list[dict]:
    return [
        {
            "config_id": config_id, "suite": "0", "task_id": f"S{i % 3 + 1:02d}",
            "context_length": 8192,
            "tool_calls": 1 if i < valid_count else 0, "invalid_calls": 0,
        }
        for i in range(total)
    ]


def _stage2a_records(config_id: str, passed: bool, progress: int, driver: str = "native") -> list[dict]:
    return [
        {
            "config_id": config_id, "suite": "W", "task_id": f"W{i:02d}", "driver": driver,
            "context_length": 8192, "passed": passed, "progress_score": progress,
        }
        for i in range(1, 10)
    ]


def test_verdict_not_run_with_no_data():
    assert report._verdict("C", []) == "not run"


def test_verdict_excluded_not_tool_capable():
    records = _stage0_records("C", valid_count=3)  # 3/9 < 2/3
    assert "not tool-capable" in report._verdict("C", records)


def test_verdict_passed_stage0_only_with_no_stage2a_data():
    records = _stage0_records("C", valid_count=9)
    assert report._verdict("C", records) == "passed Stage 0 only"


def test_verdict_excluded_failed_2a_gate():
    records = _stage0_records("C", valid_count=9)
    records += _stage2a_records("C", passed=False, progress=1)
    assert report._verdict("C", records) == "excluded: failed Stage 2A gate"


def test_verdict_passed_stage2a_gate_with_no_stage2b_data():
    records = _stage0_records("C", valid_count=9)
    records += _stage2a_records("C", passed=True, progress=4)
    assert report._verdict("C", records) == "passed Stage 2A gate"


def test_verdict_proceeded_to_stage2b():
    records = _stage0_records("C", valid_count=9)
    records += _stage2a_records("C", passed=True, progress=4)
    records.append({"config_id": "C", "context_length": 8192, "suite": "T", "driver": "native"})
    assert report._verdict("C", records) == "proceeded to Stage 2B"


def test_verdict_proceeded_to_stage3():
    records = _stage0_records("C", valid_count=9)
    records += _stage2a_records("C", passed=True, progress=4)
    records.append({"config_id": "C", "context_length": 8192, "suite": "T", "driver": "native"})
    records.append({"config_id": "C", "context_length": 16384, "suite": "W", "driver": "native"})
    assert report._verdict("C", records) == "proceeded to Stage 3"


def test_verdict_stage1_16k_records_are_not_mistaken_for_stage3():
    """Regression: Stage 1's 16K raw-inference records also carry
    context_length == 16384, but under suite "1" — they are not Stage 3
    evidence. A live report once showed "proceeded to Stage 3" for every
    config that merely had Stage 1 16K data, with no Stage 2B/3 ever run."""
    records = _stage0_records("C", valid_count=9)
    records += _stage2a_records("C", passed=True, progress=4)
    records.append({"config_id": "C", "context_length": 8192, "suite": "T", "driver": "native"})
    records.append({"config_id": "C", "context_length": 16384, "suite": "1", "driver": "native"})
    assert report._verdict("C", records) == "proceeded to Stage 2B"


def test_verdict_scoped_to_driver_pi_data_does_not_leak_into_native_verdict():
    """§4.1: pi's Stage 2A/2B records must not make a native verdict look
    more advanced than it is, and vice versa."""
    records = _stage0_records("C", valid_count=9)
    records += _stage2a_records("C", passed=True, progress=4, driver="pi")
    assert report._verdict("C", records, driver="native") == "passed Stage 0 only"
    assert report._verdict("C", records, driver="pi") == "passed Stage 2A gate"
