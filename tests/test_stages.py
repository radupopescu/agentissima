"""The stage runner (doc/benchmark.md §9), with no server and no model.

Mirrors how `harness/gates.py` exercises the harness without a model: a fake
client stands in for LM Studio, and `lmstudio.loaded`/`environment.capture`
are stubbed so nothing here touches a real backend.
"""

from __future__ import annotations

import dataclasses
import types
from contextlib import contextmanager

import pytest

from harness import lmstudio, stages
from harness.client import StreamedTurn, ToolCallFragment
from harness.tasks.smoke import STAGE0_TASKS


def _answer_turn(text="done"):
    return StreamedTurn(
        content=text, t_request=0.0, t_first=0.1, t_last=0.2,
        prompt_tokens=10, completion_tokens=5, finish_reason="stop",
    )


def _call_turn():
    return StreamedTurn(
        tool_calls=[ToolCallFragment(index=0, id="c1", name="list_files", arguments='{"path": "."}')],
        t_request=0.0, t_first=0.1, t_last=0.2,
        prompt_tokens=10, completion_tokens=5, finish_reason="tool_calls",
    )


class FakeClient:
    """Stands in for `LMStudioClient`. `turns` is consumed in order across
    every `stream_turn` call for the whole stage, exactly like the real
    client would be across a whole session."""

    def __init__(self, turns):
        self.turns = list(turns)

    def measure_overhead(self, **kwargs):
        return 0.01

    def stream_turn(self, messages, tools=None, clock=None):
        if not self.turns:
            raise AssertionError("stream_turn called with no scripted turns left")
        return self.turns.pop(0)


def _turns_for(pattern: list[bool]) -> list:
    """One run's turns per pattern entry: True = a valid tool call then an
    answer, False = an immediate answer with no tool call at all."""
    turns = []
    for valid in pattern:
        if valid:
            turns += [_call_turn(), _answer_turn()]
        else:
            turns += [_answer_turn("I don't know.")]
    return turns


def _client_factory(*rounds):
    """`stages.LMStudioClient` replacement: each `run_stage` call gets the
    next fake client in `rounds`."""
    remaining = list(rounds)

    def factory(model):
        if not remaining:
            raise AssertionError("LMStudioClient constructed more times than expected")
        return remaining.pop(0)

    return factory


RESOLVED = {
    "model_path": "Fake/Model",
    "advertised_max_context": 131072,
}


@contextmanager
def _fake_loaded(model, context_length=None, identifier=None, gpu=None):
    yield lmstudio.LoadedModel(
        identifier="bench", model_key="fake", path=model, context_length=context_length
    )


def _patch_common(monkeypatch, resolved=RESOLVED, env_sha256="env-hash-1"):
    monkeypatch.setattr(stages.environment, "load_resolved", lambda *a, **k: dict(resolved))
    monkeypatch.setattr(stages.lmstudio, "loaded", _fake_loaded)
    monkeypatch.setattr(
        stages.environment, "capture",
        lambda *a, **k: types.SimpleNamespace(sha256=env_sha256),
    )
    monkeypatch.setattr(stages, "find_inference_pid", lambda: None)


def test_stage0_gate_passes_at_six_of_nine(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    pattern = [True] * 6 + [False] * 3
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(FakeClient(_turns_for(pattern))))

    outcome = stages.run_stage0("FAKE", results_dir=tmp_path)

    assert outcome.stage.status == "completed"
    assert outcome.total_runs == 9
    assert outcome.valid_runs == 6
    assert outcome.tool_capable is True


def test_stage0_gate_fails_at_five_of_nine(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    pattern = [True] * 5 + [False] * 4
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(FakeClient(_turns_for(pattern))))

    outcome = stages.run_stage0("FAKE", results_dir=tmp_path)

    assert outcome.valid_runs == 5
    assert outcome.tool_capable is False


def test_a_second_run_resumes_and_touches_no_client(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    all_valid = [True] * 9
    empty_client = FakeClient([])
    monkeypatch.setattr(
        stages, "LMStudioClient",
        _client_factory(FakeClient(_turns_for(all_valid)), empty_client),
    )

    first = stages.run_stage0("FAKE", results_dir=tmp_path)
    assert first.total_runs == 9
    assert first.stage.session_dir == tmp_path / "FAKE-8192"

    second = stages.run_stage0("FAKE", results_dir=tmp_path)
    assert second.total_runs == 9
    assert second.valid_runs == 9
    # The second run never called stream_turn: empty_client's turns list is
    # still empty, and it was never asked to raise "no turns left" either.
    assert empty_client.turns == []


def test_unsupported_context_never_attempts_a_load(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("lmstudio.loaded must not be called for an unsupported context")

    monkeypatch.setattr(stages.environment, "load_resolved",
                         lambda *a, **k: {**RESOLVED, "advertised_max_context": 2048})
    monkeypatch.setattr(stages.lmstudio, "loaded", explode)

    outcome = stages.run_stage(
        "FAKE", STAGE0_TASKS, stage_name="stage0", suite="0",
        context_length=8192, repetitions=1, results_dir=tmp_path,
    )
    assert outcome.status == "unsupported"


def test_oversized_load_is_reported_not_raised(tmp_path, monkeypatch):
    @contextmanager
    def refuses(model, context_length=None, identifier=None, gpu=None):
        raise lmstudio.ModelOversizedError("Fake/Model does not fit: out of memory")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(stages.environment, "load_resolved", lambda *a, **k: dict(RESOLVED))
    monkeypatch.setattr(stages.lmstudio, "loaded", refuses)

    outcome = stages.run_stage(
        "FAKE", STAGE0_TASKS, stage_name="stage0", suite="0",
        context_length=8192, repetitions=1, results_dir=tmp_path,
    )
    assert outcome.status == "oversized"
    assert "out of memory" in outcome.detail


def test_environment_sha256_drift_alone_does_not_block_resume(tmp_path, monkeypatch):
    """A live run showed free_memory_bytes/swap_used_bytes_start change on
    every capture regardless of comparability, so environment_sha256 must not
    be the resume-safety signal (see `_check_task_set_version_matches`)."""
    _patch_common(monkeypatch, env_sha256="env-hash-1")
    monkeypatch.setattr(
        stages, "LMStudioClient",
        _client_factory(FakeClient(_turns_for([True] * 9))),
    )
    stages.run_stage0("FAKE", results_dir=tmp_path)

    _patch_common(monkeypatch, env_sha256="env-hash-DIFFERENT")
    empty_client = FakeClient([])
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(empty_client))

    second = stages.run_stage0("FAKE", results_dir=tmp_path)
    assert second.total_runs == 9
    assert empty_client.turns == []


def test_a_mismatched_task_set_version_on_resume_raises(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        stages, "LMStudioClient",
        _client_factory(FakeClient(_turns_for([True] * 9))),
    )
    stages.run_stage0("FAKE", results_dir=tmp_path)

    _patch_common(monkeypatch)
    monkeypatch.setattr(stages, "TASK_SET_VERSION", "v-different")
    monkeypatch.setattr(
        stages, "LMStudioClient",
        _client_factory(FakeClient(_turns_for([True] * 9))),
    )
    with pytest.raises(stages.SessionMismatchError):
        stages.run_stage0("FAKE", results_dir=tmp_path)


# --- min_context skip --------------------------------------------------------


def test_a_task_above_the_stage_context_is_skipped_not_run(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    # One normal task, one whose min_context exceeds the 8192 this stage runs
    # at. Only the normal task's turns are scripted — if the high-context task
    # were run anyway, the fake client would raise on running out of turns.
    normal = STAGE0_TASKS[0]
    too_high = dataclasses.replace(STAGE0_TASKS[1], min_context=65536)
    monkeypatch.setattr(
        stages, "LMStudioClient",
        _client_factory(FakeClient(_turns_for([True]))),
    )

    outcome = stages.run_stage(
        "FAKE", [normal, too_high], stage_name="mixed", suite="0",
        context_length=8192, repetitions=1, results_dir=tmp_path,
    )

    assert outcome.status == "completed"
    assert outcome.skipped_min_context == [too_high.id]
    assert {r["task_id"] for r in outcome.records} == {normal.id}


# --- Stage 2A gate -------------------------------------------------------


def _stage_with(records: list[dict]) -> stages.StageOutcome:
    return stages.StageOutcome(status="completed", records=records)


def _record(task_id: str, repetition: int, *, passed: bool, progress: int) -> dict:
    return {"task_id": task_id, "passed": passed, "progress_score": progress}


def test_a_task_passes_on_majority_of_its_repetitions():
    # W01: 2 of 3 pass -> counts as passed even though not unanimous.
    records = [
        _record("W01", 1, passed=True, progress=4),
        _record("W01", 2, passed=True, progress=4),
        _record("W01", 3, passed=False, progress=1),
    ]
    outcome = stages._evaluate_stage2a(_stage_with(records))
    assert outcome.tasks_total == 1
    assert outcome.tasks_passed == 1


def test_a_task_fails_on_minority_of_its_repetitions():
    records = [
        _record("W01", 1, passed=True, progress=4),
        _record("W01", 2, passed=False, progress=1),
        _record("W01", 3, passed=False, progress=1),
    ]
    outcome = stages._evaluate_stage2a(_stage_with(records))
    assert outcome.tasks_passed == 0


def test_proceeds_on_pass_count_alone():
    records = []
    for task_id in ("W01", "W02", "W03"):
        records += [_record(task_id, r, passed=True, progress=4) for r in (1, 2, 3)]
    for task_id in ("W04", "W05", "W06", "W07", "W08", "W09", "W10"):
        records += [_record(task_id, r, passed=False, progress=0) for r in (1, 2, 3)]

    outcome = stages._evaluate_stage2a(_stage_with(records))
    assert outcome.tasks_passed == 3
    assert outcome.mean_progress < 2.5
    assert outcome.proceeds is True


def test_proceeds_on_mean_progress_alone():
    records = []
    for task_id in ("W01", "W02", "W03", "W04", "W05", "W06", "W07", "W08", "W09", "W10"):
        records += [_record(task_id, r, passed=False, progress=3) for r in (1, 2, 3)]

    outcome = stages._evaluate_stage2a(_stage_with(records))
    assert outcome.tasks_passed == 0
    assert outcome.mean_progress == 3.0
    assert outcome.proceeds is True


def test_neither_gate_condition_fails_to_proceed():
    records = []
    for task_id in ("W01", "W02", "W03", "W04", "W05", "W06", "W07", "W08", "W09", "W10"):
        records += [_record(task_id, r, passed=False, progress=1) for r in (1, 2, 3)]

    outcome = stages._evaluate_stage2a(_stage_with(records))
    assert outcome.tasks_passed == 0
    assert outcome.mean_progress == 1.0
    assert outcome.proceeds is False


def test_min_context_skipped_tasks_are_excluded_not_scored_as_failing():
    """A task with no records (skipped) must not drag the mean down or count
    against tasks_total — it wasn't run, so it isn't a failure."""
    records = [_record("W01", r, passed=True, progress=4) for r in (1, 2, 3)]
    stage = stages.StageOutcome(status="completed", records=records, skipped_min_context=["W02"])

    outcome = stages._evaluate_stage2a(stage)
    assert outcome.tasks_total == 1
    assert outcome.mean_progress == 4.0


def test_gate_reports_zero_when_the_stage_did_not_complete():
    stage = stages.StageOutcome(status="oversized", detail="does not fit")
    outcome = stages._evaluate_stage2a(stage)
    assert outcome.proceeds is False
    assert outcome.tasks_total == 0
    assert outcome.mean_progress == 0.0
