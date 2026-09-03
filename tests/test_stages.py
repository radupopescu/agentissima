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
from harness.execution import HostExecutor
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

    def factory(model, sampling=None, extra_body=None):
        """`sampling`/`extra_body` are passed by every stage; only Stage 5B's
        sampling pass sends anything but `None`. The fake records them so a
        test can assert what the stage asked for."""
        if not remaining:
            raise AssertionError("LMStudioClient constructed more times than expected")
        client = remaining.pop(0)
        client.requested_sampling = sampling
        client.requested_extra_body = extra_body
        return client

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


@contextmanager
def _fake_container_session(*args, **kwargs):
    """Stand-in for the §4.6 container: runs on the host instead."""
    yield HostExecutor()


def _patch_common(monkeypatch, resolved=RESOLVED, env_sha256="env-hash-1"):
    monkeypatch.setattr(stages.environment, "load_resolved", lambda *a, **k: dict(resolved))
    monkeypatch.setattr(stages.lmstudio, "loaded", _fake_loaded)
    monkeypatch.setattr(
        stages.environment, "capture",
        lambda *a, **k: types.SimpleNamespace(sha256=env_sha256),
    )
    monkeypatch.setattr(stages, "find_inference_pid", lambda: None)
    # These tests cover stage *logic* -- gates, resume, record shape. The
    # execution environment is covered by tests/test_execution.py and
    # tests/test_isolation.py, so starting a real container here would add a
    # docker dependency and ~27 container starts for no extra coverage.
    monkeypatch.setattr(stages, "container_session", _fake_container_session)


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


# --- Stage 1: raw inference --------------------------------------------------


def _raw_turn(completion_tokens: int, finish_reason: str = "stop"):
    return StreamedTurn(
        content="x" * completion_tokens, t_request=0.0, t_first=0.1, t_last=0.2,
        prompt_tokens=8000, completion_tokens=completion_tokens, finish_reason=finish_reason,
    )


class RawFakeClient:
    """Stands in for `LMStudioClient` for `run_stage1`, which calls
    `stream_turn` directly with one message and no tool schema."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def measure_overhead(self, **kwargs):
        return 0.01

    def stream_turn(self, messages, tools=None, clock=None):
        self.seen.append(messages)
        if not self.turns:
            raise AssertionError("stream_turn called with no scripted turns left")
        return self.turns.pop(0)


def test_stage1_retries_once_when_completion_is_short(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    client = RawFakeClient([_raw_turn(50), _raw_turn(200)])
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(client))

    outcome = stages.run_stage1("FAKE", "8k", repetitions=1, results_dir=tmp_path)

    assert outcome.status == "completed"
    assert len(outcome.records) == 1
    assert outcome.records[0]["completion_tokens"] == 200
    assert len(client.seen) == 2  # primary, then the alternate retry


def test_stage1_does_not_retry_when_completion_is_long_enough(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    client = RawFakeClient([_raw_turn(150)])
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(client))

    outcome = stages.run_stage1("FAKE", "8k", repetitions=1, results_dir=tmp_path)

    assert outcome.records[0]["completion_tokens"] == 150
    assert len(client.seen) == 1


def test_stage1_records_have_no_task_progress_or_tool_calls(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    client = RawFakeClient([_raw_turn(150)])
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(client))

    outcome = stages.run_stage1("FAKE", "8k", repetitions=1, results_dir=tmp_path)

    record = outcome.records[0]
    assert record["passed"] is None
    assert record["progress_score"] is None
    assert record["tool_calls"] == 0
    assert record["suite"] == "1"
    assert record["task_id"] == "8k"


def test_stage1_resumes(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    client = RawFakeClient([_raw_turn(200) for _ in range(5)])
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(client))
    first = stages.run_stage1("FAKE", "8k", results_dir=tmp_path)
    assert len(first.records) == 5

    empty = RawFakeClient([])
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(empty))
    second = stages.run_stage1("FAKE", "8k", results_dir=tmp_path)
    assert len(second.records) == 5
    assert empty.seen == []


# --- run_stages: sequencing and gating ---------------------------------------
#
# These mock at the stage-function level (run_stage0, run_stage1, ...), not
# the client level: each of those is already independently tested above, and
# scripting a full stage-construction success path down to individual tool
# calls would mean hand-writing hundreds of turns to test logic that doesn't
# touch a client at all. What's novel in run_stages is the sequencing and
# gating, which this exercises directly.


def _stage_outcome(status="completed", records=None):
    return stages.StageOutcome(status=status, records=records or [])


def _stage0_outcome(tool_capable, status="completed"):
    return stages.Stage0Outcome(
        tool_capable=tool_capable, valid_runs=9 if tool_capable else 0, total_runs=9,
        stage=_stage_outcome(status),
    )


def _stage2a_outcome(proceeds, status="completed"):
    return stages.Stage2AOutcome(
        proceeds=proceeds, tasks_passed=5 if proceeds else 0, tasks_total=10,
        mean_progress=3.0 if proceeds else 1.0, stage=_stage_outcome(status),
    )


def test_run_stages_stops_when_not_tool_capable(monkeypatch):
    monkeypatch.setattr(stages, "run_stage0", lambda *a, **k: _stage0_outcome(False))

    def explode(*a, **k):
        raise AssertionError("must not run Stage 1 when Stage 0 failed its gate")

    monkeypatch.setattr(stages, "run_stage1", explode)

    outcome = stages.run_stages("FAKE", ["stage0", "stage1", "stage2a", "stage2b"])
    assert outcome.stopped_at == "stage0"
    assert "stage1" not in outcome.results


def test_run_stages_stops_when_stage0_does_not_complete(monkeypatch):
    monkeypatch.setattr(
        stages, "run_stage0", lambda *a, **k: _stage0_outcome(False, status="oversized")
    )
    outcome = stages.run_stages("FAKE", ["stage0", "stage1"])
    assert outcome.stopped_at == "stage0"
    assert outcome.results["stage0"].stage.status == "oversized"


def test_run_stages_stops_when_stage2a_gate_fails(monkeypatch):
    monkeypatch.setattr(stages, "run_stage0", lambda *a, **k: _stage0_outcome(True))
    monkeypatch.setattr(stages, "run_stage1", lambda config_id, tier, **k: _stage_outcome())
    monkeypatch.setattr(
        stages, "run_stage2a", lambda config_id, *, driver, **k: _stage2a_outcome(False)
    )

    def explode(*a, **k):
        raise AssertionError("must not run Stage 2B when the 2A gate failed")

    monkeypatch.setattr(stages, "run_stage2b", explode)

    outcome = stages.run_stages("FAKE", ["stage0", "stage1", "stage2a", "stage2b"])
    assert outcome.stopped_at == "stage2a"
    assert len(outcome.results["stage1"]) == 2
    assert "stage2b" not in outcome.results


def test_run_stages_stops_when_stage2b_does_not_complete(monkeypatch):
    monkeypatch.setattr(stages, "run_stage0", lambda *a, **k: _stage0_outcome(True))
    monkeypatch.setattr(stages, "run_stage1", lambda config_id, tier, **k: _stage_outcome())
    monkeypatch.setattr(
        stages, "run_stage2a", lambda config_id, *, driver, **k: _stage2a_outcome(True)
    )
    monkeypatch.setattr(
        stages, "run_stage2b",
        lambda config_id, *, driver, **k: _stage_outcome(status="oversized"),
    )

    outcome = stages.run_stages("FAKE", ["stage0", "stage1", "stage2a", "stage2b"])
    assert outcome.stopped_at == "stage2b"
    assert outcome.results["stage2b"].status == "oversized"


def test_run_stages_completes_every_requested_stage_on_success(monkeypatch):
    monkeypatch.setattr(stages, "run_stage0", lambda *a, **k: _stage0_outcome(True))
    monkeypatch.setattr(stages, "run_stage1", lambda config_id, tier, **k: _stage_outcome())
    monkeypatch.setattr(
        stages, "run_stage2a", lambda config_id, *, driver, **k: _stage2a_outcome(True)
    )
    monkeypatch.setattr(
        stages, "run_stage2b", lambda config_id, *, driver, **k: _stage_outcome()
    )

    outcome = stages.run_stages("FAKE", ["stage0", "stage1", "stage2a", "stage2b"])
    assert outcome.stopped_at is None
    assert outcome.results["stage2b"].status == "completed"


def test_run_stages_threads_driver_through_to_stage2a_and_2b(monkeypatch):
    seen = []
    monkeypatch.setattr(
        stages, "run_stage2a",
        lambda config_id, *, driver, **k: (seen.append(("2a", driver)), _stage2a_outcome(True))[1],
    )
    monkeypatch.setattr(
        stages, "run_stage2b",
        lambda config_id, *, driver, **k: (seen.append(("2b", driver)), _stage_outcome())[1],
    )

    outcome = stages.run_stages("FAKE", ["stage2a", "stage2b"], driver="pi")
    assert outcome.driver == "pi"
    assert seen == [("2a", "pi"), ("2b", "pi")]


# --- run_id uniqueness across drivers ---------------------------------------
#
# Two drivers running the same task once shared a `run_id`, and therefore a
# transcript filename, so the later stage overwrote the earlier one's
# transcripts (findings.md). The identifier now carries the driver.


def _stub_driver_with_transcript(task, sandbox):
    from harness.types import RunOutcome

    return RunOutcome(
        task_id=task.id,
        root=sandbox.root,
        answer="stub",
        transcript=[{"role": "assistant", "content": "stub"}],
    )


def _record_with_driver(tmp_path, driver_label):
    return stages._record_for(
        STAGE0_TASKS[0],
        1,
        config_id="FAKE",
        suite="W",
        session_id="FAKE-8192",
        environment_sha256="deadbeef",
        context_length=8192,
        driver=_stub_driver_with_transcript,
        driver_label=driver_label,
        pid=None,
        transcripts_dir=tmp_path / "transcripts",
    )


def test_run_id_and_transcript_path_differ_per_driver(tmp_path):
    native = _record_with_driver(tmp_path, "native")
    pi = _record_with_driver(tmp_path, "pi")

    assert native["run_id"] != pi["run_id"]
    assert native["transcript_path"] != pi["transcript_path"]


def test_a_second_driver_does_not_overwrite_the_first_transcript(tmp_path):
    native = _record_with_driver(tmp_path, "native")
    _record_with_driver(tmp_path, "pi")

    from pathlib import Path

    assert Path(native["transcript_path"]).is_file()
    assert len(list((tmp_path / "transcripts").glob("*.json"))) == 2


# --- Stage 3 and Stage 5B's sampling pass stay out of the controlled tables ---
#
# report.py pools everything under raw/ and selects on `driver`, so a stage
# that must not appear in the §10 tables needs both its own raw file and its
# own driver label (§4.1).


def test_stage3_writes_a_driver_specific_raw_file(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    captured = []
    monkeypatch.setattr(
        stages, "run_stage",
        lambda *a, **k: captured.append(k) or stages.StageOutcome(status="completed", records=[]),
    )

    stages.run_stage3("FAKE", driver="pi", results_dir=tmp_path)
    assert [k["stage_name"] for k in captured] == ["stage3-pi", "stage3-pi"]
    assert {k["driver_label"] for k in captured} == {"pi"}
    assert {k["context_length"] for k in captured} == {16384}
    assert [k["suite"] for k in captured] == ["W", "T"]


def test_stage3_defaults_to_the_controlled_driver(tmp_path, monkeypatch):
    """`pi` since `v5` (§4.1). Stage 3 had no CLI and no driver argument until
    it was needed, and defaulting to `native` would have quietly produced an
    arm nothing else compares with."""
    _patch_common(monkeypatch)
    captured = []
    monkeypatch.setattr(
        stages, "run_stage",
        lambda *a, **k: captured.append(k) or stages.StageOutcome(status="completed", records=[]),
    )

    stages.run_stage3("FAKE", results_dir=tmp_path)
    assert {k["driver_label"] for k in captured} == {"pi"}


def test_the_sampling_pass_labels_its_driver_as_sampled(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    from harness.sampling import RecommendedSampling

    recommended = RecommendedSampling(
        config_id="FAKE", source="fake.gguf", stated={"temperature": 0.1, "top_k": 50}
    )
    monkeypatch.setattr(stages.sampling_defaults, "resolve", lambda *a, **k: recommended)
    captured = []
    monkeypatch.setattr(
        stages, "run_stage",
        lambda *a, **k: captured.append(k) or stages.StageOutcome(status="completed", records=[]),
    )

    resolved, _outcomes = stages.run_stage5b_sampling("FAKE", results_dir=tmp_path)
    assert {k["driver_label"] for k in captured} == {"native-sampled"}
    assert [k["stage_name"] for k in captured] == [
        "stage5b-sampling-w", "stage5b-sampling-t",
    ]
    assert all(k["sampling"] is recommended for k in captured)
    assert resolved is recommended


def test_a_sampled_driver_is_not_a_comparison_driver():
    """The label is what keeps it out of the tables, so assert the mechanism
    rather than trusting the name."""
    from harness.report import COMPARISON_DRIVERS

    assert "native-sampled" not in COMPARISON_DRIVERS
    assert "pi-sampled" not in COMPARISON_DRIVERS


def test_the_sampling_pass_sends_the_recommended_values_to_the_client(tmp_path, monkeypatch):
    """End to end through `run_stage`: the fake client records what it was
    constructed with, so this proves the override reaches the request rather
    than stopping at the stage boundary."""
    _patch_common(monkeypatch)
    from harness.sampling import RecommendedSampling

    recommended = RecommendedSampling(
        config_id="FAKE", source="fake.gguf", stated={"temperature": 0.2, "top_k": 80}
    )
    monkeypatch.setattr(stages.sampling_defaults, "resolve", lambda *a, **k: recommended)
    client = FakeClient(_turns_for([True] * 60))
    monkeypatch.setattr(stages, "LMStudioClient", _client_factory(client, client))

    stages.run_stage5b_sampling("FAKE", results_dir=tmp_path)
    assert client.requested_sampling["temperature"] == 0.2
    assert client.requested_extra_body["top_k"] == 80
    # Pinned regardless of what the artefact said.
    assert client.requested_sampling["seed"] == 1337
