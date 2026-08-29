"""The §10.1 JSONL writer and resume-key reader."""

from __future__ import annotations

import pytest

from harness import results


def _record(**overrides) -> dict:
    base = dict.fromkeys(results.FIELDS)
    base.update(
        config_id="LFM-M8", driver="native", suite="0", task_id="S01", repetition=1,
        environment_sha256="abc", context_length=8192, task_set_version="v2",
        passed=True, progress_score=4, flaky=None, termination_reason="final_answer",
        steps=1, tool_calls=1, invalid_calls=0, path_errors=0, wall_clock_s=0.5,
    )
    base.update(overrides)
    return base


def test_append_and_read_round_trip(tmp_path):
    path = tmp_path / "raw" / "stage0.jsonl"
    results.append_record(path, _record())
    records = results.read_records(path)
    assert len(records) == 1
    assert records[0]["config_id"] == "LFM-M8"
    assert set(records[0]) == set(results.FIELDS)


def test_a_missing_field_is_refused(tmp_path):
    path = tmp_path / "stage0.jsonl"
    record = _record()
    del record["flaky"]
    with pytest.raises(ValueError, match="missing"):
        results.append_record(path, record)
    assert not path.exists()


def test_an_extra_field_is_refused(tmp_path):
    path = tmp_path / "stage0.jsonl"
    record = _record(bogus="nope")
    with pytest.raises(ValueError, match="unexpected"):
        results.append_record(path, record)


def test_existing_keys_is_empty_before_the_file_exists(tmp_path):
    assert results.existing_keys(tmp_path / "nowhere.jsonl") == set()


def test_existing_keys_reflects_what_was_written(tmp_path):
    path = tmp_path / "stage0.jsonl"
    results.append_record(path, _record(task_id="S01", repetition=1))
    results.append_record(path, _record(task_id="S01", repetition=2))
    results.append_record(path, _record(task_id="S02", repetition=1))

    keys = results.existing_keys(path)
    assert keys == {
        ("LFM-M8", "0", "S01", 1),
        ("LFM-M8", "0", "S01", 2),
        ("LFM-M8", "0", "S02", 1),
    }
