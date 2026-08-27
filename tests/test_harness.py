"""Tests for the harness contracts in doc/benchmark.md §4.4-§4.6.

These guard the properties the benchmark's validity rests on: that tool calls
are never repaired, that output truncation is exact, and that the sandbox
cannot be escaped.
"""

from __future__ import annotations

import json

import pytest

from harness.sandbox import TRUNCATE_LIMIT, Sandbox, truncate
from harness.tools import dispatch


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    return Sandbox(tmp_path, "workspace")


def call(sandbox, name, **kwargs):
    return dispatch(sandbox, name, json.dumps(kwargs))


# --- truncation -------------------------------------------------------------


def test_truncate_is_exact():
    text = "x" * (TRUNCATE_LIMIT + 25)
    out = truncate(text)
    assert out.startswith("x" * TRUNCATE_LIMIT)
    assert out.endswith("[truncated, 25 more characters]")


def test_short_output_is_untouched():
    assert truncate("short") == "short"


def test_tool_results_are_truncated(sandbox, tmp_path):
    (tmp_path / "big.txt").write_text("y" * (TRUNCATE_LIMIT + 100), encoding="utf-8")
    result = call(sandbox, "read_file", path="big.txt")
    assert "[truncated, 100 more characters]" in result.result


# --- no repair (§4.5) -------------------------------------------------------


def test_unparseable_arguments_are_invalid_not_repaired(sandbox):
    result = dispatch(sandbox, "read_file", '{"path": "data/a.txt"')
    assert result.valid is False
    assert "could not parse arguments as JSON" in result.result


def test_unknown_tool_is_invalid(sandbox):
    result = dispatch(sandbox, "read_files", '{"path": "data/a.txt"}')
    assert result.valid is False
    assert "unknown tool" in result.result


def test_missing_required_argument_is_invalid(sandbox):
    result = call(sandbox, "read_file")
    assert result.valid is False
    assert "missing required argument" in result.result


def test_wrongly_typed_argument_is_not_coerced(sandbox):
    result = dispatch(sandbox, "read_file", '{"path": 42}')
    assert result.valid is False
    assert "must be a string" in result.result


def test_non_object_arguments_are_invalid(sandbox):
    result = dispatch(sandbox, "read_file", '["data/a.txt"]')
    assert result.valid is False
    assert "must be a JSON object" in result.result


def test_valid_call_succeeds(sandbox):
    result = call(sandbox, "read_file", path="data/a.txt")
    assert result.valid is True
    assert result.result == "hello\nworld\n"


# --- sandbox containment (§4.6) --------------------------------------------


def test_path_escape_is_refused(sandbox):
    result = call(sandbox, "read_file", path="../../etc/passwd")
    assert "outside working directory" in result.result
    assert sandbox.path_errors == 1


def test_missing_file_counts_as_a_path_error(sandbox):
    call(sandbox, "read_file", path="data/nope.txt")
    assert sandbox.path_errors == 1


def test_write_then_read_round_trips(sandbox):
    assert call(sandbox, "write_file", path="out/new.txt", content="v").result == "ok"
    assert call(sandbox, "read_file", path="out/new.txt").result == "v"


def test_search_returns_path_line_text(sandbox):
    result = call(sandbox, "search_files", pattern="world", path=".")
    assert result.result == "data/a.txt:2:world"


def test_search_reports_invalid_regex(sandbox):
    result = call(sandbox, "search_files", pattern="[unclosed")
    assert "invalid regular expression" in result.result


# --- run_command allowlist --------------------------------------------------


def test_allowed_command_runs(sandbox):
    result = call(sandbox, "run_command", command="wc -l data/a.txt")
    assert result.result.startswith("exit=0")


def test_disallowed_command_is_refused(sandbox):
    result = call(sandbox, "run_command", command="curl http://example.com")
    assert result.result.startswith("exit=127")


def test_disallowed_command_cannot_hide_behind_a_pipe(sandbox):
    result = call(sandbox, "run_command", command="cat data/a.txt | sh")
    assert result.result.startswith("exit=127")


def test_command_substitution_is_refused(sandbox):
    result = call(sandbox, "run_command", command="cat $(echo data/a.txt)")
    assert "command substitution is not permitted" in result.result


def test_pytest_is_not_allowed_in_the_workspace_fixture(sandbox):
    result = call(sandbox, "run_command", command="pytest -q")
    assert result.result.startswith("exit=127")


def test_pipes_between_allowed_commands_work(sandbox):
    result = call(sandbox, "run_command", command="cat data/a.txt | wc -l")
    assert result.result.startswith("exit=0")


# --- leading slash is root-anchored inside the sandbox (§4.6) ----------------


def test_leading_slash_resolves_against_the_sandbox_root(sandbox):
    """The sandbox root is the model's whole visible filesystem, so /x denotes
    the same file as x. Without this, pathlib discards the root and the path is
    refused with a message claiming it is outside the directory."""
    assert call(sandbox, "read_file", path="/data/a.txt").result == "hello\nworld\n"
    assert call(sandbox, "read_file", path="data/a.txt").result == "hello\nworld\n"


def test_leading_slash_does_not_count_as_a_path_error(sandbox):
    call(sandbox, "read_file", path="/data/a.txt")
    assert sandbox.path_errors == 0


def test_leading_slash_list_and_search_work(sandbox):
    assert "a.txt" in call(sandbox, "list_files", path="/data").result
    assert call(sandbox, "search_files", pattern="world", path="/").result == "data/a.txt:2:world"


def test_traversal_after_a_leading_slash_is_still_refused(sandbox):
    result = call(sandbox, "read_file", path="/../../etc/passwd")
    assert "outside working directory" in result.result


def test_write_through_a_leading_slash_stays_inside(sandbox, tmp_path):
    assert call(sandbox, "write_file", path="/out/new.txt", content="v").result == "ok"
    assert (tmp_path / "out" / "new.txt").read_text() == "v"
