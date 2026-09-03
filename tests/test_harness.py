"""Tests for the harness contracts in doc/benchmark.md §4.4-§4.6.

These guard the properties the benchmark's validity rests on: that tool calls
are never repaired, that output truncation is exact, and that the sandbox
cannot be escaped.
"""

from __future__ import annotations

import json
import time

import pytest

from harness.execution import RecordingExecutor
from harness.sandbox import TRUNCATE_LIMIT, Sandbox, truncate
from harness.tools import dispatch


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    return Sandbox(tmp_path, "workspace")


@pytest.fixture
def refusing_sandbox(tmp_path):
    """A sandbox whose executor records instead of running.

    Lets the allowlist tests assert that a refused command never reached
    execution, rather than inferring it from the error string.
    """
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    recorder = RecordingExecutor()
    return Sandbox(tmp_path, "workspace", recorder), recorder


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


def test_backgrounding_cannot_smuggle_a_disallowed_command(sandbox):
    """A bare `&` used to fall through unchecked: only the first segment's
    head token was validated, so `wc -l a.txt & sleep 5` ran `sleep` — not on
    any allowlist — anyway."""
    result = call(sandbox, "run_command", command="wc -l data/a.txt & sleep 5")
    assert result.result.startswith("exit=127")


@pytest.mark.parametrize("command", ["wc -l data/a.txt 2>&1", "wc -l data/a.txt 1>&2"])
def test_stderr_redirection_is_not_mistaken_for_backgrounding(sandbox, command):
    """Regression: splitting on a bare `&` for backgrounding also split
    `2>&1`/`1>&2` in two, refusing an ordinary command with a nonsensical
    "command not permitted: 1" — found for real in LFM-G8's Stage 2A data,
    on every task where the model reached for this exact idiom."""
    result = call(sandbox, "run_command", command=command)
    assert result.result.startswith("exit=0")


# --- run_command cannot reach outside the sandbox ---------------------------
#
# A live run against LM Studio found this for real: `grep -r expense /` and
# `find / ...` (allowed heads, absolute-path arguments) reached the real
# filesystem, because run_command has no structured path parameter for the
# leading-`/` root-anchoring the other tools apply — see findings.md,
# 2026-08-29.


@pytest.mark.parametrize(
    "command",
    [
        "grep -r expense /",
        "find / -name '*.csv'",
        "cat /etc/hosts",
        "ls ~/",
        "cat ../../etc/passwd",
        "cat data/../../etc/passwd",
    ],
)
def test_absolute_and_traversal_arguments_are_refused(sandbox, command):
    result = call(sandbox, "run_command", command=command)
    assert result.result.startswith("exit=127")
    assert "outside working directory" in result.result


def test_a_relative_path_argument_still_works(sandbox):
    result = call(sandbox, "run_command", command="cat data/a.txt")
    assert result.result.startswith("exit=0")


def test_timeout_kills_the_whole_process_group(sandbox, tmp_path):
    """Regression: `subprocess.run(shell=True, timeout=)` kills only the
    top-level shell on timeout, not anything it forked — a child process kept
    running as an orphan indefinitely. This is exactly what happened for real
    with `grep -r ... /` (findings.md, 2026-08-29): the 30s timeout fired and
    reported cleanly, but the underlying grep kept scanning the real disk for
    over an hour, unobserved."""
    marker = tmp_path / "child-finished.txt"
    (tmp_path / "spawn_child.py").write_text(
        "import subprocess, time\n"
        "subprocess.Popen(['python', '-c', "
        "'import time; time.sleep(0.5); open(\"child-finished.txt\", \"w\").write(\"done\")'])\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    result = sandbox.run_command("python spawn_child.py", timeout_s=0.2)
    assert result.startswith("exit=124")

    time.sleep(1.0)  # well past the 0.5s the child needs to finish, if still alive
    assert not marker.exists(), "the forked child kept running after the timeout"


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


# --- refusal happens before execution ----------------------------------------
#
# The tests above assert a refused command returns exit=127. That alone does
# not prove nothing ran -- it would also pass if the command ran and happened
# to fail. §4.5 makes the refusal itself the measurement, so it must not
# depend on the refused command being harmless. These assert the stronger
# property against a recording executor.


@pytest.mark.parametrize(
    "command",
    [
        "curl http://example.com",
        "cat data/a.txt | sh",
        "cat $(echo data/a.txt)",
        "pytest -q",
        "wc -l a.txt & sleep 5",
        "cat /etc/hosts",
        "cat ../outside.txt",
    ],
)
def test_a_refused_command_never_reaches_the_executor(refusing_sandbox, command):
    sandbox, recorder = refusing_sandbox
    result = call(sandbox, "run_command", command=command)
    assert result.result.startswith("exit=127")
    assert recorder.calls == []


def test_an_allowed_command_does_reach_the_executor(refusing_sandbox):
    """The counterpart: proves the previous test is not vacuous."""
    sandbox, recorder = refusing_sandbox
    call(sandbox, "run_command", command="wc -l a.txt")
    assert [cmd for cmd, _ in recorder.calls] == ["wc -l a.txt"]


# --- the change baseline is not left inside the agent's reach ----------------
#
# The whole runs root is bind-mounted into the tool container, so a reference
# *copy* of the fixture sat beside the working copy where the agent could read
# it -- 8.6% of the `v6` pi runs did, and a write into it would have defeated
# `changed_paths`, which W07, T07 and T03 decide their verdict with. Only
# hashes were ever needed.


def _prepared_workspace():
    from harness.runner import prepared
    from harness.tasks.workspace import build

    task = next(t for t in build() if t.id == "W07")
    return prepared(task)


def test_a_run_directory_holds_no_second_copy_of_the_fixture():
    with _prepared_workspace() as (sandbox, _baseline):
        workdir = sandbox.root.parent
        assert [p.name for p in workdir.iterdir()] == ["root"]


def test_the_baseline_still_sees_a_change(tmp_path):
    from harness.assertions import changed_paths
    from harness.execution import HostExecutor
    from harness.types import Ctx

    with _prepared_workspace() as (sandbox, baseline):
        (sandbox.root / "data" / "rowcount.txt").write_text("120\n", encoding="utf-8")
        ctx = Ctx(
            root=sandbox.root, baseline=baseline, answer="", calls=[],
            expected={}, path_errors=0, executor=HostExecutor(),
        )
        assert changed_paths(ctx) == {"data/rowcount.txt"}


def test_the_baseline_sees_no_change_in_an_untouched_tree():
    """The counterpart, so the test above cannot pass vacuously."""
    from harness.assertions import changed_paths
    from harness.execution import HostExecutor
    from harness.types import Ctx

    with _prepared_workspace() as (sandbox, baseline):
        ctx = Ctx(
            root=sandbox.root, baseline=baseline, answer="", calls=[],
            expected={}, path_errors=0, executor=HostExecutor(),
        )
        assert changed_paths(ctx) == set()


# --- the work directory around the fixture is sealed -------------------------
#
# The tool container mounts the whole runs root and runs as this uid, so a
# write beside `root/` used to succeed silently: two `v7` W07 runs computed the
# right answer, wrote it to `<workdir>/data/`, read it back to confirm, and
# reported success while grading saw nothing (findings.md, 2026-09-03).


def test_a_write_beside_the_fixture_root_is_refused():
    import pytest as _pytest
    from harness.runner import prepared
    from harness.tasks.workspace import build

    task = next(t for t in build() if t.id == "W07")
    with prepared(task) as (sandbox, _baseline):
        with _pytest.raises(PermissionError):
            (sandbox.root.parent / "data").mkdir()
        with _pytest.raises(PermissionError):
            (sandbox.root.parent / "stray.txt").write_text("x", encoding="utf-8")


def test_writes_inside_the_fixture_root_still_work():
    """The counterpart: the seal must not break the task itself."""
    from harness.runner import prepared
    from harness.tasks.workspace import build

    task = next(t for t in build() if t.id == "W07")
    with prepared(task) as (sandbox, _baseline):
        (sandbox.root / "data" / "rowcount.txt").write_text("120", encoding="utf-8")
        assert (sandbox.root / "data" / "rowcount.txt").read_text() == "120"


def test_the_sealed_work_directory_is_still_readable():
    """Reads and traversal are untouched — only writing beside `root/` fails."""
    from harness.runner import prepared
    from harness.tasks.workspace import build

    task = next(t for t in build() if t.id == "W07")
    with prepared(task) as (sandbox, _baseline):
        assert [p.name for p in sandbox.root.parent.iterdir()] == ["root"]
        assert (sandbox.root / "data" / "expenses.csv").read_text().startswith("id,date")


def test_the_work_directory_is_cleaned_up_despite_the_seal():
    """Cleanup unlinks `root/` out of the work directory, which needs write
    permission back — a seal left in place would leak .runs/ without bound."""
    from harness.runner import prepared
    from harness.tasks.workspace import build

    task = next(t for t in build() if t.id == "W07")
    with prepared(task) as (sandbox, _baseline):
        workdir = sandbox.root.parent
    assert not workdir.exists()
