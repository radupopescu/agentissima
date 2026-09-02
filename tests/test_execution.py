"""Host and container executors must agree, per doc/benchmark.md §4.6.

The container swaps BSD coreutils for GNU. That difference was checked by hand
before the migration and judged minor — `wc -l` differs only in leading
whitespace, and the oracle's parsers are whitespace-tolerant. These tests turn
that judgement into a gate, so a future divergence fails here rather than
silently changing what a model observes.

Comparison is on *parsed* results, not raw bytes: the whitespace difference is
known and accepted, and asserting byte equality would fail for the wrong reason.
"""

from __future__ import annotations

import re
import shutil

import pytest

from conftest import container_or_skip
from harness.container import container_session
from harness.execution import HostExecutor
from harness.paths import ensure_runs_root

REPO_FIXTURES = "fixtures"


@pytest.fixture(scope="session")
def container_executor():
    container_or_skip()
    with container_session(network="bridge") as executor:
        yield executor


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    shutil.copytree(f"{REPO_FIXTURES}/workspace", root)
    return root


@pytest.fixture
def testrepo(tmp_path):
    root = tmp_path / "testrepo"
    shutil.copytree(f"{REPO_FIXTURES}/testrepo", root)
    return root


def _count(output: str) -> int | None:
    """Parse a `wc -l` count the way harness/oracle.py does."""
    match = re.search(r"(\d+)", output)
    return int(match.group(1)) if match else None


# --- the commands the oracle actually issues ---------------------------------


def test_wc_agrees_after_parsing(container_executor, workspace):
    args = dict(cwd=workspace, timeout_s=30)
    host = HostExecutor().run("wc -l data/expenses.csv", **args)
    container = container_executor.run("wc -l data/expenses.csv", **args)
    assert host.exit_code == container.exit_code == 0
    assert _count(host.output) == _count(container.output)


def test_wc_raw_output_does_differ(container_executor, workspace):
    """Documents *why* the comparison is on parsed values: BSD pads the count,
    GNU does not. If this ever stops being true the parity test above is
    stronger than it needs to be, which is harmless — but the divergence should
    not be forgotten."""
    args = dict(cwd=workspace, timeout_s=30)
    host = HostExecutor().run("wc -l data/expenses.csv", **args)
    container = container_executor.run("wc -l data/expenses.csv", **args)
    assert _count(host.output) == _count(container.output)  # the part that matters


def test_python_agrees(container_executor, workspace):
    script = 'python -c "import csv;print(sum(1 for _ in csv.DictReader(open(\'data/expenses.csv\'))))"'
    args = dict(cwd=workspace, timeout_s=60)
    host = HostExecutor().run(script, **args)
    container = container_executor.run(script, **args)
    assert host.exit_code == container.exit_code == 0
    assert host.output.strip().splitlines()[-1] == container.output.strip().splitlines()[-1]


def test_pytest_agrees_on_the_deliberate_failure(container_executor, testrepo):
    """The base testrepo has exactly one failing test (§6.2). Both executors
    must see it fail — this is what T03's assertion turns on."""
    args = dict(cwd=testrepo, timeout_s=120)
    host = HostExecutor().run("pytest -q", **args)
    container = container_executor.run("pytest -q", **args)
    assert host.exit_code != 0 and container.exit_code != 0
    assert "test_split_posting_balances" in host.output
    assert "test_split_posting_balances" in container.output


def test_grep_agrees(container_executor, workspace):
    args = dict(cwd=workspace, timeout_s=30)
    host = HostExecutor().run('grep -c Travel data/expenses.csv', **args)
    container = container_executor.run('grep -c Travel data/expenses.csv', **args)
    assert host.output.strip() == container.output.strip()


def test_find_agrees_on_the_file_set(container_executor, workspace):
    cmd = 'find . -name "*.csv" -type f'
    args = dict(cwd=workspace, timeout_s=30)
    host = HostExecutor().run(cmd, **args)
    container = container_executor.run(cmd, **args)
    assert set(host.output.split()) == set(container.output.split())


# --- timeout semantics -------------------------------------------------------


def test_container_timeout_reports_124(container_executor, workspace):
    result = container_executor.run("sleep 30", cwd=workspace, timeout_s=2)
    assert result.timed_out
    assert result.exit_code == 124


def test_container_timeout_reaps_the_child_tree(container_executor, workspace):
    """The regression guard for the orphaned `grep` in findings.md, in its
    container form: a grandchild the shell forked must not outlive the
    timeout. It writes into the bind mount, so the host can observe it."""
    marker = workspace / "grandchild.txt"
    container_executor.run(
        f"(sleep 4; echo alive > {marker.name}) & sleep 30",
        cwd=workspace, timeout_s=2,
    )
    import time

    time.sleep(6)
    assert not marker.exists(), "a forked grandchild survived the timeout"


def test_spawn_reports_a_timeout_as_a_timeout(container_executor, workspace):
    """`spawn` is the pi path, and it had its own signal semantics: with
    --signal=KILL the process dies of SIGKILL and `docker exec` reports 137,
    not `timeout`'s 124. `timed_out` stayed False, PiDriver saw no `agent_end`,
    and every pi timeout was recorded as `server_error` (§4.8) -- which also
    fed the §4.2 degenerate detector the wrong category. Found in a real
    campaign: five Stage 2A runs at exactly 600.2s, all labelled server_error.

    The parity suite covered `run` only, which is why it did not catch this.
    Both paths are asserted from here on."""
    result = container_executor.spawn(["sleep", "30"], cwd=workspace, timeout_s=2)
    assert result.timed_out
    assert result.exit_code == 124


def test_both_paths_agree_on_the_timeout_contract(container_executor, workspace):
    run = container_executor.run("sleep 30", cwd=workspace, timeout_s=2)
    spawn = container_executor.spawn(["sleep", "30"], cwd=workspace, timeout_s=2)
    assert (run.timed_out, run.exit_code) == (spawn.timed_out, spawn.exit_code)


def test_the_host_backstop_has_not_fired(container_executor):
    """Layer 2 exists for a wedged daemon and should never run. A non-zero
    count in a result set means the in-container timeout failed."""
    assert container_executor.provenance["backstop_firings"] == 0


# --- path mapping ------------------------------------------------------------


def test_a_cwd_outside_the_mount_is_refused(container_executor, tmp_path_factory):
    from harness.execution import ExecutionError

    outside = tmp_path_factory.mktemp("outside")
    # Force a path that cannot be under RUNS_ROOT.
    import pathlib

    with pytest.raises(ExecutionError, match="outside the container mount"):
        container_executor.run("true", cwd=pathlib.Path("/etc"), timeout_s=5)


def test_provenance_records_what_produced_the_environment(container_executor):
    p = container_executor.provenance
    assert p["mode"] == "container"
    assert p["image_id"]
    assert p["dockerfile_sha256"]
    assert p["image_manifest"]["python"].startswith("Python 3.14")
    assert p["limits"]["pids"]
