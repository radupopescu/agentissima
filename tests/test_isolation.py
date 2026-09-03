"""The container closes the read gap, per doc/benchmark.md §4.6.

The gap this replaces was measured, not theorised: 29 of 240 `bash` calls in
the `v4` pi data read outside the fixture, 20 of them scanning from `/`
(findings.md). The decisive test here replays those exact commands.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil

import pytest

from conftest import container_or_skip
from harness.container import container_session

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def executor():
    container_or_skip()
    with container_session(network="bridge") as ex:
        yield ex


@pytest.fixture
def root(tmp_path):
    target = tmp_path / "workspace"
    shutil.copytree(REPO_ROOT / "fixtures" / "workspace", target)
    return target


# --- the host is not visible -------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "cat /Users/radu/.zshrc",
        "ls /Users",
        "ls ~",
        "cat /etc/passwd | grep -c Users",
    ],
)
def test_the_host_home_is_unreachable(executor, root, command):
    result = executor.run(command, cwd=root, timeout_s=15)
    assert "/Users/" not in result.output or result.exit_code != 0


def test_the_repository_itself_is_invisible(executor, root):
    """The canary: only the run directory is mounted, so this project's own
    files must be unreachable even though the run happens inside the repo."""
    result = executor.run("cat /image-manifest.json; find / -name 'benchmark.md' 2>/dev/null",
                          cwd=root, timeout_s=30)
    assert "benchmark.md" not in result.output


def test_etc_hosts_is_the_containers_own(executor, root):
    result = executor.run("cat /etc/hosts", cwd=root, timeout_s=15)
    assert result.exit_code == 0
    # The host's file names the machine; the container's does not.
    assert "agentissima" not in result.output


# --- the recorded escapes, replayed ------------------------------------------

# Extracted from the v4 pi transcripts: every bash command that referenced a
# host path outside the fixture. See findings.md 2026-09-01.
RECORDED_ESCAPES = [
    'find / -name "expense.csv" -type f 2>/dev/null | head -20',
    'find / -name "balances.py" 2>/dev/null | head -20',
    'find / -name "posting.py" -path "*ledger*" 2>/dev/null | head -20',
    'find / -name "test_posting.py" -type f 2>/dev/null | head -20',
    'find / -name "*.csv" -type f 2>/dev/null | grep -i expense | head -20',
    'find / -type f -name "*.py" 2>/dev/null | grep -i balance | head -30',
    'find / -name "*expense*" -type f 2>/dev/null | head -50',
    'find / -name "expenses.csv" 2>/dev/null | head -20',
    'find / -name "*.csv" -type f 2>/dev/null | head -50',
    'find /private -name "posting.py" -type f 2>/dev/null | head -10',
    "ls -la /root/",
    "wc -l /root/data/expenses.csv",
]


@pytest.mark.parametrize("command", RECORDED_ESCAPES)
def test_a_recorded_escape_returns_no_host_content(executor, root, command):
    """Each of these reached the operator's real filesystem under the Seatbelt
    profile. None may now return anything from outside the mount."""
    result = executor.run(command, cwd=root, timeout_s=30)
    output = result.output
    for leaked in ("/Users/", "/private/var/folders", "Workspace", "agentissima"):
        assert leaked not in output, f"{command!r} leaked {leaked!r}: {output[:200]!r}"


def test_a_whole_disk_scan_is_now_cheap(executor, root):
    """`find /` was traversing the operator's disk until pi's 30 s tool
    timeout killed it. Inside the container there is almost nothing to walk,
    so it completes rather than timing out — which also recovers the wall
    clock those runs were losing."""
    result = executor.run("find / -name '*.csv' 2>/dev/null | head -50",
                          cwd=root, timeout_s=25)
    assert not result.timed_out


# --- writes are still confined ----------------------------------------------


def test_writes_outside_the_run_directory_fail(executor, root):
    result = executor.run("echo pwned > /etc/pwned.txt", cwd=root, timeout_s=15)
    assert result.exit_code != 0


def test_writes_inside_the_run_directory_succeed(executor, root):
    """The counterpart, so the test above cannot pass vacuously."""
    executor.run("echo ok > allowed.txt", cwd=root, timeout_s=15)
    assert (root / "allowed.txt").read_text().strip() == "ok"


def test_a_file_written_in_the_container_is_removable_by_the_host(executor, root, tmp_path):
    """`--user` regression guard: container-root files would break
    `prepared()`'s cleanup and grow .runs/ without bound."""
    executor.run("mkdir -p sub && echo x > sub/file.txt", cwd=root, timeout_s=15)
    shutil.rmtree(root / "sub")  # raises if the host cannot remove it
    assert not (root / "sub").exists()


# --- the pi driver's own environment -----------------------------------------


def test_pi_and_its_find_backend_are_in_the_image(executor, root):
    """pi's `find` tool calls ensureTool("fd"), which downloads a binary when
    it cannot find one -- a network fetch mid-run. Both names must resolve from
    PATH so that path is never taken."""
    result = executor.run("command -v pi; command -v fd; command -v fdfind",
                          cwd=root, timeout_s=20)
    assert result.exit_code == 0
    assert result.output.count("/") >= 3


def test_the_cached_macos_fd_is_not_visible(executor, root):
    """setup/pi_config/bin/fd is a Mach-O binary pi cached on the host. pi
    looks in its agent dir before PATH, so if that directory were bind-mounted
    whole, pi would find a binary it cannot execute."""
    result = executor.run("file /pi-config/bin/fd 2>&1 || true", cwd=root, timeout_s=20)
    assert "Mach-O" not in result.output


def test_the_pi_config_the_container_sees_points_at_the_host(executor, root):
    """Inside the container `localhost` is the container. pi reached no model
    at all until models.json named the host explicitly."""
    result = executor.run("cat /pi-config/models.json", cwd=root, timeout_s=20)
    assert "host.docker.internal" in result.output
    assert "localhost" not in result.output


# --- a misdirected write fails where it used to succeed ----------------------


@pytest.fixture
def prepared_run(executor):
    """A real run directory, sealed as a stage would produce it."""
    from harness.runner import prepared
    from harness.tasks.workspace import build

    task = next(t for t in build() if t.id == "W07")
    with prepared(task, executor) as (sandbox, _baseline):
        yield executor, sandbox


@pytest.mark.parametrize(
    "command",
    [
        # The two commands that decided W07 runs in the `v7` campaign, from
        # results/LFM-GQ4-8192/transcripts/LFM-GQ4-pi-W-W07-r1.json and
        # results/LFM-G8-8192/transcripts/LFM-G8-pi-W-W07-r2.json.
        "echo 120 > {wd}/data/rowcount.txt",
        "mkdir -p {wd}/data",
        # The same mistake spelled relatively.
        "echo 120 > ../data/rowcount.txt",
        "touch {wd}/stray.txt",
    ],
)
def test_a_write_beside_the_fixture_root_fails_in_the_container(prepared_run, command):
    executor, sandbox = prepared_run
    wd = f"/runs/{sandbox.root.parent.name}"
    result = executor.run(command.format(wd=wd), cwd=sandbox.root, timeout_s=20)
    assert result.exit_code != 0


def test_the_task_itself_still_writes(prepared_run):
    """The counterpart, so the test above cannot pass vacuously."""
    executor, sandbox = prepared_run
    result = executor.run("echo 120 > data/rowcount.txt", cwd=sandbox.root, timeout_s=20)
    assert result.exit_code == 0
    assert (sandbox.root / "data" / "rowcount.txt").read_text().strip() == "120"


def test_reads_around_the_fixture_root_are_unaffected(prepared_run):
    executor, sandbox = prepared_run
    wd = f"/runs/{sandbox.root.parent.name}"
    result = executor.run(f"ls {wd}", cwd=sandbox.root, timeout_s=20)
    assert result.exit_code == 0
    assert "root" in result.output
