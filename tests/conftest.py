"""Shared test setup.

`pyproject.toml` points pytest's `--basetemp` at `.runs/pytest` so that every
`tmp_path` fixture is inside the tool container's bind mount (§4.6). pytest
does not create the *parent* of `--basetemp`, so on a fresh clone, where
`.runs/` does not exist yet, every temp-dir fixture fails at setup. Creating it
here, at collection time, is what makes a first run work.
"""

from __future__ import annotations

import os

import pytest

from harness.container import docker_available, image_exists
from harness.paths import ensure_runs_root

ensure_runs_root()

# The container tests skip when docker or the image is missing, so a machine
# without either would report a green suite while never exercising the
# environment §4.6 actually measures in. Setting this turns that skip into a
# failure. Set it in CI and wherever a result set is about to be produced.
REQUIRE_CONTAINER = os.environ.get("AGENTISSIMA_REQUIRE_CONTAINER") == "1"


def container_or_skip():
    """Shared gate for tests that need the tool container."""
    if docker_available() and image_exists():
        return
    reason = "docker or the tool image is unavailable"
    if REQUIRE_CONTAINER:
        pytest.fail(f"AGENTISSIMA_REQUIRE_CONTAINER=1 but {reason}")
    pytest.skip(reason)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say plainly, in the summary, when the container half did not run."""
    if docker_available() and image_exists():
        return
    terminalreporter.write_sep(
        "!", "container tests did NOT run - this suite did not validate §4.6", red=True
    )
