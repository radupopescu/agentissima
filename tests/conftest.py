"""Shared test setup.

`pyproject.toml` points pytest's `--basetemp` at `.runs/pytest` so that every
`tmp_path` fixture is inside the tool container's bind mount (§4.6). pytest
does not create the *parent* of `--basetemp`, so on a fresh clone, where
`.runs/` does not exist yet, every temp-dir fixture fails at setup. Creating it
here, at collection time, is what makes a first run work.
"""

from __future__ import annotations

from harness.paths import ensure_runs_root

ensure_runs_root()
