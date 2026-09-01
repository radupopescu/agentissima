"""Where a run's working files live.

`tempfile.mkdtemp()` with no `dir=` lands under `/var/folders/...` on macOS,
which Docker does not bind-mount by default. §4.6 requires the fixture copy to
be visible inside the tool container, so runs are rooted here instead: under
the repository by default, which is inside the user's home directory and so
mountable under any Docker file-sharing configuration rather than only
OrbStack's permissive one.

Keeping this explicit also stops the harness silently depending on one
container runtime's default share list.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Overridable so a caller can put runs on a different volume -- but whatever it
# points at must be inside the container's bind mount.
RUNS_ROOT = Path(os.environ.get("AGENTISSIMA_RUNS_DIR", REPO_ROOT / ".runs"))

# macOS swept `/var/folders` for us. A repository-local directory has no such
# janitor, and an interrupted stage leaves its fixture copy behind.
STALE_AFTER_S = 24 * 60 * 60


def ensure_runs_root(root: Path | None = None) -> Path:
    path = root or RUNS_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def sweep_stale(root: Path | None = None, older_than_s: float = STALE_AFTER_S) -> int:
    """Delete run directories left behind by an interrupted stage.

    Returns how many were removed. Never raises: a sweep failure must not stop
    a stage from starting.
    """
    path = ensure_runs_root(root)
    cutoff = time.time() - older_than_s
    removed = 0
    for child in path.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
