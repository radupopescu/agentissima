"""Version identifiers recorded per session (§3) and pinned by §11.

Single source of truth for the version strings the harness writes. The task
set version is fixed by `doc/benchmark.md`'s header; §11 lists the changes
that must bump it.
"""

from __future__ import annotations

TASK_SET_VERSION = "v5"