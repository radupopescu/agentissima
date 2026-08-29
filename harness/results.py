"""The §10.1 JSONL record schema, and reading it back for resume.

One JSONL file per stage, under `results/<session>/raw/<stage_name>.jsonl`.
Append-only (`way-of-working.md`): a run's record is written once and never
rewritten. `flaky` is therefore always written `null` here — deciding it
needs every repetition of a task, which resuming across runs means cannot
always be known yet at write time, and `way-of-working.md` already commits
reporting to regenerating everything from JSONL, so grouping and flakiness
belong to `harness/report.py`, not to the write path.
"""

from __future__ import annotations

import json
from pathlib import Path

FIELDS: tuple[str, ...] = (
    "run_id", "session_id", "config_id", "driver", "suite", "task_id", "repetition",
    "environment_sha256", "context_length", "task_set_version",
    "ttft_s", "gen_tps", "prompt_tps", "ttft_turn1_s", "ttft_median_later_s",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "peak_memory_bytes", "swap_delta_bytes", "swap_flag",
    "steps", "tool_calls", "invalid_calls", "path_errors",
    "termination_reason", "passed", "progress_score", "flaky",
    "wall_clock_s", "transcript_path",
)

RunKey = tuple[str, str, str, int]


def run_key(record: dict) -> RunKey:
    return (record["config_id"], record["suite"], record["task_id"], record["repetition"])


def append_record(path: Path, record: dict) -> None:
    """Append one §10.1 record. Refuses a record with a missing or extra
    field — a silently malformed line would otherwise only surface much
    later, in a report that reads it back."""
    got = set(record)
    wanted = set(FIELDS)
    if got != wanted:
        missing = wanted - got
        extra = got - wanted
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)}")
        if extra:
            detail.append(f"unexpected {sorted(extra)}")
        raise ValueError(f"record does not match the §10.1 schema: {'; '.join(detail)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({field: record[field] for field in FIELDS}, sort_keys=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def existing_keys(path: Path) -> set[RunKey]:
    """Run keys already written, for resume. Empty for a session's first run
    of a stage — the file need not exist yet."""
    return {run_key(record) for record in read_records(path)}
