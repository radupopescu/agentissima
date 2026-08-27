"""Metrics, per doc/benchmark.md §5.

Every term here is defined against a specific observable so that two
implementations agree. Where a value cannot be measured it is recorded as
``None`` and never replaced by an estimate (§5.3).
"""

from __future__ import annotations

import re
import secrets
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

from .client import StreamedTurn

NONCE_BYTES = 8  # 16 hex characters, per §5.4

_FOOTPRINT_HEADER = re.compile(r"Footprint:\s*([\d.]+)\s*([KMGT]?B)", re.IGNORECASE)
_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


# --- §5.4 prompt-cache handling ---------------------------------------------


def nonce_prefix() -> str:
    """A fresh 16-character nonce, guaranteeing a KV-prefix cache miss.

    Phase 1 only. Phase 2 leaves caching enabled because it reflects real agent
    use, at the cost of prompt tok/s after turn 1 (§5.4).
    """
    return secrets.token_hex(NONCE_BYTES)


# --- §5.1 timing ------------------------------------------------------------


@dataclass
class TurnMetrics:
    ttft_s: float | None
    gen_tps: float | None
    prompt_tps: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None = None


def turn_metrics(turn: StreamedTurn, overhead_s: float = 0.0) -> TurnMetrics:
    """Derive §5.1 metrics from one streamed turn.

    Generation throughput divides by ``completion_tokens - 1``: the first token
    had already arrived at ``t_first`` and did not occur inside the generation
    window.
    """
    ttft = turn.ttft_s
    generation = turn.generation_s

    gen_tps = None
    if (
        generation is not None
        and generation > 0
        and turn.completion_tokens is not None
        and turn.completion_tokens > 1
    ):
        gen_tps = (turn.completion_tokens - 1) / generation

    prompt_tps = None
    if ttft is not None and turn.prompt_tokens:
        adjusted = ttft - overhead_s
        if adjusted > 0:
            prompt_tps = turn.prompt_tokens / adjusted

    return TurnMetrics(
        ttft_s=ttft,
        gen_tps=gen_tps,
        prompt_tps=prompt_tps,
        prompt_tokens=turn.prompt_tokens,
        completion_tokens=turn.completion_tokens,
        reasoning_tokens=turn.reasoning_tokens,
    )


@dataclass
class RunTiming:
    """Aggregated timing across the turns of one agent run.

    Turn 1 is reported separately from later turns because Phase 2 leaves the
    prompt cache enabled, so later turns are not comparable (§5.4).
    """

    turns: list[TurnMetrics] = field(default_factory=list)

    def add(self, metrics: TurnMetrics) -> None:
        self.turns.append(metrics)

    @property
    def ttft_turn1_s(self) -> float | None:
        return self.turns[0].ttft_s if self.turns else None

    @property
    def ttft_median_later_s(self) -> float | None:
        later = [t.ttft_s for t in self.turns[1:] if t.ttft_s is not None]
        return statistics.median(later) if later else None

    @property
    def gen_tps(self) -> float | None:
        values = [t.gen_tps for t in self.turns if t.gen_tps is not None]
        return statistics.median(values) if values else None

    @property
    def prompt_tps(self) -> float | None:
        # Turn 1 only: later turns hit the prompt cache (§5.4).
        return self.turns[0].prompt_tps if self.turns else None

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens or 0 for t in self.turns)

    @property
    def completion_tokens(self) -> int:
        return sum(t.completion_tokens or 0 for t in self.turns)

    @property
    def reasoning_tokens(self) -> int:
        """Reasoning tokens are included in completion_tokens; tracked
        separately because on a reasoning model they dominate agent latency."""
        return sum(t.reasoning_tokens or 0 for t in self.turns)


# --- §5.2 memory ------------------------------------------------------------


def _parse_footprint(output: str) -> int | None:
    match = _FOOTPRINT_HEADER.search(output)
    if not match:
        return None
    value, unit = match.group(1), match.group(2).upper()
    return int(float(value) * _UNITS.get(unit, 1))


def phys_footprint_bytes(pid: int) -> int | None:
    """`footprint -p <pid>` phys_footprint. No sudo required."""
    try:
        completed = subprocess.run(
            ["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _parse_footprint(completed.stdout)


def rss_bytes(pid: int) -> int | None:
    """Fallback when `footprint` is unavailable. Resident set size, in bytes.

    RSS is not phys_footprint and the two are not interchangeable, which is why
    the sampler records which method produced a figure.
    """
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = completed.stdout.strip()
    return int(text) * 1024 if text.isdigit() else None


_VM_STAT_KEYS = ("Pages wired down", "Pages active", "Pages occupied by compressor")


def system_used_bytes(page_size: int = 16384) -> int | None:
    """Memory actually committed system-wide: wired + active + compressed.

    This, not a per-process figure, is what §5.2 measures. MLX allocates its
    weights while llama.cpp memory-maps them, so `phys_footprint` credits one
    runtime with the model and not the other — a bias of roughly the whole model
    size, applied to the very comparison the benchmark exists to make. RSS
    inverts the same bias rather than removing it.
    """
    try:
        completed = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    total_pages = 0
    for key in _VM_STAT_KEYS:
        match = re.search(rf"{key}:\s+(\d+)", completed.stdout)
        if match is None:
            return None
        total_pages += int(match.group(1))
    return total_pages * page_size


class MemorySampler:
    """Samples system-wide committed memory every 250 ms, retaining the maximum.

    Reported as a delta against a baseline captured with no model loaded (§5.2),
    so the figure answers the question §2.2 actually cares about: how much
    unified memory this configuration needs. It does include other processes'
    churn, which is why §3.1's quiet-machine precondition is load-bearing.

    `pid` is optional and, when given, records a per-process footprint alongside
    the system figure for diagnostics only. It never enters a comparison.
    """

    def __init__(
        self,
        pid: int | None = None,
        interval_s: float = 0.25,
        baseline_bytes: int | None = None,
    ) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.baseline_bytes = baseline_bytes
        self.peak_bytes: int | None = None
        self.peak_process_bytes: int | None = None
        self.method: str = "vm_stat.wired+active+compressed"
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_delta_bytes(self) -> int | None:
        """Peak above baseline. The reported §5.2 figure."""
        if self.peak_bytes is None or self.baseline_bytes is None:
            return None
        return max(0, self.peak_bytes - self.baseline_bytes)

    def _sample_once(self) -> int | None:
        if self.pid is not None:
            process = phys_footprint_bytes(self.pid) or rss_bytes(self.pid)
            if process is not None and (
                self.peak_process_bytes is None or process > self.peak_process_bytes
            ):
                self.peak_process_bytes = process
        return system_used_bytes()

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._sample_once()
            if value is not None:
                self.samples += 1
                if self.peak_bytes is None or value > self.peak_bytes:
                    self.peak_bytes = value
            self._stop.wait(self.interval_s)

    def start(self) -> MemorySampler:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.peak_bytes

    def __enter__(self) -> MemorySampler:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def swap_used_bytes() -> int | None:
    """`sysctl vm.swapusage` used-bytes."""
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"used\s*=\s*([\d.]+)([KMGT]?)", completed.stdout)
    if not match:
        return None
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return int(float(match.group(1)) * scale.get(match.group(2).upper(), 1))


@dataclass
class SwapWindow:
    """Brackets a run to detect swapping, the likely performance cliff (§5.2)."""

    start_bytes: int | None = None
    end_bytes: int | None = None

    @property
    def delta_bytes(self) -> int | None:
        if self.start_bytes is None or self.end_bytes is None:
            return None
        return self.end_bytes - self.start_bytes

    @property
    def flagged(self) -> bool:
        delta = self.delta_bytes
        return delta is not None and delta > 0

    def __enter__(self) -> SwapWindow:
        self.start_bytes = swap_used_bytes()
        return self

    def __exit__(self, *exc) -> None:
        self.end_bytes = swap_used_bytes()


# --- process discovery (§5.2) ----------------------------------------------

# LM Studio runs its backends as separate child processes and the name differs
# between MLX and llama.cpp, so this is discovered, never hardcoded.
#
# `LM Studio Helper` is deliberately absent: the Electron renderer matches that
# name and is not the backend. Inference runs under `.lmstudio/.internal`.
PROCESS_HINTS = (
    ".lmstudio/.internal",
    "llama-server",
    "llama.cpp",
    "mlx_engine",
    "lms-",
)

# Diagnostics only since §5.2 moved to a system-wide measure, so no plausibility
# floor: llama.cpp mmaps its weights and legitimately reports a small footprint.
MIN_PLAUSIBLE_BYTES = 0


def find_inference_pid(
    hints: tuple[str, ...] = PROCESS_HINTS,
    min_bytes: int = MIN_PLAUSIBLE_BYTES,
) -> tuple[int, str] | None:
    """Return (pid, command) of the matching process with the largest footprint.

    Ranked by `phys_footprint`, not RSS. On unified memory a backend's RSS can
    badly understate what it holds, and ranking by RSS picks the UI process.
    Returns None when no candidate is large enough to plausibly hold a model,
    which is a better outcome than silently sampling the wrong process.

    **Call this after at least one inference has completed.** llama.cpp
    allocates lazily, so immediately after `lms load` the backend can still sit
    below `min_bytes` and be rejected — the guard against sampling the wrong
    process then rejects the right one.
    """
    try:
        completed = subprocess.run(
            ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    best: tuple[int, int, str] | None = None
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid_text, command = parts
        if not pid_text.isdigit():
            continue
        if not any(hint.lower() in command.lower() for hint in hints):
            continue

        pid = int(pid_text)
        footprint = phys_footprint_bytes(pid) or rss_bytes(pid) or 0
        if best is None or footprint > best[1]:
            best = (pid, footprint, command)

    if best is None or best[1] < min_bytes:
        return None
    return best[0], best[2]
