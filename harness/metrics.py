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
# The TOTAL row of `footprint -p`: Dirty | Clean | Reclaimable | Regions | TOTAL
_FOOTPRINT_TOTAL = re.compile(
    r"^\s*([\d.]+)\s*([KMGT]?B)\s+([\d.]+)\s*([KMGT]?B)\s+[\d.]+\s*[KMGT]?B\s+\d+\s+TOTAL\s*$",
    re.IGNORECASE | re.MULTILINE,
)
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


def _parse_total_row(output: str) -> int | None:
    """Dirty + Clean from the TOTAL row of `footprint -p`, in bytes.

    The table is `Dirty | Clean | Reclaimable | Regions | Category`. Reclaimable
    is deliberately excluded: it is a subset already counted, not a fourth
    quantity.
    """
    match = _FOOTPRINT_TOTAL.search(output)
    if not match:
        return None
    dirty = float(match.group(1)) * _UNITS.get(match.group(2).upper(), 1)
    clean = float(match.group(3)) * _UNITS.get(match.group(4).upper(), 1)
    return int(dirty + clean)


def resident_bytes(pid: int) -> int | None:
    """Memory the process holds resident, counted so that MLX and llama.cpp are
    comparable (§5.2).

    Neither of the obvious metrics works, because the two runtimes put the
    weights in different classes of memory:

    - llama.cpp `mmap`s the GGUF, so the weights are **clean, file-backed**
      pages. `phys_footprint` counts dirty pages and therefore excludes them:
      227 MB reported for a 2.87 GB artefact.
    - MLX allocates Metal buffers, so the weights are **dirty
      IOAccelerator (graphics)** pages. Those are owned by the GPU and are not
      in RSS: 707 MB reported for a 2.88 GB artefact.

    Each metric is blind to exactly one runtime, in the comparison that is
    question 1 of §1. Summing the Dirty and Clean columns of `footprint`'s TOTAL
    row counts both, giving 2980 MB and 3310 MB for those same two artefacts.

    This is an upper bound on what must stay resident: clean file-backed pages
    are evictable under pressure. That is the right bound for §2.2, which asks
    how much unified memory a configuration needs.
    """
    try:
        completed = subprocess.run(
            ["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _parse_total_row(completed.stdout)


def phys_footprint_bytes(pid: int) -> int | None:
    """`footprint -p <pid>` phys_footprint. Diagnostics only.

    **Not a §5.2 measure and not comparable across runtimes**: it counts dirty
    pages, so it excludes the clean mapped-file pages llama.cpp holds its
    weights in. Use `resident_bytes`. Kept because the discrepancy between the
    two is itself worth being able to inspect.
    """
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


class MemorySampler:
    """Samples the inference process every 250 ms and retains the maximum (§5.2).

    The measure is `resident_bytes` — dirty + clean — which is the only one of
    the available per-process figures that counts the weights under both
    runtimes. `footprint` costs ~50 ms, so it fits the sampling interval;
    `vmmap -summary` gives the same answer but takes ~1 s and does not.

    A previous revision measured a system-wide `vm_stat` delta to escape the
    per-process bias. It escaped the bias but was too noisy to use: six runs of
    one configuration spanned 1.54-2.82 GiB, against a spread of under 0.05 GiB
    here.
    """

    def __init__(self, pid: int, interval_s: float = 0.25) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.peak_bytes: int | None = None
        self.method: str = "footprint.dirty+clean"
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> int | None:
        return resident_bytes(self.pid)

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

# A model must account for at least this much, or we have matched a helper
# rather than the process holding the weights. Meaningful again now that
# `resident_bytes` counts the weights under both runtimes: under the previous
# system-wide measure no per-process floor could be applied, because llama.cpp
# legitimately reported ~0.2 GiB.
MIN_PLAUSIBLE_BYTES = 512 * 1024**2


def find_inference_pid(
    hints: tuple[str, ...] = PROCESS_HINTS,
    min_bytes: int = MIN_PLAUSIBLE_BYTES,
) -> tuple[int, str] | None:
    """Return (pid, command) of the matching process holding the most memory.

    Ranked by `resident_bytes`, the same measure the sampler uses, so ranking
    cannot prefer a process that merely scores well on a metric blind to the
    backend in play: `phys_footprint` ranks a llama.cpp backend at 0.2 GiB and
    RSS ranks an MLX backend at 0.7 GiB, either of which loses to a helper.

    Returns None when no candidate is large enough to plausibly hold a model,
    which is a better outcome than silently sampling the wrong process.

    **Call this after at least one inference has completed**, so that lazily
    allocated memory is in place and a backend is not rejected for sitting below
    `min_bytes` at the moment it was probed.
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
        resident = resident_bytes(pid) or rss_bytes(pid) or 0
        if best is None or resident > best[1]:
            best = (pid, resident, command)

    if best is None or best[1] < min_bytes:
        return None
    return best[0], best[2]
