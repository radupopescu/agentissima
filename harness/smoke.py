"""End-to-end smoke check against one real model.

    python -m harness.smoke [model-key] [task-id]

Not a benchmark stage. It loads a model, runs one task through the `native`
driver, and prints the §5 metrics, so that a change to the client, the loop or
the metrics can be sanity-checked against a real backend before committing to a
multi-hour stage.

What to look at: `final_finish_reason` should be `stop`, since `length` means the
answer was cut off by `max_tokens` rather than finished. `peak memory` is a
system-wide delta above a no-model baseline, but its definition is unresolved
(§5.2) and it varies by over a GiB between repeat runs of one configuration —
treat it as an indication that a model was loaded, not as a measurement.

Models are named by path (`LiquidAI/LFM2.5-2.6B-MLX-8bit`), not by LM Studio's
model key, which moves as models are installed and removed (§2.1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import lmstudio
from .client import LMStudioClient
from .driver_native import NativeDriver
from .metrics import (
    MemorySampler,
    SwapWindow,
    find_inference_pid,
    system_used_bytes,
)
from .runner import run_task
from .tasks import BY_ID

IDENTIFIER = "bench"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.smoke")
    parser.add_argument(
        "model",
        nargs="?",
        default="LiquidAI/LFM2.5-2.6B-MLX-8bit",
        help="model path, e.g. LiquidAI/LFM2.5-2.6B-MLX-8bit. An exact `lms` "
        "key is accepted too, but keys move as models are installed (§2.1)",
    )
    parser.add_argument("task_id", nargs="?", default="W01")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--overhead-samples", type=int, default=8)
    args = parser.parse_args(argv)

    if args.task_id not in BY_ID:
        print(f"unknown task: {args.task_id}", file=sys.stderr)
        return 2

    # Baseline must be captured with no model resident (§5.2).
    lmstudio.unload_all()
    baseline = system_used_bytes()
    print(f"baseline system memory {(baseline or 0) / 1024**3:.2f} GiB")

    print(f"loading {args.model} at {args.context} ...", flush=True)
    started = time.monotonic()

    with lmstudio.loaded(
        args.model, context_length=args.context, identifier=IDENTIFIER
    ) as model:
        print(
            f"  loaded in {time.monotonic() - started:.1f}s  "
            f"({model.path}, key {model.model_key})"
        )

        client = LMStudioClient(model=IDENTIFIER)

        print(f"calibrating overhead ({args.overhead_samples} samples) ...", flush=True)
        overhead = client.measure_overhead(samples=args.overhead_samples)
        print(f"  overhead_median {overhead * 1000:.1f} ms")

        # After warm-up: llama.cpp allocates lazily and is not yet identifiable
        # immediately after load.
        found = find_inference_pid()
        if found is None:
            print("  WARNING: no inference process found; peak memory unavailable")
        else:
            print(f"  inference pid {found[0]}")

        sampler = MemorySampler(
            pid=found[0] if found else None, baseline_bytes=baseline
        ).start()
        driver = NativeDriver(client=client, overhead_s=overhead)

        print(f"running {args.task_id} ...", flush=True)
        with SwapWindow() as swap:
            result = run_task(BY_ID[args.task_id], driver)
        sampler.stop()
        peak = sampler.peak_delta_bytes

    print()
    print(f"  task           {result.task_id}  ({result.suite})")
    print(f"  passed         {result.passed}")
    print(f"  progress       {result.progress}/4")
    print(f"  termination    {result.termination_reason}")
    print(f"  steps          {result.steps}")
    print(f"  tool calls     {result.tool_calls} ({result.invalid_calls} invalid, "
          f"{result.path_errors} path errors)")
    print(f"  wall clock     {result.wall_clock_s:.1f}s")
    if peak is not None:
        print(f"  peak memory    {peak / 1024**3:.2f} GiB above baseline "
              f"via {sampler.method}")
        if sampler.peak_process_bytes:
            print(f"  (process       {sampler.peak_process_bytes / 1024**3:.2f} GiB "
                  f"- diagnostics only, not comparable across runtimes)")
    print(f"  swap delta     {swap.delta_bytes} (flagged={swap.flagged})")
    print()
    print("  metrics " + json.dumps(result.metrics or {}, indent=2, default=str))
    print()
    print(f"  answer: {result.answer[:500]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
