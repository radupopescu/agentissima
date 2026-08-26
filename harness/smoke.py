"""End-to-end smoke check against one real model.

    python -m harness.smoke [model-key] [task-id]

Not a benchmark stage. It loads a model, runs one task through the `native`
driver, and prints the §5 metrics, so that a change to the client, the loop or
the metrics can be sanity-checked against a real backend before committing to a
multi-hour stage.

What to look at: `peak_memory` should be large enough to plausibly hold the
model — a small figure means process discovery matched a helper — and
`final_finish_reason` should be `stop`, since `length` means the answer was cut
off by `max_tokens` rather than finished.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import lmstudio
from .client import LMStudioClient
from .driver_native import NativeDriver
from .metrics import MemorySampler, SwapWindow, find_inference_pid
from .runner import run_task
from .tasks import BY_ID

IDENTIFIER = "bench"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.smoke")
    parser.add_argument("model_key", nargs="?", default="lfm2.5-2.6b-mlx")
    parser.add_argument("task_id", nargs="?", default="W01")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--overhead-samples", type=int, default=8)
    args = parser.parse_args(argv)

    if args.task_id not in BY_ID:
        print(f"unknown task: {args.task_id}", file=sys.stderr)
        return 2

    print(f"loading {args.model_key} at {args.context} ...", flush=True)
    started = time.monotonic()

    with lmstudio.loaded(
        args.model_key, context_length=args.context, identifier=IDENTIFIER
    ) as model:
        print(f"  loaded in {time.monotonic() - started:.1f}s  ({model.model_key})")

        found = find_inference_pid()
        if found is None:
            print("  WARNING: no inference process found; peak memory unavailable")
        else:
            print(f"  inference pid {found[0]}")

        client = LMStudioClient(model=IDENTIFIER)

        print(f"calibrating overhead ({args.overhead_samples} samples) ...", flush=True)
        overhead = client.measure_overhead(samples=args.overhead_samples)
        print(f"  overhead_median {overhead * 1000:.1f} ms")

        sampler = MemorySampler(found[0]).start() if found else None
        driver = NativeDriver(client=client, overhead_s=overhead)

        print(f"running {args.task_id} ...", flush=True)
        with SwapWindow() as swap:
            result = run_task(BY_ID[args.task_id], driver)
        peak = sampler.stop() if sampler else None

    print()
    print(f"  task           {result.task_id}  ({result.suite})")
    print(f"  passed         {result.passed}")
    print(f"  progress       {result.progress}/4")
    print(f"  termination    {result.termination_reason}")
    print(f"  steps          {result.steps}")
    print(f"  tool calls     {result.tool_calls} ({result.invalid_calls} invalid, "
          f"{result.path_errors} path errors)")
    print(f"  wall clock     {result.wall_clock_s:.1f}s")
    if peak:
        print(f"  peak memory    {peak / 1024**3:.2f} GiB via {sampler.method}")
    print(f"  swap delta     {swap.delta_bytes} (flagged={swap.flagged})")
    print()
    print("  metrics " + json.dumps(result.metrics or {}, indent=2, default=str))
    print()
    print(f"  answer: {result.answer[:500]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
