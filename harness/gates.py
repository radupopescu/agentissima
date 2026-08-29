"""Run the validation gates from doc/benchmark.md §8.

    python -m harness.gates

The oracle must score 20/20 and the negative control 0/20. Both are blocking:
no model is benchmarked until they pass.
"""

from __future__ import annotations

import sys

from .oracle import DECOY_TASKS, decoy_driver, oracle_driver, stub_driver
from .runner import run_task
from .tasks import ALL_TASKS
from .tasks.smoke import STAGE0_TASKS


def _run(driver, label: str, tasks: list = ALL_TASKS) -> tuple[int, list[str]]:
    print(f"\n=== {label} ===")
    print(f"{'task':<6}{'pass':<6}{'prog':<6}{'calls':<7}{'invalid':<9}{'secs':<7}reason")

    passed = 0
    failures = []
    for task in tasks:
        try:
            result = run_task(task, driver)
        except Exception as exc:  # a broken solver or assertion, not a model failure
            failures.append(f"{task.id}: {type(exc).__name__}: {exc}")
            print(f"{task.id:<6}{'ERR':<6}{'-':<6}{'-':<7}{'-':<9}{'-':<7}{type(exc).__name__}")
            continue

        passed += int(result.passed)
        if not result.passed:
            failures.append(f"{task.id}: answer={result.answer[:120]!r}")
        print(
            f"{result.task_id:<6}"
            f"{('yes' if result.passed else 'no'):<6}"
            f"{result.progress:<6}"
            f"{result.tool_calls:<7}"
            f"{result.invalid_calls:<9}"
            f"{result.wall_clock_s:<7.2f}"
            f"{result.termination_reason}"
        )

    print(f"{label}: {passed}/{len(tasks)}")
    return passed, failures


def main() -> int:
    total = len(ALL_TASKS)

    oracle_passed, oracle_failures = _run(oracle_driver, "oracle (must score 20/20)")
    stub_passed, _ = _run(stub_driver, "negative control (must score 0/20)")
    decoy_passed, _ = _run(
        decoy_driver, f"adversarial control (must score 0/20; decoys: {', '.join(DECOY_TASKS)})"
    )

    stage0_total = len(STAGE0_TASKS)
    stage0_oracle_passed, stage0_oracle_failures = _run(
        oracle_driver, "stage 0 oracle (must score 3/3)", STAGE0_TASKS
    )
    stage0_stub_passed, _ = _run(
        stub_driver, "stage 0 negative control (must score 0/3)", STAGE0_TASKS
    )

    print("\n=== gates ===")
    oracle_ok = oracle_passed == total
    stub_ok = stub_passed == 0
    decoy_ok = decoy_passed == 0
    stage0_oracle_ok = stage0_oracle_passed == stage0_total
    stage0_stub_ok = stage0_stub_passed == 0
    print(f"oracle              {oracle_passed}/{total}   {'PASS' if oracle_ok else 'FAIL'}")
    print(f"negative control    {stub_passed}/{total}   {'PASS' if stub_ok else 'FAIL'}")
    print(f"adversarial control {decoy_passed}/{total}   {'PASS' if decoy_ok else 'FAIL'}")
    print(
        f"stage 0 oracle      {stage0_oracle_passed}/{stage0_total}   "
        f"{'PASS' if stage0_oracle_ok else 'FAIL'}"
    )
    print(
        f"stage 0 negative    {stage0_stub_passed}/{stage0_total}   "
        f"{'PASS' if stage0_stub_ok else 'FAIL'}"
    )
    print("driver parity       pending: requires the pi driver (§8)")

    if oracle_failures:
        print("\noracle failures:")
        for line in oracle_failures:
            print(f"  {line}")
    if stage0_oracle_failures:
        print("\nstage 0 oracle failures:")
        for line in stage0_oracle_failures:
            print(f"  {line}")

    if not (oracle_ok and stub_ok and decoy_ok and stage0_oracle_ok and stage0_stub_ok):
        print("\nGates failed. Per §8 the task or assertion is wrong, not the model.")
        return 1

    print("\nGates passed. The task suites are solvable and non-trivial.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
