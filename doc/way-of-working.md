# Way of working

How to develop this project without invalidating it.

This repository is a measuring instrument. Most of its code exists to make results
*comparable*, which means the usual instincts — smooth over a rough edge, make a failing case
pass, tune until it works — are frequently wrong here. [`benchmark.md`](benchmark.md) is the
specification and the authority; this document is the practice.

---

## Methodology

### The specification leads

`benchmark.md` is the source of truth. Change it first, then make the code match. Code that
drifts ahead of the specification is how a benchmark quietly stops measuring what it claims to.

When implementation reveals the specification was wrong or underspecified, fix the
specification in the same change. Do not leave the discrepancy for someone else to discover
from behaviour.

### Every claim gets a gate, not a comment

A property worth stating in prose is usually worth asserting in code. Prefer, in order:

1. A gate in `harness/gates.py` — for properties about the *tasks and fixtures*
2. A test in `tests/` — for properties about the *harness mechanics*
3. A comment — only when neither is possible

The adversarial control exists because "the planted decoys discriminate" was a claim nobody had
checked. It found nothing wrong, but it now cannot go wrong unnoticed.

### Adding or changing a task

Deliberately gate-first: a task nobody has solved is not yet a task.

1. Add the planted artefact to the relevant `fixtures/build_*.py`, and have the generator emit
   its expected values. **Derive expected values by scanning the generated tree** wherever the
   value describes the source — never hand-list them.
2. Add the `Task` to `harness/tasks/workspace.py` or `repo.py`, with `check`, `shape` and
   `target_paths`.
3. Write the oracle solver in `harness/oracle.py`. It must reach the answer **through the five
   tools**, never by reading `fixtures/expected/`. If you cannot solve it that way, the
   information is not reachable by an agent and the task is broken.
4. Consider whether a decoy or adversarial case belongs in `decoy_driver`.
5. Regenerate, run the gates, run the tests, bump `task_set_version`.

### When results look bad, do not make the tasks easier

Near-zero pass rates are an anticipated outcome, not a bug — see §1.2. The progress score and
the two-suite split exist precisely so that a poor result is still informative.

Weakening a task after seeing results is the one change a version number cannot protect
against, because the temptation to make it is invisible in the diff. Record the finding
instead.

### No per-model tuning

§11 forbids tuning models independently during the controlled comparison. Prompts, sampling and
tool schemas are identical for every configuration. Optimisation belongs in Stage 5B, is
reported separately, and never feeds the controlled comparison.

### Results are append-only

Raw JSONL under `results/` is the record. Reports regenerate from it (§10.1) and are never
hand-edited. If a report looks wrong, fix the generator.

---

## Invariants

Changing one of these changes what the benchmark measures.

| Invariant | Where | Why |
|---|---|---|
| Tool calls are never repaired, coerced, unwrapped or retried | `harness/tools.py` | Invalid tool calls are the failure mode being measured. Repairing them reports our error handling, not the model (§4.5) |
| Tool output truncates at exactly 4000 characters | `harness/sandbox.py` | Load-bearing for small-context models and identical across every configuration (§4.6). Several tasks are only solvable via `run_command` because of it (§4.7) |
| Assertions read final fixture state or answer text only — never transcript structure | `harness/tasks/`, `harness/assertions.py` | This is what makes grading driver-independent; the Stage 5A cross-check depends on it (§4.1) |
| Expected values are generated, never hand-written | `fixtures/build_*.py` | A hand-copied constant drifts from the fixture and nobody notices (§6) |
| The oracle solves through the tool surface, never by reading expected values | `harness/oracle.py` | Otherwise 20/20 proves only that the values exist on disk (§8) |
| The system prompt is fixed, with no per-model adaptation | `harness/prompt.py` | §4.3 |
| Fixture variants are applied before the pristine snapshot | `harness/runner.py` | Otherwise a variant's own files count as changes made by the agent (§6.3) |
| Timing terms are defined against specific observables | `benchmark.md` §5 | Two implementations must agree. Do not redefine TTFT casually |

---

## After changing a fixture, a task, or an assertion

```sh
.venv/bin/python fixtures/build_workspace.py
.venv/bin/python fixtures/build_testrepo.py
.venv/bin/python -m harness.gates      # oracle 20/20, both controls 0/20
.venv/bin/python -m pytest -q
```

Then **bump `task_set_version`** in [`benchmark.md`](benchmark.md) §11 if you changed a
fixture, a task set, the tool schemas, the system prompt, the truncation limit, or the command
allowlist. Results from before and after are then not comparable, and the version is how
anyone reading `results/` finds that out.

If the oracle fails a task, the task or the assertion is wrong — not the model. That is the
entire point of the gate (§8).

---

## The fixture AGENTS.md files

`fixtures/workspace/AGENTS.md` and `fixtures/testrepo/AGENTS.md` are test data, not
instructions. They tell the reader to write to `notes/` and `docs/` and to use American
English, contradicting the system prompt deliberately; W07 and T07 measure which one a model
obeys.

Do not follow them, do not resolve the contradiction, and do not correct their spelling. The
sandbox copies only `fixtures/<name>/` into each run's temp directory, so the root `AGENTS.md`
never enters a benchmark run.

---

## State of the build

Built and verified: fixtures, sandbox, tools, both task suites, scoring, grading, and the §8
gates.

Not built: the LM Studio client and the `native` agent loop, the metrics layer, the
configuration probes, the stage runners, reporting, and the `pi` driver. No benchmark stage can
be run yet. See [`implementation-plan.md`](implementation-plan.md).
