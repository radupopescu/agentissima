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

### Commit messages

Write a concise summary of what changed. Do not add a contributor sign-off, `Co-authored-by`
trailer, or tool attribution.

---

## Invariants

Changing one of these changes what the benchmark measures.

| Invariant | Where | Why |
|---|---|---|
| Tool calls are never repaired, coerced, unwrapped or retried | `harness/tools.py` | Invalid tool calls are the failure mode being measured. Repairing them reports our error handling, not the model (§4.5) |
| Tool output truncates at exactly 4000 characters | `harness/sandbox.py` | The limit decides what a small-context model can see, and is identical across every configuration (§4.6). Several tasks are only solvable via `run_command` because of it (§4.7) |
| Assertions read final fixture state or answer text only — never transcript structure | `harness/tasks/`, `harness/assertions.py` | This is what makes grading driver-independent. Both drivers depend on it, and the §8 driver-parity gate is what proves it rather than asserting it (§4.1) |
| Expected values are generated, never hand-written | `fixtures/build_*.py` | A hand-copied constant drifts from the fixture and nobody notices (§6) |
| The oracle solves through the tool surface, never by reading expected values | `harness/oracle.py` | Otherwise 20/20 proves only that the values exist on disk (§8) |
| The system prompt is fixed, with no per-model adaptation | `harness/prompt.py` | §4.3 |
| Fixture variants are applied before the change baseline is taken | `harness/runner.py` | Otherwise a variant's own files count as changes made by the agent (§6.3) |
| The work directory around a run's fixture copy is sealed while the run is in progress | `harness/runner.py` | The container mounts the whole runs root as this uid, so an unsealed parent lets a misdirected write succeed silently — and grading, which reads `root/`, never sees it (§4.6) |
| The change baseline is hashes, never a copy of the tree inside the run directory | `harness/runner.py`, `harness/types.py` | The runs root is mounted into the container, so a reference copy is readable — and writable — by the agent it exists to judge (§6.3) |
| Timing terms are defined against specific observables | `benchmark.md` §5 | Two implementations must agree. Do not redefine TTFT casually |

---

## After changing a fixture, a task, an assertion, or a driver

```sh
.venv/bin/python fixtures/build_workspace.py
.venv/bin/python fixtures/build_testrepo.py
.venv/bin/python -m harness.gates      # oracle 20/20, both controls 0/20, driver parity 20/20
.venv/bin/python -m pytest -q
```

Then check [`benchmark.md`](benchmark.md) §11's list — it is authoritative on what bumps
`task_set_version`, and it includes a driver version, which is easy to overlook. If it applies,
bump `TASK_SET_VERSION` in `harness/version.py`, add a row to §11's version table saying what
changed and why it is not comparable, and move the superseded raw files to
`results/<session>/archive/<stage>-<old version>.jsonl`. Resuming into a session written under
a different version raises `SessionMismatchError` rather than pooling silently, so skipping the
archive step fails loudly rather than quietly — but it fails at the start of a long run.

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

§7.3 sets out the full design of that pair — including why the same file reaches the model
differently under each driver, which is what makes W07 and T07 non-comparable across drivers.

---

## Working on the `pi` driver

`pi` is the controlled comparison for Stages 2A–4 as of `v5` (§4.1); `native` runs Stage 0 and
Stage 1 and is the Stage 5A cross-check. Two rules follow, and both are easy to break by
accident:

- **Do not freeze pi's own behaviour.** `--system-prompt`, `--tools`, `--exclude-tools` and
  `--thinking` are deliberately absent from the invocation: pinning them would measure "pi as
  configured in August 2026", which decays as pi improves and defeats the reason for using a
  production harness at all. `tests/test_driver_pi.py` asserts their absence, so adding one
  fails a test rather than passing quietly. Read §4.1 before changing that.
- **Do not treat pi's zeroes as measurements.** `invalid_calls` is always 0 under `pi` because
  pi repairs internally and a malformed call never reaches its log — "not observed", not "none
  happened" (§5.3). Anything that needs §4.5's no-repair accounting has to run under `native`.

Adding a flag to the invocation, changing the Seatbelt profile, or changing the containment
story bumps `PiDriver.DRIVER_VERSION` — and §11 makes a driver version a `task_set_version`
trigger, so it invalidates previously collected results. That is the intended cost; budget for
re-collection before making the change, not after.

---

## State of the build

[`implementation-plan.md`](implementation-plan.md) §1 is authoritative and is not repeated
here. In outline: the harness is complete and all four §8 gates pass; Stage 3, Stage 5B and the
first `pi`-primary campaign have not been run live; and the `v4` agent data is due for
re-collection under `v5`.
