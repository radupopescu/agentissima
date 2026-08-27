# interview

A benchmark for local agent LLMs on Apple silicon, run through LM Studio.

It answers three separable questions — MLX or llama.cpp/Metal, LFM2.5-2.6B or
Ternary-Bonsai-8B, and which quantisation gives the best quality/latency/memory trade-off —
by measuring agent task success rather than token throughput alone.

| Document | Purpose |
|---|---|
| [`doc/benchmark.md`](doc/benchmark.md) | The specification. Authoritative on what is measured and how |
| [`doc/way-of-working.md`](doc/way-of-working.md) | Methodology and invariants for developing the project |
| [`doc/implementation-plan.md`](doc/implementation-plan.md) | Remaining work, in pickup-ready detail |

---

## Requirements

- macOS on Apple silicon
- Python 3.14 and [`uv`](https://docs.astral.sh/uv/)
- LM Studio, for the model-facing stages only

## Setup

```sh
uv venv --python 3.14
uv pip install pyyaml openai pytest
```

Run everything through `.venv/bin/python`. The sandbox puts `.venv/bin` on `PATH` for
commands it executes, so `pytest` and `python` resolve inside sandboxed runs.

---

## What you can run today

The whole measurement apparatus except the part that talks to a model. No LM Studio, no
network, no model downloads required.

### Generate the fixtures

```sh
.venv/bin/python fixtures/build_workspace.py    # workspace/ + expected/W*.json
.venv/bin/python fixtures/build_testrepo.py     # testrepo/  + expected/T*.json
```

Both are seeded and reproduce byte-for-byte. They are committed deliberately: results are only
comparable against a known fixture revision. **Regenerating a fixture bumps
`task_set_version`** and invalidates comparison with earlier results — see §11.

Each generator emits the fixture *and* the expected values the assertions read, so the two
cannot drift apart.

### Run the validation gates

```sh
.venv/bin/python -m harness.gates
```

These are blocking preconditions (§8). No model is benchmarked until all pass.

| Gate | Requirement | Rules out |
|---|---|---|
| oracle | 20/20 | Unsolvable tasks, unreachable information, broken assertions |
| negative control | 0/20 | Assertions that pass trivially |
| adversarial control | 0/20 | Decoys that do not actually discriminate |
| driver parity | pending | Needs the `pi` driver |

The oracle reaches every answer *through the same five tools an agent would use* — it never
reads the expected values. That is what makes 20/20 mean the information is genuinely
reachable, rather than merely present on disk.

If the oracle fails a task, the task or its assertion is wrong, not the model.

### Run the harness's own tests

```sh
.venv/bin/python -m pytest -q
```

These guard the properties the benchmark's validity rests on: tool calls are never repaired
(§4.5), output truncation is exact, and the sandbox cannot be escaped (§4.6).

---

### Smoke-check against a real model

Requires LM Studio running. Loads a model, runs one task through the `native` driver, unloads.

Models are named by **path**, not by LM Studio's model key: keys shift as models are installed
and removed (§2.1). `lms ls --json` lists the paths.

```sh
.venv/bin/python -m harness.smoke   # LiquidAI/LFM2.5-2.6B-MLX-8bit, task W01
.venv/bin/python -m harness.smoke LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf W05
```

An exact key still works, but anything ambiguous is refused rather than resolved to whichever
model happens to match first.

Not a benchmark stage — a sanity check after touching the client, the loop or the metrics.
`final_finish_reason` should be `stop` rather than `length`, and `peak memory` should sit a
little above the model's on-disk size — a much smaller figure means process discovery matched
a helper rather than the backend.

---

## What is not built yet

The configuration probes and environment capture (§2.1, §3), the Stage 0 tasks, the stage
runners and results output (§9, §10), reporting, and the `pi` driver (§4.1).

The LM Studio client, the `native` agent loop and the metrics layer are built and verified
end-to-end against a real model, but nothing yet orchestrates them into a stage.

Consequently **no benchmark stage can be run yet.** See
[`doc/implementation-plan.md`](doc/implementation-plan.md) for the ordered milestones and the
interfaces new code must fit.

---

## Running the benchmark, once built

Recorded here so the intended workflow is clear. None of this works today.

```sh
# One-off, per machine
.venv/bin/python -m setup.probe_process              # discover the inference process name
.venv/bin/python -m setup.probe_config  --config LFM-M8

# Per configuration
.venv/bin/python -m harness.stages stage0  --config LFM-M8
.venv/bin/python -m harness.stages stage1  --config LFM-M8 --context 8192 16384
.venv/bin/python -m harness.stages stage2a --config LFM-M8 --context 8192
.venv/bin/python -m harness.stages stage2b --config LFM-M8 --context 8192

# Reporting, regenerated from JSONL only
.venv/bin/python -m harness.report results/<session>/
```

Before any of it: LM Studio serving on `localhost:1234` with exactly one model loaded, on AC
power, Low Power Mode off. The harness asserts these and aborts rather than warning (§3.1).

Stage 0 is the cheap gate — three trivial tool calls per configuration. A configuration that
cannot emit a valid tool call is excluded from the agent stages there, before it consumes the
6–12 hours that Stage 2A takes.

---

## Layout

```
doc/
  benchmark.md            the specification
  way-of-working.md       methodology and invariants
  implementation-plan.md  remaining work
fixtures/
  build_workspace.py      generator for the non-coding fixture
  build_testrepo.py       generator for the coding fixture
  workspace/ testrepo/    generated; committed and version-pinned
  testrepo_variants/      per-task fixture variants (T09's test_close.py)
  expected/               generated expected values, read by the assertions
harness/
  sandbox.py              rooted tool implementations, truncation, command allowlist
  tools.py                tool schemas and strict no-repair dispatch
  prompt.py               the fixed system prompt
  tasks/                  Suite W and Suite T definitions and assertions
  assertions.py           shared assertion helpers
  scoring.py              the 0-4 progress score
  runner.py               fixture preparation and grading
  oracle.py               oracle, negative control, adversarial control
  gates.py                runs the §8 gates
  client.py               LM Studio streaming + §5.1 chunk timings
  driver_native.py        the agent loop and its termination rules
  metrics.py              timing, memory sampler, swap window
  lmstudio.py             model load/unload via the `lms` CLI
  smoke.py                one-task end-to-end check against a real model
tests/                    tests for the harness itself
```
