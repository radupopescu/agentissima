# agentissima

A benchmark for local agent LLMs on Apple silicon, run through LM Studio. It measures agent
task success, not token throughput alone, and answers three separable questions:

1. **Runtime** — MLX or llama.cpp/Metal, at equivalent quantisation
2. **Model** — LFM2.5-2.6B or Ternary-Bonsai-8B, on agent task success
3. **Operating point** — which quantisation gives the best quality/latency/memory trade-off

| Document | Purpose |
|---|---|
| [`doc/benchmark.md`](doc/benchmark.md) | The specification. Authoritative on what is measured and how |
| [`doc/way-of-working.md`](doc/way-of-working.md) | Methodology and invariants for developing the project |
| [`doc/implementation-plan.md`](doc/implementation-plan.md) | Remaining work, in pickup-ready detail |
| [`doc/findings.md`](doc/findings.md) | Empirical findings from real runs |

## Requirements

- macOS on Apple silicon
- Python 3.14 and [`uv`](https://docs.astral.sh/uv/)
- LM Studio, for the model-facing stages only

```sh
uv venv --python 3.14
uv pip install pyyaml openai pytest
```

Run everything through `.venv/bin/python`. The sandbox puts `.venv/bin` on `PATH` for commands
it executes, so `pytest` and `python` resolve inside sandboxed runs.

## Run without a model

No LM Studio, no network, no model downloads required.

**Generate the fixtures** — seeded, byte-for-byte reproducible, committed, and version-pinned.
Each generator emits the fixture *and* the expected values the assertions read, so the two
cannot drift apart.

```sh
.venv/bin/python fixtures/build_workspace.py    # workspace/ + expected/W*.json
.venv/bin/python fixtures/build_testrepo.py     # testrepo/  + expected/T*.json
```

Regenerating a fixture bumps `task_set_version` and invalidates comparison with earlier
results (`benchmark.md` §11).

**Run the validation gates** — blocking preconditions (§8). No model is benchmarked until all
pass.

```sh
.venv/bin/python -m harness.gates
```

| Gate | Requirement | Rules out |
|---|---|---|
| oracle | 20/20 | Unsolvable tasks, unreachable information, broken assertions |
| negative control | 0/20 | Assertions that pass trivially |
| adversarial control | 0/20 | Decoys that do not actually discriminate |
| driver parity | pending | Needs the `pi` driver |

The oracle reaches every answer *through the same five tools an agent would use* — it never
reads the expected values. 20/20 therefore means the information is genuinely reachable by an
agent, not merely present on disk. If the oracle fails a task, the task or its assertion is
wrong, not the model.

**Run the harness's own tests** — these guard the properties the benchmark's validity rests on:
tool calls are never repaired (§4.5), output truncation is exact, and the sandbox cannot be
escaped (§4.6).

```sh
.venv/bin/python -m pytest -q
```

## Run against a model

Requires LM Studio running. Smoke-check first: load a model, run one task through the `native`
driver, unload.

Models are addressed by **path**, never by LM Studio's model key (§2.1). `lms ls --json` lists
the paths.

```sh
.venv/bin/python -m harness.smoke   # LiquidAI/LFM2.5-2.6B-MLX-8bit, task W01
.venv/bin/python -m harness.smoke LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf W05
```

Sanity checks: `final_finish_reason` should be `stop` rather than `length`, and `peak memory`
should sit slightly above the model's on-disk size. A much smaller figure means process
discovery matched a helper rather than the backend.

**Stages** — one-off configuration probes, then per-configuration stages and reporting:

```sh
# Once per machine. Metadata only, no model loaded (§2.1).
.venv/bin/python -m setup.probe_config                # every §2 configuration
.venv/bin/python -m setup.probe_config --only LFM-M8  # a single one
.venv/bin/python -m setup.probe_config --hash         # also pin artefact bytes (§2.1)

# Per configuration
.venv/bin/python -m harness.stages stage0  --config LFM-M8
.venv/bin/python -m harness.stages stage1  --config LFM-M8 --context 8192 16384
.venv/bin/python -m harness.stages stage2a --config LFM-M8 --context 8192
.venv/bin/python -m harness.stages stage2b --config LFM-M8 --context 8192

# Reporting, regenerated from JSONL only
.venv/bin/python -m harness.report results/<session>/
```

Before any stage: LM Studio serving on `localhost:1234` with exactly one model loaded, on AC
power, Low Power Mode off. The harness asserts these and aborts rather than warning (§3.1).

Stage 0 is the cheap gate — three trivial tool calls per configuration. A configuration that
cannot emit a valid tool call is excluded from the agent stages there, before it consumes the
6–12 hours Stage 2A takes.

## Status

Built and verified against a real model: the LM Studio client, the `native` agent loop, the
metrics layer, Stage 0 and Stage 1 for all six configurations, and reporting against the data
those produced. The Stage 2A/2B/3 runners and `run_full()` are implemented and unit-tested but
not yet run live. Not built: the `pi` driver and Stage 5B's recommended-default sampling pass.

See [`doc/implementation-plan.md`](doc/implementation-plan.md) for the ordered milestones.

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
  stages.py               stage runner: stage0–stage3, run_full(), Stage 5B compaction
  report.py               tables regenerated from raw JSONL
  environment.py          environment capture (§3)
  smoke.py                one-task end-to-end check against a real model
setup/
  probe_config.py         per-configuration metadata and artefact hashing (§2.1)
tests/                    tests for the harness itself
```
