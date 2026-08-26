# local-llm-benchmark

An agent benchmark for local LLMs on an Apple M1 Pro (16 GB), run through LM Studio.

[`benchmark.md`](benchmark.md) is the specification and the authoritative description of the
protocol. This README covers only how to run what exists.

## Setup

```sh
uv venv --python 3.14
uv pip install pyyaml openai pytest
```

## Generate the fixtures

Both generators emit the fixture **and** the expected values the assertions read, so a fixture
and its assertions cannot drift apart (§6).

```sh
.venv/bin/python fixtures/build_workspace.py    # workspace/ + expected/W*.json
.venv/bin/python fixtures/build_testrepo.py     # testrepo/  + expected/T*.json
```

Regenerating a fixture bumps `task_set_version` and invalidates comparison with earlier
results (§11).

## Run the validation gates

Blocking preconditions from §8. No model is benchmarked until all pass.

```sh
.venv/bin/python -m harness.gates
```

- **oracle** must score 20/20 — proves every task is solvable through the tool surface
- **negative control** must score 0/20 — proves no assertion passes trivially
- **adversarial control** must score 0/20 — proves the planted decoys discriminate
- **driver parity** — pending, requires the `pi` driver

## Run the harness's own tests

```sh
.venv/bin/python -m pytest -q
```

These guard the contracts the benchmark's validity rests on: tool calls are never repaired
(§4.5), output truncation is exact, and the sandbox cannot be escaped (§4.6).

## Layout

```
benchmark.md              the specification
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
tests/                    tests for the harness itself
```

## Not yet built

The LM Studio client and the `native` agent loop (§4.2), the metrics layer (§5), the
configuration probes (§2.1), and the stage runners (§9).
