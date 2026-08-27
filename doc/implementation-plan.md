# Implementation plan

Remaining work to make [`benchmark.md`](benchmark.md) executable. Written so the work can be
picked up cold in another session.

Section references (§4.2, §5.1 …) point at `benchmark.md`, which is the authoritative
specification. Where this plan and the specification disagree, the specification wins — fix
this file.

---

## 1. Current state

### Built and verified

| Area | Files | Status |
|---|---|---|
| Fixtures + expected values | `fixtures/build_workspace.py`, `build_testrepo.py` | Both generate; byte-reproducible from seed |
| Sandbox and tools | `harness/sandbox.py`, `harness/tools.py` | 20 tests passing |
| Task suites | `harness/tasks/workspace.py`, `repo.py`, `__init__.py` | 20 tasks with programmatic assertions |
| Grading and scoring | `harness/runner.py`, `harness/scoring.py`, `harness/assertions.py` | Working |
| Validation gates | `harness/oracle.py`, `harness/gates.py` | Oracle 20/20, both controls 0/20 |
| System prompt | `harness/prompt.py` | Defined and hashed; wired into the `native` driver |
| LM Studio client | `harness/client.py` | Streaming + §5.1 chunk timings; 9 tests |
| `native` agent loop | `harness/driver_native.py` | All five §4.8 termination paths tested |
| Metrics | `harness/metrics.py` | §5.1 timing, memory sampler, swap window, process discovery |
| Model lifecycle | `harness/lmstudio.py` | `lms` load/unload/ps, stage-scoped context manager |

The harness half runs with no model and no network (43 tests):

```sh
.venv/bin/python -m harness.gates
.venv/bin/python -m pytest -q
```

**Verified end-to-end** at 8K: W01 passes, 5 valid tool calls, 0 invalid. Re-check with
`python -m harness.smoke [model-path] [task-id]` after touching the client, loop or metrics.

> The original note recorded this against model key `lfm2.5-2.6b-mlx`, which no longer resolves
> — two MLX builds are now installed, so the key gained an `@<quant>` suffix. Which artefact
> produced its timings cannot now be established, so they have been dropped. This is the §2.1
> identifier defect appearing in the project's own records, and the reason paths replaced keys.

Four defects that only a real model exposed, now fixed and covered by tests:

| Defect | Fix |
|---|---|
| LFM2.5 emits `reasoning_content` before any `content`; 60 % of its generated tokens were reasoning. TTFT measured from first *content* folded the whole reasoning phase into prefill and inflated gen tok/s | `t_first` counts reasoning; `reasoning_tokens` recorded separately (§5.1) |
| Process discovery matched the Electron renderer (0.5 GiB) instead of the backend (6.5 GiB) | Hints target `.lmstudio/.internal`; candidates ranked by footprint, not RSS; refuses a process too small to hold a model (§5.2) |
| `max_tokens=1` made LM Studio finish without emitting a token delta, so overhead calibration silently returned 0.0 and prompt tok/s went unadjusted | `max_tokens=8`; no timed sample now fails loudly (§5.1) |
| W01 failed a *correct* answer that named £85 and identified £72 as superseded | The decoy may appear if marked superseded (§7.1) |

### Not built

The configuration probes, environment capture, the Stage 0 tasks, the stage runners, results
output, reporting, and the `pi` driver.

---

## 2. Interfaces already fixed

New code must fit these. They are load-bearing and were chosen deliberately.

```python
# harness/runner.py
Driver = Callable[[Task, Sandbox], RunOutcome]
run_task(task: Task, driver: Driver) -> Graded

# harness/types.py
RunOutcome(task_id, root, answer, calls, termination_reason, steps, path_errors)
Task(id, suite, category, fixture, prompt, min_context, target_paths,
     check, shape, extra_rules, variant)

# harness/tools.py
dispatch(sandbox: Sandbox, name: str, raw_arguments: str) -> ToolCall
TOOL_SCHEMAS   # send verbatim; do not rebuild per driver

# harness/prompt.py
assemble(extra_rules: str | None) -> str
prompt_sha256(extra_rules: str | None) -> str
```

`run_task` prepares a fresh fixture copy, applies any variant, snapshots the pristine tree,
runs the driver, and grades. A new driver is a drop-in: implement the `Driver` signature and
`gates.py`-style harnessing comes free.

`RunOutcome` and `Graded` both carry `metrics: dict | None`, populated by `native` and left
`None` by the oracle and stub drivers (§5.3). M4 only has to write it out.

---

## 3. Prerequisites outside the code

These gate everything in §4 and are not automatable from here.

1. **Download the six model configurations** in LM Studio (§2 table). Record what actually
   exists — some quantisations may not be published, in which case the configuration is dropped
   and the specification's §2 table is amended, not fudged.
2. **LM Studio server running** on `localhost:1234` with exactly one model loaded (§3.1).
3. ~~Decide how models are loaded and unloaded.~~ **Resolved.** `lms` is installed and
   supports `lms load -c <context>`, `--identifier`, `--estimate-only` and `lms unload --all`,
   with `lms ps --json` for what is resident. `harness/lmstudio.py` wraps it; a stage loads once
   and unloads once (§9.0). Note `/v1/models` lists everything *downloaded*, not what is
   loaded — only `lms ps` answers that.

---

## 3a. Reconnaissance findings (2026-08-27, `task_set_version: v2`)

**Not benchmark data.** One shot per task, one context, no environment capture, no repetitions.
Nothing here may be quoted as a result or written to `results/`. It exists to de-risk M3-M5.

### Artefact mapping — all six §2 rows have a working artefact

Identified by path, which is stable. The key column is a snapshot of what LM Studio derived
from the currently installed set and **will change** if models are added or removed (§2.1).

| §2 ID | Path (stable) | Size | Key at time of writing | Notes |
|---|---|---|---|---|
| LFM-M8 | `LiquidAI/LFM2.5-2.6B-MLX-8bit` | 2.88 GB | `lfm2.5-2.6b-mlx@8bit` | replaced an `mlx-community` build that crashed the MLX backend on load; that build has been removed |
| LFM-G8 | `LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf` | 2.87 GB | `lfm2.5-2.6b@q8_0` | |
| LFM-GQ4 | `LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf` | 1.59 GB | `lfm2.5-2.6b@q4_0` | **confirmed QAD**, not ordinary Q4_0 |
| LFM-BF16 | `LiquidAI/LFM2.5-2.6B-MLX-bf16` | 5.41 GB | `lfm2.5-2.6b-mlx@bf16` | |
| BON-M2 | `prism-ml/Ternary-Bonsai-8B-mlx-2bit` | 2.32 GB | `ternary-bonsai-8b-mlx` | |
| BON-G2 | `prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0_g64.gguf` | 2.31 GB | `ternary-bonsai-8b` | unsuffixed key: only one Bonsai GGUF is installed. Installing a second renames this one. Runtime reports quantisation as `null` |

Ternary-Bonsai-8B reports architecture `qwen3`.

### Context ceilings differ within a model

| Configuration | `maxContextLength` |
|---|---|
| LFM Q8_0, 8-bit, bf16 | 131072 |
| LFM QAD Q4_0 | **128000** |
| Bonsai, both | **65536** |

Confirms §2.1's refusal to assume: two quantisations of one model do not share a ceiling.
Bonsai cannot reach 128K, so Stage 4's 64K is its maximum.

### Behaviour — W01 / W05 / T01 at 8K

| Configuration | v1 | **v2** | mean progress | gen tok/s | overhead | peak (v1, per-process) |
|---|---|---|---|---|---|---|
| LFM-M8 (MLX 8-bit) | 2/3 | **3/3** | 4.0 | 56 | 306 ms | 4.6 GiB |
| LFM-BF16 (MLX) | 2/3 | **3/3** | 4.0 | 31 | 468 ms | 6.4 GiB |
| LFM-GQ4 (QAD Q4_0) | 2/3 | **3/3** | 4.0 | 86 | 34 ms | — |
| LFM-G8 (Q8_0) | 1/3 | 2/3 | 3.3 | 53 | 37 ms | — |
| BON-M2 | 0/3 | 0/3 | 1.0 | 64 | 255 ms | 3.8 GiB |
| BON-G2 | 0/3 | 0/3 | 1.0 | 38 | 44 ms | 1.2 GiB |

v1 scores were taken before the leading-`/` fix and are shown only to size its effect. The peak
column predates the §5.2 change and is not comparable across runtimes; ignore it.

- **Zero invalid tool calls anywhere, in either version.** Tool-call *formatting* is not the
  bottleneck for any configuration, which is not what §1.2 predicted. Stage 0 as specified may
  gate nothing.
- **T01 now passes on all four LFM configurations, in 3 tool calls each.** Under v1 every one
  of them failed it. That was the harness, not the models.
- **Bonsai is unchanged at 0/3** and its failure mode never touched path handling: 0-1 tool
  calls, then an answer from parametric knowledge without exploring. Both builds return
  byte-identical answers, which is also a useful determinism signal for the harness. Stage 0
  would pass this behaviour.
- **Reasoning share is a per-model property**: LFM 60-91 %, Bonsai 0 %. It drives agent latency
  far more than raw tok/s and belongs in the §10.3 table.
- **Per-request overhead is ~10x higher on MLX** (255-468 ms) than llama.cpp (34-44 ms). Since
  `prompt_tps` subtracts `overhead_median`, this bears directly on the runtime question.
- LFM-G8 on W01 is the one remaining LFM failure: 33 calls, `empty_answer`, where LFM-GQ4
  does it in 6. Single-shot, so indistinguishable from variance — see below.

### Run-to-run variance at `temperature=0` is real

T02 on LFM-M8, varying only `max_tokens`: pass at 1024, **fail at 2048**, pass at 4096. A
budget effect cannot produce that shape. This is the near-but-not-bitwise determinism §4.2
warns about, showing up as an outcome flip, and it is the first hard evidence that §9.1's three
repetitions and the `flaky` flag are load-bearing rather than ceremonial.

### Harness defects found and fixed

| Defect | Fix |
|---|---|
| LFM2.5 emits `reasoning_content` before any `content`; TTFT measured from first *content* folded the whole reasoning phase into prefill and inflated gen tok/s | `t_first` counts reasoning; `reasoning_tokens` recorded separately (§5.1) |
| Process discovery matched the Electron renderer instead of the backend | Hints target the backend; ranked by footprint, not RSS (§5.2) |
| Process discovery ran immediately after load; llama.cpp allocates lazily and was rejected as too small | Discover **after** overhead calibration (§5.2) |
| `max_tokens=1` made LM Studio finish without emitting a token delta, so overhead calibration silently returned 0.0 | `max_tokens=8`; no timed sample now fails loudly (§5.1) |
| W01 failed a *correct* answer that named £85 and identified £72 as superseded | The decoy may appear if marked superseded (§7.1) |
| A turn with neither tool calls nor content was graded as an empty `final_answer` | New `empty_answer` termination reason (§4.8) |
| **A leading `/` escaped the sandbox and was refused as "outside working directory"** — false, since the file is inside. Turn logging showed LFM-M8 run `ls -la`, see `AGENTS.md`, be told `/AGENTS.md` was outside, and thrash for ten turns | Leading `/` is root-anchored within the sandbox; `..` still refused (§4.6). **Cost every LFM configuration the whole of T01** |
| **Peak memory was not comparable across runtimes.** llama.cpp `mmap`s its weights into clean pages, which `phys_footprint` excludes (227 MB for a 2.87 GB artefact); MLX allocates dirty GPU buffers, which RSS excludes (707 MB for a 2.88 GB artefact). Each metric is blind to one runtime | Dirty + clean from `footprint -p` counts both (§5.2). A system-wide `vm_stat` delta was tried first and rejected: unbiased, but 1.54-2.82 GiB across six runs of one configuration |
| **Model identity was tied to LM Studio's `modelKey`, which is not stable.** The key is derived from the installed set (`@<quant>` appended only to disambiguate), and `lms load` matches it as a *substring*: `lms load lfm2.5-2.6b` matches four artefacts and under `--yes` loads the first — an MLX build — warning only on stdout, which the harness discarded on success | Models are identified by path; `resolve()` requires a unique exact match before the CLI is invoked, and the resident artefact is verified by path after load (§2.1) |

---

## 4. Milestones

Ordered. Each is independently verifiable.

### ~~M1 — LM Studio client and the `native` driver~~ — done and verified

**New files:** `harness/client.py`, `harness/driver_native.py`

**Client.** Wrap the `openai` SDK against `base_url="http://localhost:1234/v1"` with a dummy
API key. One method streams a chat completion and returns the assembled message plus raw chunk
timings. It must not interpret tool calls — that is the driver's job.

Details that will bite:

- `stream_options={"include_usage": True}` is required or `prompt_tokens` never arrives. The
  usage-bearing chunk has an empty `choices` list; do not index into it blindly.
- `delta.tool_calls` arrives **fragmented across chunks**, keyed by `index`. Accumulate
  `id`, `function.name` and `function.arguments` per index; arguments come as string fragments
  that must be concatenated, never parsed until complete.
- `top_k` and `repeat_penalty` are not OpenAI-standard. Pass them via `extra_body` and verify
  LM Studio honours them; record in `environment.json` whether it did.
- Confirm LM Studio honours `seed` for both MLX and llama.cpp backends. If it does not for one
  of them, that is a finding to record in §4.2, not something to paper over.

**Driver.** The loop:

1. `messages = [system(assemble(task.extra_rules)), user(task.prompt)]`
2. Stream a completion with `TOOL_SCHEMAS`.
3. If the assistant message has no tool calls → `final_answer` when it carries content,
   `empty_answer` when it does not (§4.8); return either way.
4. Otherwise append the assistant message **with its `tool_calls` intact**, then one `tool`
   message per call carrying `tool_call_id` and the result of
   `dispatch(sandbox, name, raw_arguments)`.
5. Repeat until a §4.8 termination condition fires.

Termination rules to implement exactly: `final_answer`, `max_steps` (25), `timeout` (600 s),
`loop_detected` (3 consecutive identical name+arguments), `malformed_calls` (5 consecutive
invalid).

**Do not** add retries, argument repair, JSON coercion, or a "you must call a tool" nudge.
§4.5 is the point of the exercise.

**Done when:** a fake client replaying canned chunk sequences drives the loop through every
termination reason in a unit test, and one real task completes end-to-end against a loaded
model.

### ~~M2 — Metrics~~ — done and verified

**New file:** `harness/metrics.py`

- Timing per §5.1. The subtle rule: `t_first` is the first chunk with non-empty
  `delta.content` **or** any `delta.tool_calls`; a role-only chunk does not count. Generation
  throughput uses `completion_tokens − 1`.
- `overhead_median`: 20 minimal-prompt requests at `max_tokens=1`, median TTFT, measured once
  per session and stored in `environment.json`.
- Memory sampler: background thread, `footprint -p <pid>` every 250 ms, retain the maximum of
  Dirty + Clean (§5.2). Start before the first request, stop after the last.
- Swap delta from `sysctl vm.swapusage` bracketing the run; non-zero sets `swap_flag`.
- Nonce prefix helper for Phase 1 (§5.4). Phase 2 does **not** use it.
- Per-turn TTFT list, so `ttft_turn1_s` and `ttft_median_later_s` can be reported separately.

**Done when:** unit tests over synthetic chunk sequences confirm the `t_first` rule and the
`−1` correction, and a real run produces plausible figures.

### M3 — Configuration probes and environment capture (§2.1, §3)

**New files:** `setup/probe_config.py`, `setup/probe_process.py`, `harness/environment.py`,
`configs/*.yaml`

- `probe_config.py` writes `configs/<id>.resolved.yaml` with every §2.1 field. Attention
  geometry comes from the model's own config — remember the LFM2 caveat that
  `n_attention_layers` is not the total layer count.
- KV admissibility function per §2.2, with a unit test against hand-computed numbers.
  Returns `admissible` / `unsupported` / `oversized`.
- `probe_process.py` discovers the LM Studio inference process name; it differs between MLX and
  llama.cpp backends and must never be hardcoded.
- `environment.py` emits `environment.json` and asserts the §3.1 preconditions: AC power
  (`pmset -g batt`), Low Power Mode off (`pmset -g`), exactly one model loaded (`GET /v1/models`
  plus whatever LM Studio exposes for loaded state), swap baseline recorded. **A failed
  precondition aborts the session** — no warnings.

### M4 — Stage 0 tasks, results output, stage runner (§9, §10.1)

**New files:** `harness/tasks/smoke.py`, `harness/results.py`, `harness/stages.py`

- Three trivial single-tool tasks for Stage 0: e.g. list the root, read one named small file,
  search for one literal string. Each must be solvable in a single correct tool call. They are
  a tool-calling check, not a reasoning check.
- JSONL writer against the exact §10.1 field list. Nullable fields carry `null`; never
  substitute an estimate. Transcripts persisted alongside, path recorded.
- Stage runner: repetitions, `flaky` flagging for non-unanimous results, `min_context` skip,
  `unsupported` / `oversized` skip, and the Stage 2A gate (≥3/10 passes **or** mean progress
  ≥2.5).
- A resume capability is worth having: Stage 2A is 180 runs and 6–12 hours. Key runs by
  `(config_id, suite, task_id, repetition)` and skip those already present in the JSONL.

### M5 — Stage 1 raw-inference corpus (§9 Stage 1)

**New file:** `fixtures/build_prompts.py`

- One prompt corpus, **identical text for every model**, sized to roughly 8K and 16K tokens.
  Token counts differ by tokeniser and are recorded, not equalised.
- A long-form generation prompt that reliably produces `completion_tokens ≥ 128`, plus an
  alternate for the retry path.
- Nonce prefix applied per §5.4.
- Model load time: decide how it is measured. LM Studio may report it; otherwise time an
  explicit load via `lms load`, or take the first-request latency after load and label it as
  such. Whatever is chosen, define it in §5.1 rather than leaving it implicit.

### M6 — Reporting (§10.2–§10.4)

**New file:** `harness/report.py`

Regenerate every table from JSONL only. Suite W and Suite T scores stay in separate columns and
are never averaged. Headline metric is successful tasks per hour of wall clock, per suite; ties
broken by peak memory. Swap-flagged runs are excluded from medians but still reported.

### M7 — `pi` driver and the parity gate (§4.1, §8)

**New file:** `harness/driver_pi.py`

Implement the same `Driver` signature by shelling out to pi against the same endpoint and the
same fixture copy. Record its version and system-prompt hash. Then add the **driver parity
gate** to `gates.py`: the oracle's tool sequence replayed through pi's fixture handling must
also score 20/20, proving the assertions are genuinely driver-independent before Stage 5A is
trusted.

### M8 — Stage 5B (§9 Stage 5B)

The context-compaction experiment and, only if greedy decoding proved degenerate, the
recommended-defaults sampling pass. Neither feeds the controlled comparison.

---

## 5. Suggested order of attack

Do **M1 + M2, then run Stage 0 against a single configuration**, before building M3–M5.

Three trivial tool calls will establish very quickly whether these models can emit valid tool
calls at all. Given the §1.2 design premise, that answer plausibly reshapes everything after
it — there is no sense building a resumable 180-run stage runner before knowing whether any
configuration clears the Stage 0 gate.

---

## 6. Open questions

Not blockers, but each needs an answer recorded in `benchmark.md` when resolved.

| Question | Why it matters |
|---|---|
| Does LM Studio honour `seed`, `top_k` and `repeat_penalty` on both backends? | §4.2 claims a fixed sampling block. **Still open.** `extra_body` is accepted without error, but that only proves it is not rejected. A first attempt to test `seed` was invalid: at `max_tokens=40` every token went to reasoning, so both samples were empty strings and compared equal. Retest with enough tokens for content to appear |
| ~~Is the reasoning ratio stable across configurations?~~ | **Resolved:** no. LFM 60-91 %, Bonsai 0 %. Promote to the §10.3 table |
| Is `max_tokens=1024` too tight for reasoning models? | **Mostly resolved, narrowed.** The v1 evidence for raising it was the leading-`/` defect, not the token budget: LFM-M8's T01 was thrashing on false path errors, and under v2 it completes in 3 turns well inside 1024. Keep 1024. **Still genuinely open for the write-heavy tasks** — T03, T07 and T09 require emitting whole file contents through `write_file`, and with 60-91 % of the budget going to reasoning, a 40-line file plausibly will not fit in one turn. Check T03 and T09 specifically before Stage 2B; do not generalise from the retrieval tasks |
| Is Stage 0 worth keeping? | Every configuration emitted valid tool calls with zero formatting errors, so the gate as specified may exclude nothing. Bonsai fails by giving up after one call, which Stage 0 would pass. Consider whether the gate should test persistence rather than syntax — but decide before running, not after seeing scores |
| How is model load time measured? | Listed as a Stage 1 metric but never defined in §5.1 |
| Can `lms` set context length at load time? | Determines whether stages can run unattended across configurations |
| ~~Do all six quantisations actually exist?~~ | **Resolved:** all six exist and all six now load. See §3a |
| ~~How should peak memory be measured?~~ | **Resolved:** dirty + clean from `footprint -p`. Each runtime puts the weights where one standard metric cannot see them — llama.cpp in clean mapped-file pages (invisible to `phys_footprint`), MLX in dirty GPU buffers (invisible to RSS). Summing both columns counts both. Spread 0.03 GiB over three repetitions, against 1.28 GiB for the system-wide delta it replaced. See §5.2 |
| Does the 2 GiB headroom in §2.2 hold in practice? | It is a stated margin, not a measurement. Worth checking against observed peak memory once Stage 1 has run. First figures: llama.cpp Q8_0 2.90-2.93 GiB, MLX 8-bit 4.18 GiB, both at 8K on W05 |

---

## 7. Known risks

- **Near-zero pass rates.** Anticipated, and the progress score (§7.3) and two-suite split (§7)
  exist for it. Resist the urge to make tasks easier once results arrive — that changes
  `task_set_version` and discards comparability. Record the finding instead.
- **Truncation dominating failures.** §4.7 explains why several Suite W tasks require
  `run_command`. If failure analysis shows models never recognising the truncation marker, that
  is a legitimate finding; do not raise the limit mid-benchmark.
- **Swapping.** The likely cliff on a memory-constrained machine, especially at 32K+ with
  BF16. `swap_flag` exists so
  this shows up as a flag rather than as mysteriously poor throughput.
- **Wall clock.** Stage 2A alone is 6–12 hours. Build the resume capability in M4 before
  starting a long stage, not after losing one to an interruption.
