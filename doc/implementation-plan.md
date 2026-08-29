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
| Configuration probes | `setup/probe_config.py`, `setup/probe_process.py` | All six `configs/*.yaml` resolved; 31 tests |
| Environment capture | `harness/environment.py`, `harness/admissibility.py` | §3.1 preconditions verified live; 20 tests |
| Stage 0/1 + resumable stage runner + Stage 2A gate + `run_full` | `harness/tasks/smoke.py`, `harness/results.py`, `harness/stages.py` | Stage 0 (all six configs) and Stage 1 (LFM-M8) verified live; Stage 2A/2B/3/`run_full` unit-tested only; 44 tests |
| Stage 1 corpus | `fixtures/build_prompts.py` | Verified live via Stage 1; 6 tests |
| Reporting | `harness/report.py` | Verified live against real Stage 0/1 data; 17 tests |
| Stage 5B compaction | `NativeDriver.history_mode`, `harness/stages.py`'s `run_stage5b_compact` | Unit-tested only, not run live |

The harness half runs with no model and no network (162 tests):

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

Five defects that only a real model exposed, now fixed and covered by tests:

| Defect | Fix |
|---|---|
| LFM2.5 emits `reasoning_content` before any `content`; 60 % of its generated tokens were reasoning. TTFT measured from first *content* folded the whole reasoning phase into prefill and inflated gen tok/s | `t_first` counts reasoning; `reasoning_tokens` recorded separately (§5.1) |
| Process discovery matched the Electron renderer (0.5 GiB) instead of the backend (6.5 GiB) | Hints target `.lmstudio/.internal`; candidates ranked by footprint, not RSS; refuses a process too small to hold a model (§5.2) |
| `max_tokens=1` made LM Studio finish without emitting a token delta, so overhead calibration silently returned 0.0 and prompt tok/s went unadjusted | `max_tokens=8`; no timed sample now fails loudly (§5.1) |
| W01 failed a *correct* answer that named £85 and identified £72 as superseded | The decoy may appear if marked superseded (§7.1) |
| **An uncaught `openai.APIError` crashed the whole process.** A live Stage 2A run against LFM-GQ4 hit a real backend 500 mid-stream (`"Invalid diff: ... not found at start of ..."`, llama.cpp choking on a long tool-call argument) on W02's third repetition; the exception propagated all the way up and killed the run, losing every task still queued behind it — resumability meant only the already-written records survived. Reran after the fix below: **root cause was `useLlamaCppEngineProtocolRuntime3: true` in `~/.lmstudio/settings.json`** — a known-buggy LM Studio developer setting ([lmstudio-ai/lmstudio-bug-tracker#1922](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1922)) that corrupts streamed tool-call arguments; disabling it is the documented workaround | `NativeDriver` catches `APIError` around the one `stream_turn` call and ends just that run with a new `server_error` termination reason (§4.8), never retried or repaired — this is an infrastructure fault, not the model's mistake, so §4.5's no-repair rule doesn't apply to it. Kept regardless of the LM Studio fix: a legitimate defensive property, not conditional on this one root cause |

### Not built

The `pi` driver (M7), and Stage 5B's recommended-default sampling pass (conditional — its
detector is built, but nothing has triggered it yet). Stage 2A/2B/3 and the compaction variant
of Stage 5B are built and unit-tested but not yet run live — each is hours long against a real
model, and `run_full()` chains all of Stage 0-3 in one command once that's wanted.

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

Reconnaissance findings from before M3-M5 were built, and every empirical finding since, now
live in [`findings.md`](findings.md) — including the six-configuration artefact mapping, the
context-ceiling and reasoning-share data, and the harness defects that reconnaissance run
exposed. Kept there rather than here because a finding is not "remaining work"; this document
tracks only what still needs building.

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

### ~~M3 — Configuration probes and environment capture~~ — done and verified

**New files:** `setup/probe_config.py`, `setup/probe_process.py`, `harness/environment.py`,
`configs/*.yaml`

- `probe_config.py` writes `configs/<id>.resolved.yaml` with every §2.1 field. Attention
  geometry for GGUF builds comes from the `gguf` package (now in `pyproject.toml`); MLX
  builds read `config.json`. The LFM2 caveat is concrete and testable: LFM2.5-2.6B has 30
  hidden layers of which only 8 are `full_attention` — count occurrences in `layer_types`,
  never `num_hidden_layers`. Caveat: Bonsai's custom `Q2_0_g64` quant (GGML type 42)
  crashes `gguf`'s tensor parse with no skip option, so a minimal metadata-only GGUF header
  parser is the fallback for files the library cannot open — it reads the same fields and
  stops before the tensor section.
- **No KV runtime probe.** An earlier design measured `kv_elem_bytes` by loading each
  configuration at two context lengths, or two prompt lengths, and dividing a footprint delta
  by a token delta. It was tried, cost ~15 minutes across the six configurations, and was
  abandoned: llama.cpp allocates KV eagerly at load but MLX allocates lazily on first touch, so
  no single probe design measured both correctly, and the errors ran in the damaging
  direction — an MLX measurement of essentially zero passed admissibility at every context,
  while an over-estimate would have marked BON-M2 `oversized` at 32K/64K although it loads
  fine. See §2.2 for the replacement and the full account.
- `probe_config.py` therefore reads metadata only and loads no model: the whole §2 table
  resolves in seconds. Geometry (`n_attention_layers`, `n_kv_heads`, `head_dim`) is still
  extracted as a cross-check, never as an admissibility input.
- `quant_sha256`: SHA-256 of the weights file — `.gguf` for GGUF builds, `model.safetensors`
  for these MLX builds; a sharded artefact records a sorted `(relpath, sha256)` list.
  `model_revision` is dropped from the schema — LM Studio does not expose it. **Optional at
  both ends** (§2.1): off by default at setup (`--hash` to enable — it is otherwise the whole
  cost of resolving a configuration) and skipped at session start when no hash was recorded.
  `environment.json` records `quant_sha256_verified` so a result set never implies a check
  that did not run.
- Admissibility per §2.2 is now three cheap checks, not one arithmetic function: recorded
  metadata rules out `unsupported`, the load attempt itself is `oversized` on a memory
  refusal (exact for llama.cpp, silent for MLX by construction), and `peak_memory_bytes` /
  `swap_flag` from §5.2 are the record of what a run actually cost. `harness/admissibility.py`
  holds `classify_declared` and `classify_load_failure`, both covered by unit tests.
- `probe_process.py` derives `backend_runtime` (name + version) for `environment.json` from
  the resident model's `format` (`lms ps`, `safetensors`/`gguf`) matched against the engine
  `lms runtime ls` reports selected for that format. An earlier design read the resident
  process's command line instead, reusing `metrics.py`'s hint-and-footprint search (§5.2); a
  live verification run showed LM Studio's worker process (`llmworker.js`) names neither
  engine nor version on either backend, so that design always returned `None, None`. Process
  discovery in `metrics.py` remains the only source for *memory*, which is a property of the
  process; `backend_runtime` identity is not.
- `environment.py` emits `environment.json` and asserts the §3.1 preconditions: AC power
  (`pmset -g batt`), Low Power Mode off (`pmset -g`), exactly one model loaded (`GET
  /v1/models` plus `lms ps`), swap baseline recorded. **A failed precondition aborts the
  session** — no warnings.

**Verified end-to-end** (2026-08-29): LFM-M8 and LFM-G8 each loaded via `lms`, `environment.capture()`
run live, and `environment.json` inspected — `ac_power`, `low_power_mode`, `backend_runtime`
(`mlx`/`1.11.0` and `llama.cpp`/`2.29.1`) and `context_length` all correct. The single-resident-model
precondition was also confirmed to fire: loading a second configuration while the first stayed
resident raised `PreconditionError` with the expected message. `lms load --context-length` was
confirmed to actually set the context (answers the open question below).

### ~~M4a — Stage 0 tasks, results output, resumable stage runner~~ — done and verified

**New files:** `harness/tasks/smoke.py`, `harness/results.py`, `harness/stages.py`

- Three trivial single-tool tasks for Stage 0 (`S01`-`S03`): list the root, read `README.md`,
  search for a literal phrase — all fixed text in `fixtures/build_workspace.py`, independent of
  Suite W/T content so Stage 0 never moves when a fixture is regenerated. Each `check` inspects
  `ctx.calls` directly for a valid call of the right tool against the right target; it is a
  tool-calling check, not a reasoning check, so the final answer is not graded.
- `harness/results.py`: JSONL writer against the exact §10.1 field list, one file per stage
  under `results/<session>/raw/<stage_name>.jsonl`. `append_record` refuses a record whose keys
  don't match the schema exactly. `flaky` is always written `null` — see §9.1's amended text:
  deciding it needs every repetition of a task, which resume (below) writes one at a time, and
  JSONL is append-only, so it is a reporting-layer computation (M6), not a write-time one.
- `harness/stages.py`'s `run_stage()`: loads a model once (§9.0), runs `tasks × repetitions`
  under a fresh `MemorySampler`/`SwapWindow` pair per run (mirrors `harness/smoke.py`), persists
  each run's transcript, and unloads on the way out — including on a `ModelOversizedError`,
  reported as an `oversized` stage outcome rather than raised. `classify_declared` short-circuits
  an `unsupported` context before any load is attempted. Generic over the task list and
  repetition count, so Stage 2A/2B reuse it unchanged; `run_stage0()` is the first caller,
  applying the §9 Stage 0 gate (below).
- **Resume is automatic, not a flag.** `environment.capture()` gained an optional `session_id`
  parameter; `run_stage()` always passes `f"{config_id}-{context_length}"`, so re-running the
  same stage command lands in the same `results/<config_id>-<context>/` directory and
  `results.existing_keys()` on that stage's own raw file skips what is already there. Verified
  live (below): a second identical invocation touched no model call and left the raw file's
  record count unchanged.
- **Stage 0 gate math, resolved.** §9's "fewer than 2 of 3 valid tool calls" for "3 tasks × 3
  repetitions" is read as an aggregate rate over the 9 runs (≥6/9), not a per-task count — a
  Stage 0 task is solved in exactly one tool call, so a per-task "2 of 3" is not a meaningful
  quantity. Recorded in `benchmark.md` §9 Stage 0.
- **`task_set_version` is the resume-safety signal, not `environment_sha256`.** The first design
  compared the full `environment_sha256` across a resumed session's existing records and the
  freshly captured one, refusing to append on any mismatch. A live verification run broke this
  immediately: `free_memory_bytes` and `swap_used_bytes_start` are instant-in-time machine
  readings that differ between any two captures regardless of comparability, so the hash never
  matched twice and resume was defeated outright — the harness raised `SessionMismatchError` on
  the second, identical invocation. Fixed to compare `task_set_version` instead (`harness/
  stages.py`'s `_check_task_set_version_matches`), which §11 already designates as the one flag
  that means "results are/aren't comparable," and is a per-record field so no extra file read is
  needed. `environment_sha256` is unchanged as a per-record §10.1 field — each run still points
  at exactly the capture it was taken under.

**Verified end-to-end** (2026-08-29): `python -m harness.stages stage0 LFM-M8` against the real
LFM-M8 configuration — 9/9 valid tool calls, `tool_capable=True`, transcripts persisted and
readable, model unloaded afterwards. Re-running the identical command resumed correctly: the raw
file stayed at 9 records and no new tool call was made (confirmed by timing — the second run
took ~18 s, all of it load and overhead calibration, none of it task execution).

**Stage 0 has now actually been run for all six §2 configurations** (2026-08-29), not just
reconnaissance-style probing — `results/<config_id>-8192/raw/stage0.jsonl`, one directory per
configuration. All six: 9/9 valid tool calls, `tool_capable=True`. `backend_runtime` resolved
correctly for every one (`llama.cpp`/2.29.1 for the three GGUF configurations, `mlx`/1.11.0 for
the three MLX ones), confirming the M3 backend-identity fix (`findings.md`'s defect table) holds
across the whole matrix, not only the one configuration it was diagnosed against. This is real §9
Stage 0 data — kept under `results/`, not deleted as scratch, per `.gitignore`'s own note that
the raw JSONL is the record.

### ~~M4b — the Stage 2A gate and the `min_context` skip~~ — built and unit-tested

`run_stage()` gained the `min_context` skip it was missing: a task whose `min_context` exceeds
the stage's `context_length` is excluded entirely — no runs, no records, and not scored as a
failure — recorded on `StageOutcome.skipped_min_context`. Nothing in Stage 0 exercised this
(all three tasks declare `min_context=8192`, equal to the only context Stage 0 runs at), so it
is covered by `tests/test_stages.py` against a synthetic mixed-context task pair, not by a live
run.

`run_stage2a()` calls `run_stage()` against `SUITE_W` at 3 repetitions and applies the gate
recorded in `benchmark.md` §9 Stage 2A: majority-vote per task (≥2 of 3 repetitions), mean
progress over every included run, either condition sufficient to proceed. The arithmetic is
split into `_evaluate_stage2a(stage: StageOutcome)`, a pure function over records, so the gate's
edge cases — majority vs minority, pass-count-alone, mean-alone, neither, and a
`min_context`-skipped task correctly excluded rather than counted as failing — are unit-tested
directly against synthetic records without driving all ten Suite W tasks through a fake client.

**Not yet run live.** Stage 2A is 180 runs, 6-12 hours (§9.2) — a deliberate, separate action,
not something to kick off incidentally while finishing the runner. The CLI is ready:
`python -m harness.stages stage2a <config_id>`.

### ~~M5 — Stage 1 raw-inference corpus~~ — done and verified

**New file:** `fixtures/build_prompts.py`

Writes `fixtures/prompts/{8k,16k}_{primary,alternate}.txt` — synthetic, deterministic, rng-
templated documents (same spirit as `build_workspace.py`), sized by a 4-chars/token heuristic
used only to decide how much filler to generate; actual token counts are recorded from the API
response (§9 Stage 1's own text on this). Each ends with a closing instruction that reliably
produces `completion_tokens ≥ 128`; `primary`/`alternate` per tier are independently generated
bodies, not the same text with a different last line.

`harness/stages.py`'s `run_stage1()` runs the raw completions directly (no `Task`/`Sandbox` —
Stage 1 isn't task-based), retrying once with the alternate prompt on a short completion, and
maps the result onto a §10.1 record with `passed`/`progress_score` left `null`. Details recorded
in `benchmark.md` §9 Stage 1.

**Verified end-to-end** (2026-08-29): `python -m harness.stages stage1 LFM-M8` against the real
configuration, both tiers. 8K: ~6500 prompt tokens (against a 8192-token heuristic target — the
tokeniser gap is exactly what §9 Stage 1 says to expect, not equalised), 1023 completion tokens
every repetition (no retry needed), TTFT ~14.6s, gen tok/s ~49, prompt tok/s ~455. 16K: ~13018
prompt tokens, same completion/gen-tps pattern, TTFT ~30s. No swap, model unloaded afterwards.
Every repetition hit `termination_reason: "length"` (cut off by `max_tokens=1024`, not a natural
stop) — a property of pairing a long-response instruction with the fixed sampling budget, not a
defect; changing `max_tokens` per stage would be per-model tuning, which §11 forbids.

### ~~M6 — Reporting~~ — done and verified

**New file:** `harness/report.py`

Regenerates §10.1's `flaky` (grouping raw records by `(config_id, suite, task_id)`, unanimity of
`passed` — deliberately deferred to here by `stages.py`, per §9.1), §10.2's headline metric, and
§10.3's final table, all from JSONL only. Two computations needed a precise, recorded decision
because the spec doesn't fully pin them down — both are now in `benchmark.md`:

- **Headline metric denominator** (§10.2): every run's wall clock in that suite's stage, not
  only the passing runs' — a fast failure must not outscore a slow one.
- **Final table's throughput columns** (§10.3) come from Stage 1 at 8K specifically, never
  Suite W/T, because §5.4 rules out comparing Phase 2 numbers across configurations. **Verdict**
  is a mechanical stage-progression status, not the qualitative judgement the prose implies —
  that's §10.4, written by a person once real multi-configuration data exists. §10.4 itself is
  not auto-generated for the same reason.

Also built here: the §4.2 degenerate-rate detector (`is_degenerate_triggered`) that Stage 5B's
recommended-sampling pass depends on — see M8.

**Verified end-to-end** (2026-08-29): `python -m harness.report` against the real committed
Stage 0 (all six configurations) and Stage 1 (LFM-M8) data — correct verdicts (`"passed Stage 0
only"` where no Suite W data exists yet), correct throughput row for LFM-M8, graceful handling
of configurations with no Suite W/T data rather than crashing.

### M7 — `pi` driver and the parity gate (§4.1, §8)

**New file:** `harness/driver_pi.py`

Implement the same `Driver` signature by shelling out to pi against the same endpoint and the
same fixture copy. Record its version and system-prompt hash. Then add the **driver parity
gate** to `gates.py`: the oracle's tool sequence replayed through pi's fixture handling must
also score 20/20, proving the assertions are genuinely driver-independent before Stage 5A is
trusted. Not started — not requested yet.

### M8 — Stage 5B — partially done: the compaction experiment; unit-tested, not run live

§9 Stage 5B has three parts, only one of which is a concrete, buildable deliverable — see
`benchmark.md` §9 Stage 5B for the reasoning recorded against each:

- **Alternative quantisations**: no new code needed. `config_id` is already a free parameter.
- **Recommended-default sampling**: conditional on `harness/report.py`'s
  `is_degenerate_triggered` actually firing for some configuration, which requires real Stage
  2A/2B data that doesn't exist yet. The detector is built; the sampling pass itself is an
  operator action once triggered, not an automatic pipeline step.
- **Context-compaction experiment** — built: `NativeDriver` gained a `history_mode` field
  (`"full"`/`"compact"`); compacting sends only `[system, user] + <the most recent
  assistant+tool exchange>`, tracked by turn boundary rather than a fixed message count so a
  multi-tool-call turn is never split (which would leave a `tool_call_id` unanswered and produce
  an invalid request). `harness/stages.py`'s `run_stage5b_compact()` runs Suite W and T through
  it, writing `driver="native-compact"` to its own raw files
  (`stage5b-compact-{w,t}.jsonl`) — never pooled with the controlled comparison.

**Not run live** — Suite W/T through the compaction driver is exactly as long as Stage 2A/2B
themselves; verified by `tests/test_driver.py`'s compaction tests (full history vs. compact,
multi-call turns kept together, the transcript unaffected) and `tests/test_stages.py`, not
against a real model.

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
| ~~Is Stage 0 worth keeping?~~ | **Resolved: keep, unchanged.** Reconnaissance showed all six configurations emit valid tool calls with zero formatting errors, so the gate is expected to exclude nothing in the current set. It is retained as a pre-flight check against harness/configuration mismatch — the configuration set is a snapshot, and a future model that cannot call tools would otherwise cost Stage 2A hours to discover. A persistence-style gate was considered and rejected: persistence is a continuous capability already measured by the progress score and the Stage 2A gate, and gating on it would blur gate and measurement. §9 Stage 0 and §1.2 amended accordingly |
| ~~How is model load time measured?~~ | **Resolved: it is not measured.** Removed from the Stage 1 metric list in §9. §5.1 timings are defined against an already-serving model, and §9.0 loads once per stage precisely to keep load time out of them; the duration of `lms load` would measure LM Studio's loader, not agent work. Time to first token on a loaded model, which is relevant, is already covered by TTFT |
| ~~Can `lms` set context length at load time?~~ | **Resolved: yes.** `lms load --context-length 8192` confirmed live: `lms ps` and the returned `LoadedModel.context_length` both report 8192 for both an MLX and a GGUF configuration |
| ~~Do all six quantisations actually exist?~~ | **Resolved:** all six exist and all six now load. See `findings.md` |
| ~~How should peak memory be measured?~~ | **Resolved:** dirty + clean from `footprint -p`. Each runtime puts the weights where one standard metric cannot see them — llama.cpp in clean mapped-file pages (invisible to `phys_footprint`), MLX in dirty GPU buffers (invisible to RSS). Summing both columns counts both. Spread 0.03 GiB over three repetitions, against 1.28 GiB for the system-wide delta it replaced. See §5.2 |
| ~~Does the 2 GiB headroom in §2.2 hold in practice?~~ | **Moot.** §2.2 dropped the arithmetic margin entirely — admissibility is now a metadata check, a load attempt, and §5.2 measurement, so there is no headroom figure left to validate. First real peak-memory figures: llama.cpp Q8_0 2.90-2.93 GiB, MLX 8-bit 4.18 GiB, both at 8K on W05 |
| Do LM Studio's memory-refusal error messages match `classify_load_failure`'s patterns? | **Untested against a real refusal.** No §2 pair on this 16 GiB machine actually triggers `oversized`; the string patterns ("out of memory", "failed to allocate", "guardrail", …) are unit-tested against assumed wording only. A real refusal with different phrasing surfaces as a plain `LMStudioError` rather than `oversized` — loud, but mislabelled. Capture the actual message the first time a stage hits one |

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
