# Findings

A running, dated log of empirical observations from real runs — model behaviour, environment
quirks, anything learned by actually using the harness against a real backend. Distinct from the
other two documents on purpose:

- [`benchmark.md`](benchmark.md) is the specification: authoritative on what is measured and
  how, and stays that way. It is not the place for a single configuration's anecdotal behaviour.
- [`implementation-plan.md`](implementation-plan.md) tracks *defects in this codebase* found
  while building it — bugs in our code, fixed and covered by tests, filed against the milestone
  that exposed them. It is scoped to shrink to nothing once the harness is done.

A finding here is neither: not a spec change, and not something to fix in this repository. It is
a fact about a model, a backend, or the environment, worth keeping because it will otherwise be
rediscovered — or worse, silently forgotten and misinterpreted as noise — the next time someone
reads the data it explains. Each entry cites the record/transcript paths it is drawn from, so it
stays checkable rather than asserted.

---

## 2026-08-29 — LM Studio's Engine Protocol runtime corrupts streamed tool-call arguments

**What happened.** A live Stage 2A run against LFM-GQ4 crashed the whole process on W02's third
repetition: LM Studio's backend returned a 500 mid-stream, `"Invalid diff: '...' not found at
start of '...'"`. It reproduced deterministically — byte-identical message, 3/3 repetitions — on
every task whose `run_command` argument embeds a Python one-liner with a single-quoted `open(...)`
call (W02, W03, W09; 9 of the 30 Stage 2A runs, 30%). LM Studio's own server log
(`~/.lmstudio/server-logs/`) named the responsible subsystem: `Engine protocol predict stream
returned an error`.

**Root cause.** `~/.lmstudio/settings.json` had `useLlamaCppEngineProtocolRuntime3: true` — a
developer setting behind a known LM Studio bug,
[lmstudio-ai/lmstudio-bug-tracker#1922](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1922),
in the same "Engine Protocol runtime" subsystem: it corrupts streamed tool-call arguments (a
different exact message there — "Unrepairable tool_call arguments... replaced with empty
object" — but the same failure family and code path). The documented workaround is to disable
the setting in LM Studio (Settings → Developer).

**Resolution and evidence.** The user disabled the setting; Stage 2A was re-run for LFM-GQ4 from
scratch (the pre-fix run couldn't just be resumed into, since resuming would have silently pooled
runs made under two different environments — see the `report.py` note below). Zero
`server_error` in the clean run. The three previously-crashing tasks now fail genuinely
(`empty_answer` — see the next entry) rather than crashing the backend.

- Pre-fix run (archived, excluded from reporting): `results/LFM-GQ4-8192/archive/stage2a-engine-protocol-runtime-bug.jsonl`
- Clean re-run: `results/LFM-GQ4-8192/raw/stage2a.jsonl`

**Why this matters beyond LFM-GQ4.** The setting is machine-wide, not per-configuration — it was
on for every run before this was found, including all six configurations' Stage 0 data. Stage 0's
trivial single-tool-call tasks never embed the kind of deeply-nested-quoted argument that
triggers it, so it's unlikely to have silently affected that data, but this is worth remembering
if a `server_error` shows up in *any* future stage: check this setting before assuming it's a new
bug. `environment.json` does not currently capture it (§3's field list has no entry for backend
developer settings) — a real gap, not yet worth a schema change on the strength of one incident,
but promote it to one if this recurs.

**A second bug found while investigating this one.** `harness/report.py`'s `load_all_records`
globs every file under a session's `raw/`, unconditionally — no name filter. Archiving the
pre-fix run by renaming it *within* `raw/` (the first attempt) was silently pooled straight back
into the report, doubling the apparent run count (24/60 instead of the correct 12/30). Fixed by
moving the archive to a sibling `archive/` directory instead — `raw/` now means "live, reportable
data," full stop, and that convention is documented in `report.py`'s module docstring.

---

## 2026-08-29 — LFM-GQ4 never discovers the one permitted `run_command` idiom

**What happened.** In the clean (post-fix) Stage 2A run, LFM-GQ4 passed all four tasks solvable
by pure retrieval (`read_file`/`list_files`/`search_files` alone — W01, W04, W05, W08) and failed
all four requiring computed aggregation over CSV data (W02, W03, W09, W10) — a 100% split along
exactly that line.

**The failure mechanism, from the transcripts.** The model reaches for `run_command` with
`cd /workspace && python3 -c "..."`, gets `exit=127 command not permitted: cd`. Retries with
just `python3 -c "..."`, gets `exit=127 command not permitted: python3`. In one case it then
tries `awk`, also refused; in another its retry has unmatched quoting and fails to parse. **It
never once tries plain `python`** — the one binary actually on the §4.6 allowlist (`ls cat grep
find head tail wc python`, no `cd`; the oracle solves these same tasks with exactly that). After
a few refusals it falls back to reading the raw CSV directly instead of computing over it, and
its final turn produces neither a tool call nor content — `empty_answer`.

**This is not a harness defect.** The allowlist is deliberate and documented (§4.6, protected by
§11 — changing it bumps `task_set_version`), and the system prompt gives no hint about permitted
commands or a `/workspace` path by design: recovering from an unhinted refusal through trial and
error is exactly the capability under test (§4.5's no-repair rule exists so the harness doesn't
do this recovery *for* the model). LFM-GQ4 has a strong prior toward `python3`/`cd` — reasonable
conventions in many other contexts — and doesn't converge to the one idiom that would work, nor
does it fall back to doing the arithmetic in its own reasoning once tool use stalls.

**Why this matters beyond LFM-GQ4.** LFM-M8, LFM-BF16 and LFM-G8 are the same base model
(LFM2.5-2.6B) at different quantisations, so this is plausibly a property of the model family's
training, not this one quantisation — worth checking specifically once their Stage 2A data
exists, rather than treating each configuration's `run_command`-dependent failures as
independent data points. If it holds across all four, it's a genuine, reportable finding about
LFM2.5 specifically, not noise.

- Evidence: `results/LFM-GQ4-8192/raw/stage2a.jsonl` (task_id W02/W03/W09/W10, all three
  repetitions each) and the matching transcripts under `results/LFM-GQ4-8192/transcripts/`.

---

## 2026-08-27 — Reconnaissance (`task_set_version: v2`)

**Not benchmark data.** One shot per task, one context, no environment capture, no repetitions.
Nothing here was ever quoted as a result or written to `results/`. It existed to de-risk M3-M5
before they were built, and is kept for the record now that they are.

### Artefact mapping — all six §2 rows have a working artefact

Identified by path, which is stable. The key column was a snapshot of what LM Studio derived
from the installed set at the time and moves as models are installed or removed (§2.1).

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

Confirms §2.1's refusal to assume: two quantisations of one model do not share a ceiling. Bonsai
cannot reach 128K, so Stage 4's 64K is its maximum.

### Behaviour — W01 / W05 / T01 at 8K

| Configuration | v1 | **v2** | mean progress | gen tok/s | overhead | peak (v1, per-process) |
|---|---|---|---|---|---|---|
| LFM-M8 (MLX 8-bit) | 2/3 | **3/3** | 4.0 | 56 | 306 ms | 4.6 GiB |
| LFM-BF16 (MLX) | 2/3 | **3/3** | 4.0 | 31 | 468 ms | 6.4 GiB |
| LFM-GQ4 (QAD Q4_0) | 2/3 | **3/3** | 4.0 | 86 | 34 ms | — |
| LFM-G8 (Q8_0) | 1/3 | 2/3 | 3.3 | 53 | 37 ms | — |
| BON-M2 | 0/3 | 0/3 | 1.0 | 64 | 255 ms | 3.8 GiB |
| BON-G2 | 0/3 | 0/3 | 1.0 | 38 | 44 ms | 1.2 GiB |

v1 scores were taken before the leading-`/` fix (see `implementation-plan.md`'s M1 defect table)
and are shown only to size its effect. The peak column predates the §5.2 change and is not
comparable across runtimes; ignore it.

- **Zero invalid tool calls anywhere, in either version.** Tool-call *formatting* is not the
  bottleneck for any configuration, which is not what §1.2 predicted. Stage 0 as specified gates
  nothing in this set, and was later confirmed live against all six (see the M4 notes in
  `implementation-plan.md`); kept as a pre-flight check regardless.
- **T01 now passes on all four LFM configurations, in 3 tool calls each.** Under v1 every one of
  them failed it. That was the harness, not the models.
- **Bonsai is unchanged at 0/3** and its failure mode never touched path handling: 0-1 tool
  calls, then an answer from parametric knowledge without exploring. Both builds return
  byte-identical answers, which is also a useful determinism signal for the harness. Stage 0
  would pass this behaviour.
- **Reasoning share is a per-model property**: LFM 60-91 %, Bonsai 0 %. It drives agent latency
  far more than raw tok/s and belongs in the §10.3 table.
- **Per-request overhead is ~10x higher on MLX** (255-468 ms) than llama.cpp (34-44 ms). Since
  `prompt_tps` subtracts `overhead_median`, this bears directly on the runtime question.
- LFM-G8 on W01 was the one remaining LFM failure at the time: 33 calls, `empty_answer`, where
  LFM-GQ4 did it in 6. Single-shot, so indistinguishable from variance at the time — worth
  revisiting once real Stage 2A data exists for LFM-G8.

### Run-to-run variance at `temperature=0` is real

T02 on LFM-M8, varying only `max_tokens`: pass at 1024, **fail at 2048**, pass at 4096. A budget
effect cannot produce that shape. This is the near-but-not-bitwise determinism §4.2 warns about,
showing up as an outcome flip, and was the first hard evidence that §9.1's three repetitions and
the `flaky` flag are load-bearing rather than ceremonial.

### Harness defects found and fixed

Filed here rather than `implementation-plan.md` because this reconnaissance run predates the
milestone structure that document now tracks defects against — these were found across what
became M1-M3, in one session, before the harness could run a real stage.

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
| **The GGUF metadata reader could not read either artefact.** Reads were not bounds-checked — only the 4-byte value type was — so any string or array straddling a 1 MB chunk boundary failed on a valid file; the alignment retry then walked the whole 2.3 GB artefact to EOF, at `buf += chunk` (quadratic). Minutes per file, then failure | Every read goes through `ensure()`, with plausibility bounds on string and array lengths so a misaligned decode fails immediately. `bytearray.extend` replaces the concatenation. Both artefacts now parse in 0.4 s |
| **`n_kv_heads` read as 0 for every LFM2 GGUF**, blocking all three GGUF configurations. `attention.head_count_kv` is a **per-layer array** there — `[0, 0, 8, 0, 0, 8, ...]`, zero for each conv block — and the reader took element 0 | Arrays are preserved and the distinct non-zero value taken (§2.2 caveat). GGUF and MLX geometry now agree independently: LFM 8/8/64, Bonsai 36/8/128 |
| **KV probe measured nothing for MLX.** It varied *declared* context, which llama.cpp allocates to at load but MLX does not — MLX allocates on first touch, sized to the sequence, and the warm-up was 2 tokens. Reported 0.058 and 0.0005 bytes/element, i.e. a 0.00 GiB KV cache that **passed admissibility at every context**, silently | Two slopes, larger wins: declared context (eager) and prompt length (lazy). A geometry cross-check now refuses any measurement implying <0.25 or >8 bytes/element, so this failure mode is loud (§2.2) |
