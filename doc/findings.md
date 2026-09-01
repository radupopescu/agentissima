# Findings

A dated log of empirical observations from real runs: model behaviour, environment issues,
anything learned by using the harness against a real backend.

This is separate from the other two documents:

- [`benchmark.md`](benchmark.md) is the specification: what is measured and how. It does not
  hold observations about one configuration's behaviour.
- [`implementation-plan.md`](implementation-plan.md) tracks defects in this codebase found
  while building it. It shrinks to nothing once the harness is finished.

A finding here is neither a spec change nor a bug to fix in this repository. It is a fact about
a model, a backend, or the environment. Each entry cites the record or transcript paths it is
drawn from.

---

## 2026-09-01 — the agent's `bash` tool read outside the fixture in 12% of calls

**What happened.** The Seatbelt profile in `harness/driver_pi.py` denies writes outside the
fixture copy and says nothing about reads, and `pi-permission-system` cannot inspect `bash`
because a shell command is an opaque string with no structured path argument. Surveying every
`bash` call in the 120 `v4` pi runs across LFM-G8 and LFM-M8:

| Count | Command |
|---:|---|
| 7 | `find / -name "expense.csv" -type f` |
| 4 | `find / -name "balances.py"` |
| 1 each | `find / -name "*.csv" -type f \| head -50`, `find / -type f -name "*.py" \| grep -i balance`, `find / -name "*expense*"`, `find / -name "expenses.csv"`, `find / -name "posting.py"`, `find / -name "test_posting.py"`, `find /private -name "posting.py"`, `cd /private && python -m pytest …`, `ls -la /root/`, `wc -l /root/data/expenses.csv`, and three more |

**29 of 240 bash calls — 12% — reached outside the fixture; 20 scanned from `/`.**

**Why the model does it.** W04 and T04 deliberately give a wrong path in the prompt. When
`read` fails, the model falls back to searching the filesystem, and nothing bounds where. This
is a tool-recovery behaviour the tasks are designed to provoke, so it is not rare or incidental.

**What it cost.** The exposure was read-only, under the user's own account, into a process the
user launched — no worse than the access pi already has when run normally, and nothing was
exfiltrated (the agent has no network path out). Two real costs, though: a `find /` traverses
the whole disk until pi's 30-second tool timeout kills it, which plausibly accounts for several
of the 600-second run timeouts; and file names from the host enter the model's context, which
no assertion reads but which leaves the run not fully specified.

**Resolution.** Recorded in `benchmark.md` §4.6: execution moves into a pinned Linux container
mounting only the fixture copy, so reads outside it are impossible rather than merely
discouraged. The 29 commands above are replayed as a regression gate.

**Evidence:** `results/{LFM-G8,LFM-M8}-8192/transcripts/*.json`, the `v4` pi transcripts.

---

## 2026-08-31 — pi injects the fixture's `AGENTS.md` into its system prompt on every run

**What happens.** pi discovers context files and embeds them in the system prompt, so they never
appear in the message log. Read from pi 0.84.4's own source rather than inferred:
`dist/core/resource-loader.js`'s `loadProjectContextFiles` takes the first match of
`AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, `CLAUDE.MD` from the agent
directory, then walks from `cwd` upward taking one per directory;
`dist/core/system-prompt.js` assembles base prompt → appended text → `<project_context>` →
skills → cwd.

The `pi` driver sets `cwd` to the fixture copy, whose root holds the deliberately adversarial
`AGENTS.md`. It is therefore loaded on **every** run, for all 20 tasks.

**The isolation that does hold.** The global slot reads from `PI_CODING_AGENT_DIR`, which the
driver points at `setup/pi_config/` — a directory containing no context file. The operator's
own `~/.claude/CLAUDE.md` is not reachable: that path is neither the agent directory nor an
ancestor of the temp fixture tree. Only the fixture's own file is picked up.

**Why it is left enabled.** Loading `AGENTS.md` is what a production harness does, and it makes
W07 and T07's instruction conflict live rather than contingent on the model choosing to read the
file. Under `native` the same file reaches the model only through a tool call, so a model that
never opens it passes those tasks without ever meeting the conflict.

**Consequence.** W07 and T07 are not comparable across drivers, even after the `extra_rules`
fix. Recorded in §4.1; `environment.json` carries `context_files_discovered` for `pi` sessions.

**Related defect, not a finding.** Until `PiDriver.DRIVER_VERSION` `2`, pi received the
adversarial `AGENTS.md` but *not* the task's `extra_rules`, so those two tasks were graded
against rules the model was never given. That is our defect, recorded in
`implementation-plan.md`'s table, and it invalidated the `v4` pi W07/T07 records.

---

## 2026-08-31 — pi's `--thinking` is inert for a local LM Studio model

**What happens.** `dist/core/agent-session.js` clamps the requested level through
`getSupportedThinkingLevels`, which returns `["off"]` for any model whose catalogue entry does
not declare `reasoning`. `setup/pi_config/models.json` declares the LM Studio provider's model
as `{"id": "bench"}` with no such flag, so every thinking level clamps to `off`.

**Why it matters.** LFM2.5 still emits `reasoning_content` under pi — its transcripts carry
`thinking` blocks with `thinkingSignature: "reasoning_content"`. That reasoning is the model's
own behaviour, not something pi requested. So thinking level is neither a lever we can pull nor
a drift vector to guard against here, and §4.1 does not pin it. `environment.json` records the
resolved value (`off`) for completeness.

---

## 2026-08-30 — pi's own permission extension does not contain its `bash` tool

**What happened.** While building the `pi` driver (§4.1, M7), a research spike tested whether
`pi-permission-system`'s `special.external_directory: deny` rule keeps pi's tools inside a given
working directory. It does for `read`, `write`, `edit`, `find`, `grep` and `ls`: a `read` call
for a path outside the working directory was refused. It does not for `bash`: with
`bash: allow`, pi ran `cat /etc/hosts` through its `bash` tool and the real file's contents came
back — no denial, no prompt.

**Mechanism.** The extension's own documentation is explicit about the boundary: the
`external_directory` guard is implemented for path-bearing tool arguments, and `bash` commands
take an opaque string, not a structured path, so nothing extracts a path from them to check.
The extension does offer bash command wildcard patterns (e.g. `"rm -rf *": "deny"`), but these
match the raw, unparsed command string — the same class of defect as this project's own
`run_command` path-escape bug (`implementation-plan.md`, the `_escapes_sandbox` fix), just in
someone else's tool instead of this one.

**Resolution.** A hand-written macOS Seatbelt profile (`sandbox-exec`), generated per run by
`harness/driver_pi.py` and wrapped around every `pi` invocation, denies all filesystem writes
outside the fixture copy, the isolated pi config directory, and an isolated temp directory.
Verified directly: a write attempted from inside pi's `bash` tool to a path outside those three
failed with `Operation not permitted`, while writes inside the allowed paths succeeded and the
LM Studio round-trip through the model worked normally under the sandbox. This is kernel
enforcement, not string matching, and needed no new dependency (`sandbox-exec` is Apple's own
tool). A third-party extension, `@erichll/pi-sandbox`, offers the same kind of OS-level
containment as an installable package; it was considered and set aside for this round — it
needs a second package (`pi-auto-review`) and its own verification pass, and was one day old at
the time.

**Residual gap, accepted rather than fixed.** The Seatbelt profile only denies *writes*. A
`bash` command reading a path outside the working directory (`cat /etc/hosts`) still succeeds.
Real Stage 2A data confirms this is exercised, not merely possible: in
`results/LFM-G8-8192/transcripts/LFM-G8-W-W04-r1.json` the model ran
`find / -name "expense.csv" -type f` through pi's `bash` tool after a failed `read`, and the
call ended on pi's own 30-second timeout rather than on any containment boundary.
Denying `bash` outright would close this, but at a real cost: Suite T's T03 and T09 expect the
agent to self-verify a fix by running the test suite, and `pi-permission-system`'s own guard
already covers reads through pi's other tools. The read-side gap is a lower-severity exposure
than the write/destroy risk the Seatbelt profile closes, and no worse than the read access pi's
host process already has under the account it runs as.

---

## 2026-08-30 — Ternary-Bonsai-8B fails Stage 2A on a specific pattern, not on tool calling

**What happened.** Both Bonsai configurations (BON-M2, BON-G2) scored 0/10 on Suite W in Stage
2A. Neither made an invalid tool call across all 60 runs, and both passed Stage 0 (9/9).

**Mechanism**, from the transcripts:

- Search patterns are too literal. W01: `search_files(pattern="per-diem cap international
  travel")` — the whole question used as the search string, not a short distinctive phrase.
  Same shape on W08. Neither matches real file content.
- Most runs stop after one failed attempt. 21 of 30 runs, in both configs, make exactly one
  tool call, get a miss or an error, and stop. W04 shows this clearly: the prompt names the
  wrong file on purpose (`data/expense.csv`; the real file is `data/expenseS.csv`). Bonsai
  tries the literal path, gets "no such file," and never tries `list_files` to find the real
  one — the oracle's own next step at that point.
- `run_command` is never used when it's needed. W07 requires a row count past the
  4000-character truncation (§4.7); the oracle solves it with `wc -l`. Bonsai read the
  truncated file twice, tried to invent a `count_lines` tool (correctly refused as unknown),
  then wrote `1889` to the output file — not derived from anything visible in the transcript.

**Analysis.** Tool calling itself is not the problem: zero invalid calls, and W07's transcript
shows a real multi-step sequence, including recovery from a hallucinated tool. The gap is
specific — query formulation, retrying after a failed first attempt, and recognising when
`run_command` is required instead of `read_file`. This is what §1.2's progress score exists to
show: a weak model, not a broken harness.

Specific to this harness: no query hints, no retry assistance, no other agent framework's
prompting. §1.1 already states results do not transfer to a different harness or prompting
style.

- Evidence: `results/{BON-M2,BON-G2}-8192/archive/stage2a-v3.jsonl` and the matching
  transcripts, particularly W01/W04/W08 (single failed attempt) and BON-G2's W07 (8 tool calls,
  hallucinated tool, fabricated answer). Collected under `v3`; archived when `v4` superseded it,
  and not yet re-collected for either of these configurations.

---

## 2026-08-30 — A second, unexplained `server_error` pattern

**What happened.** In the full six-configuration Stage 2A rebuild (`v3`, after the sandbox fix
below), four of six configurations hit `server_error` on exactly one task each, all 3
repetitions, 0 elsewhere: LFM-M8 and LFM-BF16 on W03, LFM-G8 on W02, BON-M2 on W07. LFM-GQ4 and
BON-G2 had none.

**Not the Engine Protocol bug.** That bug always left a matching `ERROR` line in LM Studio's
server log (`Engine protocol predict stream returned an error`, with the "Invalid diff" message).
None of these four crashes have any matching log entry — `grep -n "ERROR" ~/.lmstudio/
server-logs/2026-08/2026-08-30.1.log` finds nothing for any of them. That rules out a server-side
500 of the same shape. The transcripts leading up to each crash are unremarkable — no absolute
paths, no unusually long arguments, nothing the sandbox fix (below) would have refused.

**Root cause unknown.** Deterministic per (config, task) at `temperature=0`, but no server-side
trace to explain it — client-side connection issue, or a different failure inside the same
`openai.APIError` family, are both plausible and neither is confirmed. Not investigated further
this session: the harness already handles it correctly (`server_error`, §4.8) — it recorded each
occurrence and moved on rather than losing the run or the stage. The cost is three lost
repetitions per affected task, not a crash.

- Evidence: `results/{LFM-M8,LFM-G8,LFM-BF16,BON-M2}-8192/archive/stage2a-v3.jsonl`, plus the
  matching transcripts. The `v4` re-collection reproduced the same pattern for LFM-M8 and
  LFM-G8 (`raw/stage2a.jsonl`, 3 `server_error` runs each).

---

## 2026-08-29 — LM Studio's Engine Protocol runtime corrupts streamed tool-call arguments

**What happened.** A live Stage 2A run against LFM-GQ4 crashed on W02's third repetition.
LM Studio returned a 500 mid-stream: `"Invalid diff: '...' not found at start of '...'"`. The
crash was deterministic: the same message, on the same three tasks (W02, W03, W09), 3/3
repetitions each — 9 of 30 Stage 2A runs. Each of these tasks needs a `run_command` argument
with a Python one-liner using a single-quoted `open(...)` call. LM Studio's server log named the
subsystem: `Engine protocol predict stream returned an error`.

**Root cause.** `~/.lmstudio/settings.json` had `useLlamaCppEngineProtocolRuntime3: true`. This
is a known bug, [lmstudio-ai/lmstudio-bug-tracker#1922](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1922),
in the same Engine Protocol runtime subsystem: it corrupts streamed tool-call arguments. The
message there differs ("Unrepairable tool_call arguments... replaced with empty object") but
the code path and failure family are the same. The documented fix is to disable the setting
(LM Studio → Settings → Developer).

**Resolution.** The user disabled the setting. Stage 2A was re-run for LFM-GQ4 from scratch
rather than resumed, since resuming would have mixed runs from two different environments in
one file. Zero `server_error` in the clean run. The three affected tasks now fail with
`empty_answer` instead of crashing the backend — see the next entry.

- Pre-fix run (archived, excluded from reporting): `results/LFM-GQ4-8192/archive/stage2a-engine-protocol-runtime-bug.jsonl`
- Clean re-run: `results/LFM-GQ4-8192/archive/stage2a-v3.jsonl`

**Analysis.** The setting is machine-wide, not per-configuration. It was on for every run before
this was found, including all six configurations' Stage 0 data. Stage 0's tasks are single tool
calls with simple arguments, so they are unlikely to trigger this bug, but if `server_error`
appears in any future stage, check this setting first. `environment.json` does not record it;
add it as a field if this recurs.

A second bug turned up while archiving the pre-fix run: `harness/report.py`'s
`load_all_records` reads every file under a session's `raw/` directory with no name filter.
Renaming the pre-fix run within `raw/` pooled it straight back into the report, doubling the run
count (24/60 instead of 12/30). Fixed by moving the archive to a sibling `archive/` directory.
`report.py`'s docstring now documents the convention: everything in `raw/` is reported, so
anything that should not be belongs elsewhere.

---

## 2026-08-29/30 — LFM2.5 never tries the one permitted `run_command` command

**Confirmed across all four LFM configurations**, under `task_set_version: v3` (the `v2` run
this was first observed in was affected by the `run_command` sandbox bug — see
`implementation-plan.md`'s defect table — and has been superseded; the pattern below held up in
the clean re-run). Checked directly: LFM-GQ4, LFM-M8, LFM-G8 and LFM-BF16 each tried `cd`,
`python3`, `awk`, `pwd`, `ls` or `echo` on their `run_command` calls for W02/W03/W09 — never
plain `python`, in any of the four. Same base model, four quantisations, same blind spot.

**What happened.** In each configuration's Stage 2A run, LFM2.5 passed the tasks solvable by
retrieval alone (`read_file`/`list_files`/`search_files`: W01, W04, W05, W08) and failed the
tasks that need computed aggregation over CSV data (W02, W03, W09, W10).

**Mechanism.** The transcripts show the model trying `run_command` with `cd /workspace &&
python3 -c "..."`, refused: `exit=127 command not permitted: cd`. It retries with `python3 -c
"..."` alone, refused again: `command not permitted: python3`. In one case it then tries `awk`,
also refused; in another its command has unmatched quoting and fails to parse. It never tries
plain `python`, the one binary on the §4.6 allowlist (`ls cat grep find head tail wc python`, no
`cd`) — the same command the oracle uses to solve these tasks. After the refusals it falls back
to reading the raw CSV directly, then its final turn produces no tool call and no content:
`empty_answer`.

**Analysis.** This is not a harness defect. The allowlist is deliberate (§4.6, protected by
§11), and the system prompt gives no hint about which commands are permitted or what path to
use — recovering from an unhinted refusal is the capability under test, and §4.5's no-repair
rule exists so the harness does not do that recovery for the model. LFM-GQ4 defaults to
`python3`/`cd`, common conventions elsewhere, and does not converge on `python` after two
refusals, nor does it compute the answer itself once tool use stalls.

Since this holds across all four quantisations, it reads as a property of LFM2.5-2.6B's
training, not of any one artefact — worth stating as a finding about the model, not about a
configuration.

- Evidence: `results/{LFM-GQ4,LFM-M8,LFM-G8,LFM-BF16}-8192/archive/stage2a-v3.jsonl` (task_id
  W02/W03/W09/W10) and the matching transcripts.

---

## 2026-08-27 — Reconnaissance (`task_set_version: v2`)

**Not benchmark data.** One shot per task, one context, no environment capture, no repetitions.
Nothing here was quoted as a result or written to `results/`. It existed to de-risk M3-M5 before
they were built, and is kept for the record now that they are.

### Artefact mapping — all six §2 rows have a working artefact

Identified by path, which is stable. The key column was a snapshot of what LM Studio derived
from the installed set at the time, and moves as models are installed or removed (§2.1).

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

- Zero invalid tool calls anywhere, in either version. Tool-call formatting is not the
  bottleneck for any configuration, which is not what §1.2 predicted. Stage 0 as specified gates
  nothing in this set, and was later confirmed live against all six (see the M4 notes in
  `implementation-plan.md`); kept as a pre-flight check regardless.
- T01 now passes on all four LFM configurations, in 3 tool calls each. Under v1 every one of
  them failed it. That was the harness, not the models.
- Bonsai is unchanged at 0/3, and its failure mode never touched path handling: 0-1 tool calls,
  then an answer from parametric knowledge without exploring. Both builds return byte-identical
  answers, a useful determinism signal for the harness. Stage 0 would pass this behaviour.
- Reasoning share is a per-model property: LFM 60-91%, Bonsai 0%. It drives agent latency more
  than raw tok/s and belongs in the §10.3 table.
- Per-request overhead is about 10x higher on MLX (255-468 ms) than llama.cpp (34-44 ms). Since
  `prompt_tps` subtracts `overhead_median`, this bears directly on the runtime question.
- LFM-G8 on W01 was the one remaining LFM failure at the time: 33 calls, `empty_answer`, where
  LFM-GQ4 did it in 6. Single-shot, so indistinguishable from variance at the time — worth
  revisiting once real Stage 2A data exists for LFM-G8.

### Run-to-run variance at `temperature=0` is real

T02 on LFM-M8, varying only `max_tokens`: pass at 1024, fail at 2048, pass at 4096. A budget
effect cannot produce that shape. This is the near-but-not-bitwise determinism §4.2 warns about,
showing up as an outcome flip. It was the first hard evidence that §9.1's three repetitions and
the `flaky` flag are needed: a single run of this task can report either outcome.

### Harness defects found and fixed

Filed here, not in `implementation-plan.md`, because this predates the milestone structure that
document tracks defects against. Found across M1-M3, in one session, before the harness could
run a real stage.

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
