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

## 2026-09-03 — the artefacts disagree about their own recommended sampling

**What happened.** Preparing Stage 5B's sampling pass meant answering what "the model's
recommended defaults" are. They are stated by the artefacts themselves, and the four LFM
artefacts of one model do not agree:

| Configuration | Source | Stated |
|---|---|---|
| LFM-G8 (Q8_0) | GGUF header, `general.sampling.*` | `temperature 0.1`, `top_k 50` |
| LFM-GQ4 (QAD-Q4_0) | GGUF header | **`temperature 0.2`, `top_k 80`** |
| LFM-M8, LFM-BF16 (MLX) | `generation_config.json` | `temperature 0.1`, `top_k 50`, `repetition_penalty 1.1` |
| BON-G2 | GGUF header | `temperature 0.5`, `top_k 20`, `top_p 0.85` |
| BON-M2 | — | **nothing; no `generation_config.json` ships with the artefact** |

**Why it matters.** "Run the model at its recommended defaults" is not one setting per model. It
is one per *artefact*, and the QAD quantisation asks for twice the temperature and a wider `top_k`
than the Q8_0 build of the same weights — plausibly because quantisation-aware distillation
changes the output distribution, though nothing here tests that. Resolving the defaults once per
model and sharing them would have run three of the four LFM configurations at settings their own
artefacts do not recommend.

It also means the sampling pass is not available for every configuration: BON-M2 states nothing,
so asking for one fails rather than inventing values. That is the right outcome — but it makes
the Bonsai pair asymmetric if the pass is ever used to compare them.

**Also recorded:** neither MLX artefact states `top_p`, and neither GGUF states a repeat penalty.
Unstated parameters keep their §4.2 controlled value, and every session records which keys the
artefact actually supplied.

**Related harness defect, found while reading this.** `harness/gguf_meta.py` decoded float32
metadata with `"<I"`, so `general.sampling.temp` came back as `1036831949` rather than `0.1`.
Invisible while the reader was only asked for integer geometry (§2.2); wrong the moment anything
read a real float. Fixed, with a second fix alongside it: `parse()` accepted a metadata layout on
the strength of `general.architecture` alone, which the *wrong* layout can decode correctly before
walking off into padding — it now requires every declared key to be present and non-empty, which
is what its docstring always claimed.

**Evidence:** the artefacts under `~/.lmstudio/models/`, read by `harness/sampling.py`.

---

## 2026-09-03 — the `v7` campaign moved the instrument, not the models

**What was run.** The full six-configuration campaign at `v7`: Stage 0, Stage 1 (8K and 16K),
Stage 2A and Stage 2B under `pi`, plus the Stage 5A `native` arm for LFM-G8 and LFM-GQ4. 534
records, 6.86 h of run wall clock. Nothing in the `v7` bump touches inference — the changes are
two task wordings, one assertion, the change baseline and the pi driver's record-keeping — so
the campaign doubles as a replication of `v6`.

**It replicates.** Where nothing we changed applies, nothing moved: BON-M2 3/30, BON-G2 0/30,
LFM-G8 `native` 9/30, LFM-GQ4 `native` 12/30, all identical to `v6`. Stage 1 throughput matched
to two decimal places on all six configurations — LFM-GQ4 8K, for instance, TTFT 8.58 both
times, 769.2 prompt tok/s both times. The environment did not drift between campaigns, so
Suite W and T differences are attributable rather than ambient.

**The Suite T movement is entirely grading.** T05's assertion changed while its prompt did not,
so the `v6` answers can be re-graded under the `v7` check — the same text, judged both ways:
7/12 as graded, **12/12 re-graded**. The observed Suite T change across the four LFM
configurations is +5 in total (+2, 0, +3, 0). The grading fix accounts for all of it, and the
residual is zero. No Suite T improvement at `v7` should be read as models doing better.

**W07 now measures what it is specified to measure**, per the entry below:

| W07, `pi`, 12 runs | `v6` | `v7` |
|---|---|---|
| Passed | 2/12 | 8/12 |
| Failed `rowcount_correct` | 10 | 2 |
| Failed `british_spelling` | 2 | 2 |

The counting failures collapsed and the conflict arm held flat. Both surviving
`rowcount_correct` failures miscounted nothing — see the misdirected-write entry below.

**Evidence:** `results/*/raw/*.jsonl` (`v7`) against `results/*/archive/*-v6.jsonl`.

---

## 2026-09-03 — Suite W does not discriminate between the four LFM configurations

**What happened.** At `v7` all four LFM configurations scored **20/30** on Suite W. At `v6` they
spread 17-22, which read as a ranking and was quoted as one.

**It was never a ranking.** Testing `v6`'s four cells for homogeneity against their pooled rate
of 0.642: **χ² = 1.85 on 3 degrees of freedom, against a 5% critical value of 7.81.** The 17-22
spread is entirely consistent with four configurations of identical ability. The `v7` four-way
tie is not a coincidence needing explanation; it is what `v6` already implied.

The test overstates the available resolution, so the real position is weaker still: the 30 runs
per cell are 10 tasks × 3 repetitions, and repetitions of one task are correlated, so the
effective sample is nearer 10 than 30.

**What Suite W does separate.** The task-level structure is bimodal and stable across both
campaigns: W01, W04, W05, W08 pass 12/12 and W02, W03, W06 pass 1-3/12. Nothing sits in the
middle where discrimination would live. Suite W separates *models* — LFM from Bonsai, decisively
— and does not separate *quantisations of one model*.

**Consequence for §10.4.** No claim of the form "Q8_0 beats 8-bit MLX on agent task success" is
supported by either campaign. Question 3 (operating point) rests on latency and memory, which
are tight and reproducible; question 1 (runtime) rests on LFM-G8 vs LFM-M8's timing gap, not on
their identical 20/30. Reporting Suite W as a bare fraction invites the misreading — an interval
would not have.

**Evidence:** `results/LFM-*-8192/raw/stage2a-pi.jsonl` and the `v6` archives.

---

## 2026-09-03 — agents write beside the fixture root, and the mount lets them succeed

**What happened.** Two of the three remaining W07 failures at `v7` — LFM-G8 r2 and LFM-GQ4 r1 —
computed the row count correctly and then wrote it to `<workdir>/data/rowcount.txt` instead of
`<workdir>/root/data/rowcount.txt`. LFM-G8's run ran `mkdir -p` to create that directory, wrote
`120` into it four times, verified its own write by reading it back, and reported success. Its
final message is correct in every particular: "121 total lines (1 header + 120 data rows), so I
counted the data rows as 120".

**Why the write succeeds.** `container_session` bind-mounts the whole runs root, so everything
beside `root/` is writable. Removing the `pristine/` copy left `root/` alone in the work
directory, which is tidier but gives a stray write nothing to collide with. The agent gets no
error, cannot detect the mistake, and cannot recover from it.

**What it costs.** Two of twelve W07 runs were decided by this rather than by the task. Grading
is not wrong — the file genuinely is not where it was asked for — but the run measures path
handling instead of instruction adherence, which is the same class of problem the W07 wording
fix was for.

**Resolution.** Fixed at `v8`, and not by changing the mount: the container is started once per
stage and a run's directory does not exist yet at that point, so a per-run bind mount would mean
a container per run. Instead the work directory *around* `root/` is sealed to mode `0555` for the
duration of the run — the container executes as the harness's own uid, so a plain filesystem
permission is enough, and it is kernel-enforced rather than matched on a string. Both commands
from the runs above were replayed in the container and now fail with `Permission denied`, while
writes inside `root/` and every read still succeed (`tests/test_isolation.py`). The residual: the
runs root itself stays writable, because new work directories are created there while a stage is
in progress, so a write to `/runs/stray.txt` still succeeds. That is outside every graded tree and
is not the mistake the models make — they resolve output paths against their own parent.

**Evidence:** `results/LFM-G8-8192/transcripts/LFM-G8-pi-W-W07-r2.json`,
`results/LFM-GQ4-8192/transcripts/LFM-GQ4-pi-W-W07-r1.json`.

---

## 2026-09-03 — `swap_flag` is a machine-state reading, not a configuration property

**What happened.** Two swap flags flipped between campaigns, both towards not swapping:
LFM-M8 at 16K (`v6` yes, `v7` no) and BON-G2 at 8K (`v6` yes, `v7` no). Peak memory for the
same cells barely moved — BON-G2 8K is 5.56 GiB in both — so this is not a different allocation
pattern but a different machine state around it.

**What is stable.** The configurations under genuine pressure swap in both campaigns: BON-M2 on
both tiers (8.06 and 12.02 GiB) and BON-G2 at 16K (8.46 GiB). An 8B model at 16 GiB is the real
signal; the flips are the marginal cases either side of it.

**Consequence.** `swap_flag` feeds the report's Swap column and, through §2.2, the admissibility
story. A single campaign's flag is not evidence that a configuration does or does not swap —
only repeated agreement is. LFM-M8's `v6` swap at 16K should not be quoted as a property of that
configuration.

**Evidence:** `results/{LFM-M8,BON-G2}-{8192,16384}/raw/stage1.jsonl` against the `v6` archives.

---

## 2026-09-03 — the driver decides what a task measures, again

**What happened.** The W07 wording change moved the task from 2/12 to 8/12 under `pi` and had
**no effect at all** under `native`: LFM-G8 failed 3/3 in both campaigns, every run on
`rowcount_correct`. The runs did not miscount — they never wrote the file, terminating
`empty_answer` (`v7`) or `loop_detected` (`v6`) at progress 2.

The same thing happened to T02's prompt change. Under `pi` it took the task to 11/12; under
`native` LFM-G8 went 2/3 → 0/3 and LFM-GQ4 3/3 → 1/3, with every new failure an `empty_answer`
or `loop_detected` at progress 2. The added sentence was never reached.

**Why it matters.** A task-level fix can only be validated on a driver whose runs reach the
behaviour being fixed. `native` ends 30 of its 60 Suite W runs degenerately — half — so it cannot
serve as a control for a change to what a final answer must contain. This is the 2026-09-02 termination
finding arriving from the other direction: the same task, model and fixture measure instruction
adherence under one driver and termination under the other.

**Stage 5A replicates exactly.** Conditional on reaching a final answer, `native` still matches
or beats `pi` in every cell at `v7`: LFM-G8 Suite W 75% vs 74%, Suite T 100% vs 97%; LFM-GQ4
Suite W 80% vs 71%, Suite T 100% vs 93%. And `invalid_calls`, which only `native` can observe,
is 3 and 12 for LFM-G8's two suites, 0 and 9 for LFM-GQ4's.

**Evidence:** `results/{LFM-G8,LFM-GQ4}-8192/raw/stage2{a,b}{,-pi}.jsonl`, `v7`.

---

## 2026-09-02 — two Suite T tasks were failing correct answers

**T05 — the careful answer failed.** `_check_t05` compared the answer's basename set to the four
expected raisers exactly. Five `v6` runs (one per LFM configuration, plus a second LFM-G8 run)
named all four correctly and then added a note of the form "`ledger/validation.py` defines the
`ValidationError` class but does not raise it itself" — which put a fifth name in the set and
failed the run. W01 already had the pattern for this: naming the decoy is acceptable when the
answer identifies it as superseded. T05 had no equivalent.

Replaying the assertion over the `v6` answers: all twelve LFM runs pass under the `v7` check,
against seven before. **Consequence worth stating: T05 no longer discriminates between the LFM
configurations at all.** That is the correct outcome — the answers were right — but it removes a
task from the four that separated them, and Suite T is already near-saturated for LFM.

**T02 — the assertion asked for more than the prompt did.** `_check_t02` requires the answer to
name `posting.py`; the prompt said only "Explain what causes it". Two `v6` runs (LFM-BF16 r2,
LFM-M8 r2) diagnosed the fault exactly — float conversion, per-share rounding, 99.99 ≠ 100.00 —
naming `split_amount` but never the file, and failed. Checked directly: `posting.py` is absent
from the final message in both and present in every passing run.

Fixed by changing the prompt, not the assertion. Dropping the filename requirement was
considered and rejected: "float rounding" is a plausible guess about a splitting function, so an
answer that never locates the fault is not evidence of investigation — which is what §8's
negative control exists to detect.

**Why both were only visible now.** Neither is a model behaviour; both are grading defects that
a pass rate alone cannot show. They surfaced from reading the transcripts of failing runs, which
is the argument for doing that routinely rather than only when a number looks surprising.

**Evidence:** the `v6` T02 and T05 transcripts under `results/LFM-*-8192/transcripts/`.

---

## 2026-09-02 — W07 was measuring a counting convention, not instruction adherence

**What happened.** Across the `v6` pi runs, ten of twelve LFM runs wrote **121** to
`data/rowcount.txt` where the expected value is 120 — the file's line count including its
header. `condition_failures` names `rowcount_correct` as the only failing arm in eight of them.
`notes_untouched` failed in none. `british_spelling` failed twice, both LFM-M8, on "organized".
So the instruction conflict the task exists to measure was reached by almost nobody: the task
was decided on the header.

**Why it is the wording and not the models.** W04 expects the same 120 from the same file, and
every LFM configuration passes it 3/3 under `pi`. The counting method is identical in both
tasks — `wc -l` — but the correction is not:

| | W04 ("How many **expense rows** are recorded…") | W07, `v6` ("Count **the rows** in the expense register…") |
|---|---|---|
| LFM-G8 r1, r3 | `wc -l`, then `tail -n +2 \| wc -l` → 120 | `wc -l` → 121 |
| LFM-GQ4 r3 | `wc -l`, then `tail -n +2 \| wc -l` → 120 | `wc -l` → 121 |
| LFM-M8 r3 | `wc -l`, then `tail -n +2 \| wc -l` → 120 | `wc -l` → 121 |

The same model, on the same file, in the same session, corrects for the header when the noun is
"expense rows" and does not when it is "the rows". Only LFM-G8 r2 made the correction on W07,
and it passed.

**Resolution.** W07's prompt asks for the *data* rows from `v7` (§7.3). Recorded here as well as
in the spec because `way-of-working.md` warns that weakening a task after seeing results is
invisible in a diff: the change removes an ambiguity in the work half, leaves the conflict arms
untouched, and is defensible only because the W04 comparison shows the models can do the
counting when asked unambiguously. The `v6` W07 numbers should not be read as adherence data.

**Evidence:** the `v6` Stage 2A records — `results/LFM-*-8192/raw/stage2a-pi.jsonl` until the
`v7` archive step moves them under `archive/` — read for `condition_failures`, and the matching
W04/W07 transcripts.

---

## 2026-09-02 — the change baseline sat inside the agent's reach

**What happened.** `prepared()` copied the fixture to `<workdir>/pristine` beside the working
copy at `<workdir>/root`, and `container_session` bind-mounts the entire runs root into the
container. The reference tree the grader compares against was therefore readable by the agent
being graded. **25 of 291 `v6` pi runs (8.6%) touched it.** Every reference is a read — no run
wrote into it, so no `v6` verdict is affected.

**Why it mattered anyway.** `changed_paths` compares the two trees, so a write into `pristine`
would have made a modified file look unchanged, or manufactured a change that never happened —
silently defeating `unchanged_under` (W07, T07) and `only_changed` (T03), the three tasks whose
verdict *is* the tree comparison. It also cost runs: several LFM-GQ4 W02 runs spent a dozen
turns establishing which copy of `fx_rates.yaml` was authoritative, and LFM-BF16 passed W06 by
computing over `pristine/data/expenses.csv`.

**Resolution.** The baseline is a map of content hashes taken before the driver runs; the second
tree is gone (§6.3). Nothing ever read its bytes — `changed_paths` was its only consumer.
Guarded by `tests/test_harness.py`, which asserts a run directory holds nothing but `root`.

**Not fixed, and deliberately:** the mount is still the runs root rather than one run's
directory. `prepared()` removes each work directory as its run ends, so a run sees no other
run's tree; per-run containers would buy isolation that cleanup already provides.

**Evidence:** the `v6` pi transcripts; `grep pristine` across `results/*/transcripts/*-pi-*.json`.

---

## 2026-09-02 — a timed-out pi run recorded nothing about itself

**What happened.** Nine `v6` pi runs hit the 600 s wall clock — LFM-BF16 ×4, LFM-GQ4 ×3,
LFM-M8 ×2, all on W02/W06/W07/W10/T03. Every one recorded `tool_calls: 0`, `progress_score: 0`
and no transcript, despite step counts of 20-59.

**Mechanism.** `pi` streams JSONL events, but the message log arrives only in the terminal
`agent_end` event, which a killed process never emits. `driver_pi.py` read the transcript from
that field alone, so `_calls_from_transcript(None)` returned no calls and the progress score had
nothing to score. `steps` survived only because it counts `turn_start` events already on the
stream — hence records showing 59 steps and zero tool calls.

**What the surviving fields show.** Token totals come from `message_end` and did survive: the
LFM-GQ4 W02 timeouts accumulated 288k and 260k prompt tokens over 54-59 turns. These are
genuine thrash loops, not stalls — which is exactly why losing their transcripts was expensive.

**Resolution.** From `v7`, the log is accumulated from `message_end` events as they stream
(`PiDriver.DRIVER_VERSION` `4`, §5.3). Verified against pi 0.84.4's
`dist/core/agent-session.js`, where `message_end` fires once per settled message and carries the
whole message object for `user`, `assistant` and `toolResult` roles alike — the same objects
`agent_end` would have returned. The answer is still taken only from a settled run, so a
timeout's verdict does not change; what changes is that its transcript survives.

**Evidence:** the `v6` agent records for LFM-BF16, LFM-GQ4 and LFM-M8 (`stage2{a,b}-pi.jsonl`),
filtered on `termination_reason: "timeout"`.

---

## 2026-09-02 — pi's advantage over the bare loop is termination, not capability

**The comparison.** Stage 5A, `v6`, LFM-G8 and LFM-GQ4. Same task set, same container image,
same fixtures, same model artefacts, same §4.2 sampling. **Only the driver varies.** This is the
first controlled driver comparison the project has run; earlier attempts compared `v3` `native`
against `v4`/`v6` `pi` and were worth nothing as evidence.

Headline scores look like a large scaffolding effect:

| | Suite W | Suite T |
|---|---|---|
| LFM-G8 | pi 22/30, native 9/30 (**+13**) | pi 27/30, native 20/30 (**+7**) |
| LFM-GQ4 | pi 19/30, native 12/30 (**+7**) | pi 26/30, native 27/30 (**−1**) |

The effect is not uniform, and the −1 is the clue: on one cell the bare loop *wins*.

**Where the difference actually is.** Under `native`, runs pile up at progress 2 — the model
read or searched the correct target and then produced no answer, terminating `empty_answer` or
`loop_detected`. 18 of 30 Suite W runs for LFM-G8, 15 of 30 for LFM-GQ4. Under `pi`, 0 and 1.

**Conditional on reaching a final answer, `native` matches or beats `pi` in every cell:**

| | pi | native |
|---|---|---|
| LFM-G8 Suite W | 22/30 = 73% | 9/12 = **75%** |
| LFM-G8 Suite T | 27/30 = 90% | 20/20 = **100%** |
| LFM-GQ4 Suite W | 19/28 = 68% | 12/18 = 67% |
| LFM-GQ4 Suite T | 26/28 = 93% | 27/27 = **100%** |

So the production harness is not improving retrieval, reasoning or tool use. It is converting a
completed investigation into a final message — the one thing §4.5 forbids the `native` loop from
helping with, and the thing these models cannot reliably do alone.

**Why this matters for reading the whole benchmark.** A `pi` score is a capability measurement
with a termination floor underneath it. A `native` score conflates capability with a failure to
stop. Neither is wrong; they answer different questions, and §4.1 is right that the driver is
part of a run's identity.

**The measurement only `native` can make.** `invalid_calls`: 3 and 15 for LFM-G8's two suites, 0
and 6 for LFM-GQ4. `pi` reports 0 everywhere because it repairs internally (§5.3) — these are
the malformed calls its zeros were not observing. This is the concrete case for keeping `native`
after `pi` became the controlled comparison.

**Evidence:** `results/{LFM-G8,LFM-GQ4}-8192/raw/stage2{a,b}{,-pi}.jsonl`, all `v6`.

---

## 2026-09-02 — Ternary-Bonsai-8B's failure is capability, and now attributable

**What happened.** Both Bonsai configurations failed the Stage 2A gate under `pi` at `v6`:
BON-M2 1/10 tasks (3/30 runs), BON-G2 0/10 (0/30). Suite T was not attempted — the gate stopped
both, as §9 specifies.

**Why this is attributable where the `v3` finding was not.** The 2026-08-30 entry recorded the
mechanism as "most runs stop after one failed attempt" under the `native` driver, which left open
whether Bonsai was weak or merely under-scaffolded. This run answers it: under `pi`, **both
configurations terminated `final_answer` on 30/30 runs**, with tool calls in 25/30 and 21/30. The
scaffolding worked. The scores did not move.

Read alongside the Stage 5A entry above — where `pi`'s whole advantage turns out to be
termination — this separates cleanly. Bonsai had no termination problem to fix, so there was
nothing for the scaffolding to buy. Progress piles at level 1 (16 and 14 runs): a valid tool
call, and no further.

**Two configurations, opposite directions.** MLX 2-bit and llama.cpp `Q2_0_g64` land in the same
place, so this is the model rather than a runtime or quantisation artefact.

**Also recorded:** both swap on this machine and show TTFT around 27 s against LFM's 8-15 s. An
8B model at 8192 context is at the edge of a 16 GiB machine regardless of how it scores.

**Relationship to the `v3` entry.** That entry is not superseded — its transcript-level mechanism
(whole questions as search strings, `run_command` unused where truncation requires it) still
stands and is more specific than anything here. What is added is attribution: the failure
survives a production harness.

**Evidence:** `results/{BON-M2,BON-G2}-8192/raw/stage2a-pi.jsonl`, `v6`.

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

**Resolution.** Recorded in `benchmark.md` §4.6: execution moved into a pinned Linux container
mounting only the fixture copy, so reads outside it are impossible rather than merely
discouraged. The commands above are replayed as a regression gate
(`tests/test_isolation.py`). Two side effects worth noting: `find /` now *completes* instead of
dying on pi's 30-second tool timeout, recovering wall clock those runs were losing; and the
macOS Seatbelt profile it replaced has been deleted, since maintaining two containment
mechanisms with different semantics is worse than one.

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

**Superseded on 2026-09-01.** pi now runs inside the §4.6 container, so both the write
confinement described below and the read gap it left are replaced by one boundary. The entry is
kept because it is the record of why the Seatbelt profile existed and what it did not cover.

**Residual gap, accepted rather than fixed** (as it stood before the container). The Seatbelt
profile only denies *writes*. A
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
