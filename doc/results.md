# Results

Where the benchmark stands after the `v7` campaign of 2026-09-03, and what the data does and
does not support. Written as a starting point for a write-up, not as a specification.

This document interprets. The authorities it draws on are elsewhere and are not restated here:

- [`benchmark.md`](benchmark.md) — what is measured and how
- [`findings.md`](findings.md) — the dated empirical log each claim below cites
- [`report-v7.md`](report-v7.md), [`report-v6.md`](report-v6.md) — the generated tables
- `results/*/raw/*.jsonl` — the record; every figure here regenerates from it

---

## 1. What has been run

| Campaign | Date | Scope | Cost |
|---|---|---|---|
| `v6` | 2026-09-02 | Six configurations, Stage 0/1/2A/2B under `pi`; `native` arm for LFM-G8 and LFM-GQ4 | 7.13 h |
| `v7` | 2026-09-03 | The same scope, after five changes to the instrument | 6.86 h |

Both at 8192 context for the agent stages, both on one machine, both with 534 records. `v7` is
the current data. `v6` is archived under `results/*/archive/` and is not comparable cell-by-cell
(§4.1, §11) — but because nothing in the `v7` bump touches inference, the two campaigns together
work as a replication, which §7 below uses.

Not yet run: Stage 3 (both suites at 16K), Stage 5B's compaction experiment, and Stage 5B's
recommended-default sampling pass.

## 2. Headline results (`v7`)

| Configuration | Driver | Suite W | Suite T | TTFT (s) | Gen tok/s | Peak RAM | Swap | Verdict |
|---|---|---:|---:|---:|---:|---|---|---|
| LFM-GQ4 | pi | 20/30 | 26/30 | 8.59 | **83.0** | **2.33 GiB** | no | proceeded to 2B |
| LFM-G8 | pi | 20/30 | **29/30** | 8.62 | 52.9 | 3.52 GiB | no | proceeded to 2B |
| LFM-M8 | pi | 20/30 | 28/30 | 14.55 | 50.2 | 5.86 GiB | no | proceeded to 2B |
| LFM-BF16 | pi | 20/30 | 26/30 | 8.99 | 28.8 | 7.65 GiB | no | proceeded to 2B |
| BON-M2 | pi | 3/30 | — | 27.43 | 34.6 | 10.02 GiB | yes | excluded at the Stage 2A gate |
| BON-G2 | pi | 0/30 | — | 27.57 | 27.2 | 7.60 GiB | no | excluded at the Stage 2A gate |
| LFM-G8 | native | 9/30 | 18/30 | — | — | — | — | Stage 5A arm |
| LFM-GQ4 | native | 12/30 | 25/30 | — | — | — | — | Stage 5A arm |

Stage 1, both context tiers:

| Configuration | TTFT 8K | Gen 8K | Prompt 8K | Peak 8K | Swap | TTFT 16K | Gen 16K | Peak 16K | Swap |
|---|---:|---:|---:|---|---|---:|---:|---|---|
| LFM-GQ4 | 8.58 | 83.0 | 769 | 2.09 GiB | no | 18.29 | 78.3 | 2.43 GiB | no |
| LFM-G8 | 8.62 | 52.9 | 770 | 3.28 GiB | no | 18.36 | 50.9 | 3.62 GiB | no |
| LFM-M8 | 14.52 | 50.2 | 457 | 5.15 GiB | no | 29.85 | 45.7 | 5.61 GiB | no |
| LFM-BF16 | 8.97 | 28.8 | 760 | 7.50 GiB | no | 18.53 | 27.4 | 7.95 GiB | no |
| BON-M2 | 27.45 | 34.6 | 235 | 8.06 GiB | yes | 62.95 | 25.7 | 12.02 GiB | yes |
| BON-G2 | 27.55 | 27.2 | 234 | 5.56 GiB | no | 66.19 | 23.1 | 8.46 GiB | yes |

## 3. The three questions

### Question 2 — the model: answered, decisively

LFM2.5-2.6B solves the task set; Ternary-Bonsai-8B does not. Bonsai scored 3/30 and 0/30 on
Suite W and was gate-stopped before Suite T in both campaigns, having passed exactly one task
(W05) across 60 runs of each configuration.

The failure is capability, and it is now attributable. Under `pi`, both Bonsai configurations
terminated `final_answer` on 30/30 runs with tool calls in most of them: the scaffolding worked
and the scores did not move. Progress piles at level 1 — one valid tool call, then nothing.
Stage 0 passes 9/9, so tool-call formatting is not the bottleneck either. Two runtimes and two
quantisations land in the same place, so it is the model rather than an artefact.

Confidence: high. Two independent campaigns, two runtimes, 120 agent runs.

### Question 3 — the operating point: answered on latency and memory, not on task success

**LFM-GQ4 (QAD Q4_0) is the operating point the data supports.** It matches every other LFM
configuration on task success while running 1.6× faster than Q8_0, 2.9× faster than BF16, and in
under a third of BF16's memory (2.09 against 7.50 GiB at 8K). Quantisation to 4 bits costs nothing measurable here on agent task
success, which is the interesting result.

The qualification matters as much as the conclusion: **Suite W cannot separate the four LFM
configurations at all.** All four scored 20/30 at `v7`; at `v6` they spread 17–22, which read as
a ranking and was quoted as one. Testing `v6`'s cells for homogeneity gives χ² = 1.85 on 3
degrees of freedom against a 5% critical value of 7.81 — the spread is what four identical
configurations would produce. The `v7` tie is not a coincidence; it is what `v6` already implied.
Suite T is no better placed: its only movement between campaigns was a grading fix (§7).

So the recommendation rests on the timing and memory columns, which are tight, reproducible to
two decimal places across campaigns, and separated by factors rather than by a few runs.

Confidence: high for the latency/memory basis; the "no quality cost" half is an absence of
evidence for a difference, not evidence of no difference — see §8.

### Question 1 — the runtime: llama.cpp ahead, on speed and memory only

At matched 8-bit quantisation, LFM-G8 (llama.cpp/Metal) against LFM-M8 (MLX):

| | LFM-G8 | LFM-M8 |
|---|---|---|
| TTFT | 8.62 s | 14.55 s |
| Generation | 52.9 tok/s | 50.2 tok/s |
| Prompt | 770 tok/s | 457 tok/s |
| Peak RAM | 3.28 GiB | 5.15 GiB |
| Suite W / T | 20/30, 29/30 | 20/30, 28/30 |

llama.cpp wins TTFT by 1.7×, prompt throughput by 1.7×, and memory by 1.9 GiB, on the same
weights at the same quantisation. The task-success columns are a tie and, per §3 above, could not
have shown otherwise.

The mechanism behind the prompt-throughput gap was measured during reconnaissance and is
per-request overhead: roughly 255–468 ms on MLX against 34–44 ms on llama.cpp
(`findings.md`, 2026-08-27). Since `prompt_tps` subtracts the overhead median, this bears
directly on the comparison rather than being an artefact of it.

Confidence: high on the numbers, narrow in scope — one model family, one machine, two context
tiers.

## 4. What the task suites actually discriminate

Suite W (`pi`, `v7`), passes out of 3 repetitions:

| Task | Category | G8 | GQ4 | M8 | BF16 | Total |
|---|---|---|---|---|---|---|
| W01 | retrieval | 3 | 3 | 3 | 3 | 12/12 |
| W04 | tool-recovery | 3 | 3 | 3 | 3 | 12/12 |
| W05 | search | 3 | 3 | 3 | 3 | 12/12 |
| W08 | long-context | 3 | 3 | 3 | 3 | 12/12 |
| W10 | state-retention | 3 | 3 | 2 | 2 | 10/12 |
| W07 | instruction-adherence | 2 | 2 | 2 | 2 | 8/12 |
| W09 | conflict-resolution | 1 | 2 | 2 | 3 | 8/12 |
| W03 | extraction | 2 | 0 | 1 | 0 | 3/12 |
| W02 | aggregation | 0 | 0 | 1 | 1 | 2/12 |
| W06 | multi-hop | 0 | 1 | 0 | 0 | 1/12 |

The distribution is bimodal: four tasks at ceiling, three near the floor, and almost nothing in
between. That shape is why the suite separates models decisively and quantisations not at all —
discrimination requires tasks in the middle, and there are three.

The floor tasks share a mechanism, documented across both campaigns: they need computed
aggregation over a 120-row CSV. Failures are hand-arithmetic (dropped rows, transcription
errors), FX conversion applied in the wrong direction against an explicit instruction to
multiply, or invented conversion rates hard-coded into otherwise correct Python.

The split between computing and narrating is the strongest single predictor of success on these
tasks, and it reproduces across campaigns. Over W02, W03, W06 and W10 under `pi`: at `v7`, runs
that invoked `python` passed **14/22** while runs that reasoned in prose passed **2/26**; at
`v6`, 13/28 against 3/20.

Suite T is near-saturated for LFM: T01, T04, T08, T09, T10 pass 12/12, and only T03 (8/12),
T05 (9/12) and T07 (10/12) discriminate. T03 is the least stable task in the set — LFM-M8 went
0/3 to 3/3 between campaigns with no relevant change to the instrument.

## 5. Drivers: the scaffolding buys termination, not capability

The Stage 5A comparison (`v7`, same task set, fixtures, container and sampling; only the driver
varies) reproduces `v6` exactly.

Headline scores look like a large scaffolding effect: LFM-G8 20/30 against 9/30 on Suite W.
Conditional on the run reaching a final answer, it disappears:

| | pi | native |
|---|---|---|
| LFM-G8 Suite W | 20/30 = 74% | 9/12 = **75%** |
| LFM-G8 Suite T | 29/30 = 97% | 18/18 = **100%** |
| LFM-GQ4 Suite W | 20/28 = 71% | 12/15 = **80%** |
| LFM-GQ4 Suite T | 26/28 = 93% | 25/25 = **100%** |

`native` matches or beats `pi` in every cell. What the production harness supplies is not
retrieval, reasoning or tool use — it is the conversion of a completed investigation into a
final message, which §4.5 forbids the bare loop from helping with and which these models cannot
reliably do alone. Degenerate terminations, over the same two configurations, run at **30 of 60
`native` Suite W runs (50%) against 5 of 60 under `pi` (8%)**; on Suite T, 17 of 60 (28%) against
4 of 120 across all four LFM configurations (3%).

Two consequences for reading any number in this document:

- A `pi` score is a capability measurement with a termination floor underneath it. A `native`
  score conflates capability with a failure to stop. They answer different questions.
- `invalid_calls` is measurable only under `native` — 3 and 12 for LFM-G8's two suites, 0 and 9
  for LFM-GQ4's. `pi` reports 0 everywhere because it repairs internally, which means "not
  observed", never "none happened". This is the concrete case for retaining `native`.

## 6. Context scaling

TTFT roughly doubles from 8K to 16K for every configuration. Generation degrades 3–7% for the
LFM configurations and 15–25% for Bonsai. Prompt throughput drops about 6%.

Memory is where the model choice tells: the GGUF LFM builds stay at 2.4–3.6 GiB at 16K, while
Bonsai reaches 8.5–12.0 GiB and swaps. An 8B model at 16K is at the edge of a 16 GiB machine
regardless of how it scores.

Swap flags should be read with care: two flipped between campaigns, both towards not swapping,
and a single campaign's flag is not evidence about a configuration (`findings.md`, 2026-09-03).
Only the configurations that swap in both campaigns — BON-M2 on both tiers, BON-G2 at 16K — are
making a claim about memory pressure rather than about the machine that evening.

## 7. What the `v6` → `v7` comparison establishes about the instrument

The `v7` bump changed two task wordings, one assertion, the change baseline and the pi driver's
record-keeping. None of it touches inference, so the campaign doubles as a replication.

**It replicates.** Where nothing changed applies, nothing moved: BON-M2 3/30, BON-G2 0/30,
LFM-G8 `native` 9/30, LFM-GQ4 `native` 12/30, identical in both campaigns. Stage 1 throughput
agreed to two decimal places on all six configurations.

**The Suite T movement is entirely grading, and this is measurable rather than argued.** T05's
assertion changed while its prompt did not, so the `v6` answers can be re-graded under the `v7`
check — the same text, judged both ways: 7/12 as graded, 12/12 re-graded. The observed Suite T
change across the four LFM configurations is +5 in total. The grading fix accounts for all of
it. No Suite T improvement at `v7` reflects a model doing better.

**W07 now measures its stated property.** Under `pi`, 2/12 → 8/12, with failures on
`rowcount_correct` collapsing from 10 to 2 while the instruction-conflict arm held flat at 2.
The two survivors miscounted nothing — they wrote the right number to the wrong directory (§8).

**Timed-out runs stopped being blank.** `v6` recorded 9 timeouts with no transcript, no tool
calls and progress 0. `v7` recorded 14 timeouts, all 14 with transcripts, 711 tool calls between
them and progress recorded for every one.

**Suite W's movement is noise**, per §3.

## 8. Limitations and threats to validity

1. **Resolution.** 30 runs per cell is 10 tasks × 3 repetitions, and repetitions of one task are
   correlated, so the effective sample is nearer 10. At the observed pass rates this cannot
   separate configurations differing by less than roughly 15 percentage points. Reporting Suite W
   as a bare fraction invites over-reading; an interval would not.
2. **Two tasks are decided by our own path handling, not the model.** Two of twelve W07 runs
   computed the right count and wrote it beside the fixture root, where the container mount lets
   the write silently succeed. Fix recorded in `implementation-plan.md`.
3. **T05's disclaimer allowance is incomplete.** It matches the phrasings seen in `v6` and misses
   others; one `v7` run failed for naming the file the exception applies to. Also recorded as a
   defect.
4. **One machine.** Peak memory, swap and admissibility are machine-dependent by construction,
   and pooling across hosts is never done (§11).
5. **One harness, one prompt.** §1.1 already states results do not transfer to a different
   agent harness or prompting style. The `native`/`pi` gap in §5 is direct evidence of how large
   that effect is.
6. **`temperature=0` is not determinism.** Outcome flips between campaigns on identical inputs
   are documented; T03 moved 0/3 → 3/3 for LFM-M8 with no relevant change.
7. **W07 and T07 are not comparable across drivers**, by design: `pi` loads the fixture's
   adversarial `AGENTS.md` into its system prompt, `native` exposes it only if the model reads it.

## 9. Open questions

- **Does the suite need more resolution, or is the honest answer that these four operating
  points are equivalent on task success?** More repetitions, more mid-difficulty tasks, or
  reporting intervals are the three options; the first two cost campaign hours, the third is
  nearly free and would prevent the misreading that has already happened once.
- **Would a prompt change fix the aggregation failures?** The evidence splits: telling the model
  to compute rather than reason in prose plausibly helps (it already does so half the time), but
  the FX-direction errors survive into the code the model writes, so a prompt would convert a
  wrong mental sum into a correctly-executed wrong program. Stage 5B is where this belongs, and
  §11 requires it to be a separate arm rather than an edit.
- **Why does `native` degenerate on half its Suite W runs?** The §4.2 detector has fired in
  three consecutive campaigns — at `v7`, LFM-G8 39% and LFM-GQ4 29% as the report computes it
  across their native records — and Stage 5B's sampling pass remains unrun. That is the largest
  single unexplained effect in the data.
- **Does the Stage 5A conclusion hold at 16K?** Stage 3 has never been run.

## 10. Regenerating anything here

```sh
.venv/bin/python -m harness.report                  # the current table, from raw JSONL only
.venv/bin/python -m harness.report --out doc/report-v7.md
```

Reports regenerate from `results/*/raw/*.jsonl` and are never hand-edited. Everything under
`raw/` is pooled unconditionally; superseded campaigns live in `results/*/archive/` and are
excluded by being there. The per-task and statistical figures in this document are derived from
the same files.
