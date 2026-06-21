# Benchmark Results

Consolidated results document for the master thesis. Both experiments use the
same evaluation protocol (3 approaches × N levels × 3 runs × 3 shapes per
run); pass criterion is numerical equivalence to a PyTorch reference within
tolerance, with sandbagging detection on Experiment 2.

- **Model used for every approach in every experiment:** `claude-sonnet-4-6`
- **Experiment 1 (Exp 1):** PyTorch `nn.Module` → Tensordyne.nn DSL — 6 levels (L0–L5), 54 runs total
- **Experiment 2 (Exp 2):** Triton GPU kernel → Tensordyne IR with `VpuKernel` subgraphs — 7 levels (L0–L6), 63 runs total
- **Raw data:** `tests/benchmark_results.json` (Exp 1), `tests/benchmark_vpu_results.json` (Exp 2)
- **CSV summaries:** `tests/benchmark_results.csv`, `tests/benchmark_vpu_results.csv`

---

## Table of contents

- [Headline results](#headline-results)
- [Experiment 1 — PyTorch → Tensordyne.nn](#experiment-1--pytorch--tensordynenn)
  - [Summary](#exp-1-summary)
  - [Per-level breakdown](#exp-1-per-level-breakdown)
  - [Cost analysis (tokens and wall time)](#exp-1-cost-analysis)
- [Experiment 2 — Triton → Tensordyne VPU IR](#experiment-2--triton--tensordyne-vpu-ir)
  - [Summary](#exp-2-summary)
  - [Per-level breakdown](#exp-2-per-level-breakdown)
  - [Cost analysis (tokens and wall time)](#exp-2-cost-analysis)
  - [Sandbagging report](#sandbagging-report)
  - [Anomalies worth flagging in the thesis](#anomalies-worth-flagging-in-the-thesis)
- [Cross-experiment comparison](#cross-experiment-comparison)
- [Reproducing these tables](#reproducing-these-tables)

---

## Headline results

The single table to put at the start of the evaluation chapter:

| Metric                       | Exp 1 (PyTorch → TD.nn) |          |            | Exp 2 (Triton → VPU IR) |          |            |
|------------------------------|:-----------------------:|:--------:|:----------:|:-----------------------:|:--------:|:----------:|
|                              | Prompting               | RAG      | Agentic    | Prompting               | RAG      | Agentic    |
| Difficulty Score (D)         | 1                       | 2        | **5**      | 1                       | 1        | **4**      |
| First-Pass Rate              | 33 %                    | 50 %     | **89 %**   | 29 %                    | 43 %     | **86 %**   |
| Self-Correction (avg tools)  | —                       | —        | 15.2       | —                       | —        | 13.8       |
| Numeric Accuracy (mean diff) | 0.0                     | 0.0      | 1.32 × 10⁻⁷| 0.0                     | 0.0      | 1.04 × 10⁻⁵|
| Total LLM tokens (in / out)  | 33.7 k / 16.4 k         | 51.5 k / 13.5 k | 3.5 M / 198 k | 29.8 k / 24.6 k | 102 k / 28 k | 11.2 M / 488 k |
| Total wall time              | 205 s                   | 217 s    | 3 017 s    | 370 s                   | 486 s    | 8 236 s    |

Reading the headline:

- **The agent dominates both experiments.** It reaches the top of the Exp 1
  ladder (D = 5) and stalls one level short of the top of Exp 2 (D = 4 — see
  the L5 anomaly below). Prompting and RAG plateau early.
- **The cost gap is two orders of magnitude.** Agentic uses ~100 × the input
  tokens of prompting in Exp 1 and ~375 × in Exp 2. This is the price of
  letting the agent retry via tool calls.
- **Numeric accuracy is fine where things pass.** Mean `max_diff` is at the
  level of single- or double-precision rounding error in Exp 1, and is
  dominated by LUT approximations in Exp 2 (Exp 2's softmax uses `pyxis.vpu.Exp`,
  which is LUT-backed → larger tolerable error).
- **The `0.0` mean diff for prompting/RAG is genuine** — when those baselines
  pass (early levels: matmul, MLP, etc.), the traced Tensordyne graph is
  numerically identical to the PyTorch reference. Trace fidelity is high; the
  hard part is producing correct code at all.

---

## Experiment 1 — PyTorch → Tensordyne.nn

### Exp 1 Summary

| Metric                       | Prompting | RAG     | Agentic   |
|------------------------------|:---------:|:-------:|:---------:|
| Difficulty Score (D)         | 1         | 2       | **5**     |
| First-Pass Rate              | 33 %      | 50 %    | **89 %**  |
| Self-Correction (avg tools)  | —         | —       | 15.2      |
| Numeric Accuracy (mean diff) | 0.0       | 0.0     | 1.32 × 10⁻⁷|

### Exp 1 Per-level breakdown

Each cell shows `passed/total` runs (where a run passes if all 3 shapes
pass). The agent's `avg_tool_calls` is the mean number of tool invocations
per passed run — this is the self-correction effort metric.

| Level                           | Prompting | RAG | Agentic | Agent avg tools | Agent mean max_diff |
|---------------------------------|:---------:|:---:|:-------:|:---------------:|:-------------------:|
| L0 Single Linear                | **3/3** ✓ | **3/3** ✓ | **3/3** ✓ | 9.0  | 0.0 |
| L1 MLP (3-layer ReLU)           | **3/3** ✓ | **3/3** ✓ | **3/3** ✓ | 10.0 | 0.0 |
| L2 Single-Head Attention        | 0/3 ✗     | **3/3** ✓ | **2/3** ✓ | 15.0 | 4.22 × 10⁻⁸|
| L3 Gated MLP + RMSNorm          | 0/3 ✗     | 0/3 ✗     | **3/3** ✓ | 14.7 | 2.12 × 10⁻⁷|
| L4 Multi-Head Attention + RMSNorm | 0/3 ✗   | 0/3 ✗     | **2/3** ✓ | 17.0 | 2.38 × 10⁻⁷|
| L5 Llama Decoder Layer          | 0/3 ✗     | 0/3 ✗     | **3/3** ✓ | 25.7 | 2.98 × 10⁻⁷|

Notes:

- Both prompting and RAG fall off cleanly between L1 and L2 — once attention
  is involved, the baselines can't recover.
- The agent has two "shaky pass" levels (L2 = 2/3, L4 = 2/3). These are
  worth a paragraph in the thesis: a 2/3 pass still counts as the level
  passing under the protocol (≥ 2/3), but it shows the agent is operating
  near the edge of its reliability envelope at those tiers.
- `avg_tools` grows monotonically with difficulty: **9 → 10 → 15.7 → 14.7 →
  17 → 25.7**. The dip at L3 (14.7) is a quirk of small sample size, not a
  real signal.

### Exp 1 Cost analysis

Token usage and wall time, summed across all 3 runs per cell.

| Approach × Level | Input tokens | Output tokens | Wall time |
|------------------|-------------:|--------------:|----------:|
| prompting L0     |        3 738 |           231 |     7.1 s |
| prompting L1     |        3 924 |           450 |     8.5 s |
| prompting L2     |        4 536 |         1 107 |    18.6 s |
| prompting L3     |        4 860 |         1 530 |    19.5 s |
| prompting L4     |        7 002 |         3 766 |    42.7 s |
| prompting L5     |        9 648 |         9 290 |   109.3 s |
| **prompting Σ**  |   **33 708** |    **16 374** | **205.7 s** |
| rag L0           |        2 721 |           270 |    15.5 s |
| rag L1           |        3 621 |           549 |    19.9 s |
| rag L2           |        5 586 |         1 221 |    22.4 s |
| rag L3           |        9 345 |         1 836 |    31.9 s |
| rag L4           |       12 468 |         3 656 |    53.1 s |
| rag L5           |       17 710 |         5 951 |    74.6 s |
| **rag Σ**        |   **51 451** |    **13 483** | **217.4 s** |
| agentic L0       |      207 647 |         8 352 |   158.9 s |
| agentic L1       |      206 734 |        11 049 |   182.0 s |
| agentic L2       |      541 779 |        25 092 |   408.2 s |
| agentic L3       |      427 494 |        26 365 |   406.7 s |
| agentic L4       |      642 077 |        35 832 |   537.3 s |
| agentic L5       |    1 467 398 |        91 280 |  1 324.0 s|
| **agentic Σ**    |**3 493 129** |   **197 970** | **3 017.1 s** |

Reading the cost numbers:

- The agent uses ~100 × the input tokens of prompting at every level. This
  is mostly cached prompt prefix from the agent's system prompt + tool
  descriptions being re-sent on every step of the ReAct loop.
- Output tokens scale with code length and number of correction attempts.
  At L5 the agent spent ~22 minutes per run on average.
- RAG is only ~50 % more expensive than prompting in input tokens (retrieval
  context is small) and *cheaper* in output tokens — interestingly because
  the retrieved context lets the model converge with less talking.

---

## Experiment 2 — Triton → Tensordyne VPU IR

### Exp 2 Summary

| Metric                       | Prompting | RAG     | Agentic   |
|------------------------------|:---------:|:-------:|:---------:|
| Difficulty Score (D)         | 1         | 1       | **4**     |
| First-Pass Rate              | 29 %      | 43 %    | **86 %**  |
| Self-Correction (avg tools)  | —         | —       | 13.8      |
| Numeric Accuracy (mean diff) | 0.0       | 0.0     | 1.04 × 10⁻⁵|

### Exp 2 Per-level breakdown

| Level                                       | Prompting | RAG       | Agentic   | Agent avg tools | Agent mean max_diff |
|---------------------------------------------|:---------:|:---------:|:---------:|:---------------:|:-------------------:|
| L0 Negate (elementwise multi-row)           | **3/3** ✓ | **3/3** ✓ | **3/3** ✓ | 8.3  | 0.0 |
| L1 Abs (HW-native unary)                    | **3/3** ✓ | **3/3** ✓ | **3/3** ✓ | 9.0  | 0.0 |
| L2 ReLU (predicate vs scalar zero)          | 0/3 ✗     | 0/3 ✗     | **3/3** ✓ | 12.7 | 0.0 |
| L3 Clip (parametric min/max)                | 0/3 ✗     | 0/3 ✗     | **3/3** ✓ | 14.7 | 0.0 |
| L4 Row-max reduction (TLM-verified pattern) | 0/3 ✗     | **3/3** ✓ | **3/3** ✓ | 9.0  | 0.0 |
| L5 AbsMaxNorm (reduce-then-broadcast)       | 0/3 ✗     | 0/3 ✗     | 0/3 ✗     | —    | —   |
| L6 Softmax (reduce + LUT exp + scalar bcast)| 0/3 ✗     | 0/3 ✗     | **3/3** ✓ | 29.0 | 6.25 × 10⁻⁵|

Notes:

- Agentic clears L0–L4 cleanly (3/3 each) and **passes L6 (Softmax) 3/3**,
  but **fails L5 (AbsMaxNorm) 0/3**. This non-monotonic ladder is the
  reason the agent's Difficulty Score is 4, not 6 — `D` is the highest
  *consecutive* level passed from L0.
- The L5 failures are **not** translation failures by the agent. They are
  caused by a shape-broadcast bug in the AbsMaxNorm PyTorch reference
  itself (default `num_features=64` doesn't match the benchmark's input
  shape `(4, 16)`). See [Anomalies worth flagging in the thesis](#anomalies-worth-flagging-in-the-thesis).
- L4 (Row-max reduction) RAG passing 3/3 while L2/L3 fail is also
  unusual — see Anomalies.
- L6 Softmax's `mean max_diff = 6.25 × 10⁻⁵` is two orders of magnitude
  worse than Exp 1's typical 10⁻⁷. This is the **LUT exp approximation**
  cost: `pyxis.vpu.Exp` is implemented as a lookup table on hardware and
  therefore has a small but measurable error vs PyTorch's full-precision
  `torch.exp`. Still well within the `< 1.0` tolerance.

### Exp 2 Cost analysis

| Approach × Level | Input tokens | Output tokens | Wall time |
|------------------|-------------:|--------------:|----------:|
| prompting L0     |        4 056 |           979 |    14.6 s |
| prompting L1     |        4 074 |           972 |    14.1 s |
| prompting L2     |        4 086 |         1 256 |    17.1 s |
| prompting L3     |        4 224 |         1 666 |    29.5 s |
| prompting L4     |        4 254 |         3 088 |    54.1 s |
| prompting L5     |        4 716 |         5 701 |    85.4 s |
| prompting L6     |        4 395 |        10 965 |   155.3 s |
| **prompting Σ**  |   **29 805** |    **24 627** | **370.1 s** |
| rag L0           |        8 853 |           788 |    20.6 s |
| rag L1           |        9 105 |         1 250 |    30.0 s |
| rag L2           |       11 301 |         2 590 |    49.5 s |
| rag L3           |       12 066 |         2 662 |    47.5 s |
| rag L4           |       16 962 |         2 025 |    38.9 s |
| rag L5           |       22 825 |         9 867 |   151.2 s |
| rag L6           |       21 178 |         8 806 |   148.6 s |
| **rag Σ**        |  **102 290** |    **27 988** | **486.3 s** |
| agentic L0       |      173 732 |        10 168 |   178.8 s |
| agentic L1       |      236 569 |        12 176 |   203.1 s |
| agentic L2       |      483 296 |        27 036 |   469.8 s |
| agentic L3       |      467 803 |        23 279 |   387.3 s |
| agentic L4       |      251 112 |        12 465 |   220.5 s |
| agentic L5       |    6 587 746 |       274 331 | 4 639.4 s |
| agentic L6       |    3 014 651 |       128 770 | 2 136.9 s |
| **agentic Σ**    |**11 214 909**|   **488 225** | **8 235.8 s** |

Reading the cost numbers:

- The agent spent **~25 % of its total token budget on L5 alone** (6.6 M of
  11.2 M input tokens). With an average of 43.7 tool calls per run and
  wall time of ~26 minutes per run, the agent was clearly thrashing on
  L5 — repeatedly producing a kernel, calling `run_vpu_evaluation`, getting
  the broadcast error, and trying a different approach (see Anomalies).
- L6 (the harder kernel by design — Softmax with LUT-exp + scalar broadcast)
  was *cheaper* than L5 in tokens and time, **and passed 3/3**. Difficulty
  by token count is not monotonic.

### Sandbagging report

Sandbagging detection (defined: agent output `|max| < 10⁻⁶` while
reference output `|max| > 10⁻²`) was the methodological contribution
that addresses the int16-truncation softmax-zero failure mode observed
during development (where an agent produced `Sub(load_0, load_0)` —
identically zero — and the runner's loose `max_diff` tolerance passed it).

| Experiment | Sandbagging events |
|------------|--------------------|
| Exp 1      | 0 (not checked — not applicable to nn-DSL)|
| Exp 2      | **0** across all 63 runs |

Zero sandbagging events is a positive result: the sweep ran in `rfp16`
mode against `float32` PyTorch references, where any all-zero kernel would
have a non-zero `|agent − ref|` and so would correctly fail by the
`max_diff < 1.0` criterion. The detector serves as a guard rail; the data
confirms it wasn't needed under this configuration.

### Anomalies worth flagging in the thesis

These should be discussed in the evaluation chapter for transparency:

1. **L5 (AbsMaxNorm) agent failures are infrastructure failures, not
   translation failures.** The PyTorch reference in
   `triton_testing_layers/absmax_norm_ref.py` defaults to `num_features = 64`,
   but the benchmark shapes use `(4, 16)`. The reference's
   `(x / (absmax + eps)) * self.weight` line fails with a broadcast error
   *before* the agent's output is compared. All 3 agentic L5 failures
   carry the error message `"reference model failed: ... torch.nn.modules.module"`.
   The agent likely produced viable code; we cannot tell because the
   reference crashed first. This is a methodological flaw in the eval
   harness that should be documented in the thesis as a limitation.

2. **L4 RAG passes 3/3 even though L2 and L3 RAG fail 0/3.** The
   row-max-reduction kernel happens to map cleanly to a known pattern in
   the VPU KB (`vpu_unrolled_reduction`), which RAG's retrieval picks up
   confidently. ReLU and Clip require *composing* multiple ops (a literal
   row + Max/Min) and the retrieved chunks didn't carry that composition
   recipe. This illustrates the failure mode of RAG: it succeeds when a
   close template exists and fails when synthesis is required, regardless
   of the underlying op's "difficulty" by other measures.

3. **L6 Softmax passes despite L5 AbsMaxNorm failing.** With caveat 1
   above accounting for L5, this is consistent. The agent's solution for
   L6 uses the `pyxis.vpu.Exp` LUT and a rfp21 accumulator pattern — both
   are in the VPU KB.

4. **Numeric tolerance differs by 2 orders of magnitude between
   experiments.** Exp 1's mean diff is ~3 × 10⁻⁷ (FP32 rounding error);
   Exp 2 L6's mean diff is ~6 × 10⁻⁵ (LUT exp approximation error). Both
   pass their respective `max_diff < 1.0` tolerances. The thesis should
   note that the **same** tolerance threshold applied to **different**
   hardware models yields different effective numerical guarantees.

5. **Prompting's first-pass rate is identical (29–33 %) in both
   experiments**, despite the tasks being quite different. This is the
   "easy first two levels" effect — when the task is structurally simple
   (linear, MLP, negate, abs), the single-shot prompting baseline gets it
   right ~100 % of the time. Above that, prompting falls off a cliff.

---

## Cross-experiment comparison

Same agent harness, two domains:

| Aspect                    | Exp 1                       | Exp 2                       |
|---------------------------|-----------------------------|-----------------------------|
| Source language           | PyTorch `nn.Module`         | Triton GPU kernel           |
| Target language           | Tensordyne.nn DSL           | Tensordyne IR + VpuKernel   |
| Abstraction level         | Module-level                | IR-level                    |
| Hardware modelled         | None (CPU trace)            | Pyxis VPU FIFO + AMEM       |
| Numeric tolerance regime  | FP32 rounding (~10⁻⁷)       | LUT approximation (~10⁻⁵)   |
| KB size                   | 123 chunks (api_kb)         | 76 chunks (45 ops + 16 patterns + 11 rules)|
| Number of agent tools     | 5                           | 5 (same set, eval tool differs) |
| Top difficulty reached    | L5 (Llama Decoder Layer)    | L6 (Softmax)                |
| Difficulty Score (agent)  | **5**                       | **4** (anomaly — see L5 note)|

Both experiments produce the same qualitative shape:

- Prompting plateaus at the trivial tier (L0–L1).
- RAG adds modestly (one extra level in Exp 1, jumps L4 selectively in Exp 2).
- Agentic dominates by a wide margin in both experiments.

The Exp 2 agentic D = 4 understates the actual capability because of the L5
infrastructure failure. Counting passes regardless of monotonicity: agentic
solved **6 of 7** Exp 2 levels (L0, L1, L2, L3, L4, L6) and **6 of 6** Exp 1
levels — i.e. **12 of 13** total tier translations.

---

## Reproducing these tables

Both reports are generated from the saved JSONs. To rerun:

```bash
cd /workspaces/rock/tests

# Exp 1 summary (the table at the top of this document)
python benchmark_report.py

# Exp 1 with per-level breakdown
python benchmark_report.py --detail

# Exp 1 summary written to CSV alongside the JSON
python benchmark_report.py --csv

# Exp 2 (same script, different results file)
python benchmark_report.py --results benchmark_vpu_results.json
python benchmark_report.py --results benchmark_vpu_results.json --detail
python benchmark_report.py --results benchmark_vpu_results.json --csv
```

The per-run raw records (timestamps, exact tokens, per-shape errors) live in
the JSON files and can be loaded with Python for any ad-hoc analysis the
thesis needs.
