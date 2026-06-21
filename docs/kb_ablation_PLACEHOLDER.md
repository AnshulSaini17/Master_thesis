# KB Ablation — ILLUSTRATIVE / PLACEHOLDER DATA

> ⚠️ **DO NOT PRESENT THESE NUMBERS AS REAL RESULTS IN THE THESIS.**
>
> The KB ablation experiment (`tests/benchmark_kb_ablation.py`) was implemented but
> **not executed** for this thesis due to time and API-credit constraints. The
> tables below show the *shape* of what an ablation would produce and the
> *qualitative pattern* we'd expect based on the design of the three KBs and
> the per-level results from the main benchmark — they are **fabricated for
> illustration only**.
>
> Use this document to:
> - Lock down the **table layout** for the thesis evaluation chapter
> - Decide which **metrics** the thesis will report
> - Write the **methodology** and **expected-result** prose
>
> Before submission, either:
> (a) **rerun the ablation** and replace these numbers with the real values, **or**
> (b) **move this content into a "Future Work" section** with the explicit caveat
>    that the ablation was not executed and the table is hypothetical.
>
> Never copy these numbers into the results chapter as if measured.

---

## Hypothetical setup (matches the actual script)

| Parameter             | Value |
|-----------------------|---|
| KB variants compared  | `curated` (api_kb), `ast` (api_kb_ast), `token_600` (api_kb_token_600) |
| Approaches            | RAG, Agentic (Prompting omitted — no KB) |
| Levels                | L0 – L5 (Exp 1 corpus) |
| Runs per cell         | 3 (same as main benchmark) |
| Shapes per run        | 3 (same as main benchmark) |
| Pass criterion        | A level passes for an `(approach, KB)` pair if ≥ 2 of 3 runs pass (each run passes if all 3 shapes pass) |
| Model                 | `claude-sonnet-4-6` |
| Embedding model       | `text-embedding-3-large` (3072-d) |

Total runs in a full sweep would be **3 KBs × 2 approaches × 6 levels × 3 runs = 108 runs**.

---

## Table 1 — Pass rate per (KB × approach × level) [PLACEHOLDER]

> *Illustrative numbers; not measured.* Indices `pass / total`.

### RAG approach

| Level                              | curated | ast      | token_600 |
|------------------------------------|:-------:|:--------:|:---------:|
| L0 Single Linear                   | 3/3 ✓   | 3/3 ✓    | 3/3 ✓     |
| L1 MLP (3-layer ReLU)              | 3/3 ✓   | 3/3 ✓    | 3/3 ✓     |
| L2 Single-Head Attention           | **3/3 ✓** | 1/3 ✗  | 2/3 ✓     |
| L3 Gated MLP + RMSNorm             | 0/3 ✗   | 0/3 ✗    | 0/3 ✗     |
| L4 Multi-Head Attention + RMSNorm  | 0/3 ✗   | 0/3 ✗    | 0/3 ✗     |
| L5 Llama Decoder Layer             | 0/3 ✗   | 0/3 ✗    | 0/3 ✗     |
| **Difficulty Score (D)**           | **2**   | 1        | 2         |

### Agentic approach

| Level                              | curated   | ast       | token_600 |
|------------------------------------|:---------:|:---------:|:---------:|
| L0 Single Linear                   | 3/3 ✓     | 3/3 ✓     | 3/3 ✓     |
| L1 MLP (3-layer ReLU)              | 3/3 ✓     | 3/3 ✓     | 3/3 ✓     |
| L2 Single-Head Attention           | 2/3 ✓     | 2/3 ✓     | 3/3 ✓     |
| L3 Gated MLP + RMSNorm             | 3/3 ✓     | 2/3 ✓     | 2/3 ✓     |
| L4 Multi-Head Attention + RMSNorm  | 2/3 ✓     | 1/3 ✗     | 1/3 ✗     |
| L5 Llama Decoder Layer             | **3/3 ✓** | 1/3 ✗     | 2/3 ✓     |
| **Difficulty Score (D)**           | **5**     | 3         | 3         |

**Expected qualitative reading.** The agent dominates regardless of KB. RAG is
brittle to KB quality (clears L2 with curated, not with AST). The agent's
Difficulty Score drops from 5 (curated) to 3 (auto-built) — a real gap, but
much smaller than RAG's collapse.

---

## Table 2 — Mean tool calls (agentic only) per passed cell [PLACEHOLDER]

> *Illustrative numbers; not measured.* Larger = agent needed more retrievals
> to reach a passing solution.

| Level                              | curated | ast  | token_600 | Δ (ast − curated) |
|------------------------------------|--------:|-----:|----------:|------------------:|
| L0 Single Linear                   | 9.0     | 11.0 | 10.3      | +2.0              |
| L1 MLP (3-layer ReLU)              | 10.0    | 12.7 | 11.3      | +2.7              |
| L2 Single-Head Attention           | 15.0    | 19.5 | 17.7      | +4.5              |
| L3 Gated MLP + RMSNorm             | 14.7    | 19.0 | 17.0      | +4.3              |
| L4 Multi-Head Attention + RMSNorm  | 17.0    | 22.0 | 21.3      | +5.0              |
| L5 Llama Decoder Layer             | 25.7    | 33.0 | 30.5      | +7.3              |
| **Mean across passed levels**      | **15.2**| 19.5 | 18.0      | +4.3              |

**Expected qualitative reading.** The agent compensates for KB noise by issuing
more `lookup_kb` calls. The compensation cost grows roughly linearly with
level difficulty.

---

## Table 3 — Total input tokens per (KB × approach × all levels) [PLACEHOLDER]

> *Illustrative numbers; not measured.* Summed across all 18 runs in that column
> (6 levels × 3 runs).

| Approach   | curated     | ast         | token_600   |
|------------|------------:|------------:|------------:|
| RAG        | 51 k        | ~ 75 k      | ~ 90 k      |
| Agentic    | 3.5 M       | ~ 4.2 M     | ~ 4.0 M     |

**Expected qualitative reading.** RAG's input-token cost scales with chunk
size: `token_600` retrieves three ~1 185-char chunks per query while
`curated` retrieves three ~316-char chunks. The agent's input-token cost is
dominated by its persistent system prompt and tool descriptions; the
chunk-size effect is smaller in relative terms.

---

## Suggested thesis prose around the tables

Whichever way you frame the ablation in the thesis, the prose can be honest
without overclaiming. Two options:

### Option A — "Future work" framing (if the ablation is not rerun)

> **Future work — KB ablation.** A controlled ablation comparing three KB
> construction strategies (curated, AST-chunked, token-600-chunked) under
> identical retrieval parameters was implemented (`tests/benchmark_kb_ablation.py`)
> but not executed within the time budget of this thesis. The hypothesised
> outcome, given the chunk-quality characteristics described in
> Section ~\ref{sec:kb}, is that (i) RAG pass rates would degrade
> substantially under auto-chunked KBs because single-shot retrieval relies
> on each retrieved chunk being a self-contained API record, and (ii)
> agentic pass rates would degrade less, because the agent can issue
> follow-up `lookup_kb` calls to compensate for individual noisy retrievals.
> Quantifying these effects empirically is left to future work.

### Option B — "Methodology described, results pending" framing

> **KB ablation methodology.** To isolate the contribution of KB curation
> from the contribution of the retrieval / agent pipeline, an ablation
> compares three KB variants (curated, AST-chunked, token-600-chunked)
> under identical retrieval parameters (k = 3, top-20 unique chunks, same
> embedding model). For each `(KB, approach, level)` cell, three
> independent runs are evaluated under the same three-shape robustness
> protocol as the main benchmark; pass rate, mean tool-call count
> (agentic only), and total input-token cost are reported per cell.
> Empirical results of this ablation are deferred to future work due to
> resource constraints.

Both keep the structure of the methodology (which is real) and avoid
claiming empirical results that weren't actually measured.

---

## Why the placeholder numbers look the way they do

These choices were made to make the *qualitative pattern* match expectations,
not to predict any specific real outcome:

1. **L0–L1 are saturated across all KBs.** Linear and MLP can be retrieved
   correctly from any chunked source — there's only one relevant API entry
   each (`nn.Linear`, `nn.relu`).
2. **RAG plateau shifts down with noisier KBs.** Curated lets RAG clear L2
   (single ops to retrieve); auto-chunked KBs surface fragments of code that
   the LLM struggles to assemble in one shot.
3. **Agent always wins, with reduced lead under noisy KBs.** Because the
   agent can retry, it absorbs KB noise — but the cost shows up as more
   `lookup_kb` calls per passed run.
4. **Difficulty Score for the agent drops from 5 → 3 under auto-chunked KBs**
   without any failure being catastrophic. The thesis story stays:
   "agent dominates everywhere, but a good KB still matters."

Once the ablation is actually run, these qualitative shapes may or may not
hold. The real numbers must replace this placeholder before any of them is
quoted in the thesis as empirical evidence.
