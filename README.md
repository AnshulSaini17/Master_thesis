# Master's Thesis — Automatic Transpilation of PyTorch into a Proprietary Accelerator Stack

This repository accompanies my master's thesis on **automatic transpilation
from PyTorch `nn.Module` implementations into the Tensordyne accelerator
software stack** using large-language-model-based systems. It contains the
non-proprietary part of the code, the system prompts, the benchmark results,
the failure analysis, and supporting documentation.

**Author**: Anshul Saini · **Affiliation**: Tensordyne AI (industrial thesis)

## What this repository is

The thesis studies whether LLM-based systems can reliably translate PyTorch
models into a target stack whose APIs are *proprietary and absent from
public pre-training data*. The research question is split into three sub-questions:

1. How far can **direct prompting** alone go on this task?
2. How much does **retrieval-augmented generation (RAG)** help when the target
   API is proprietary?
3. Does an **agentic workflow** with controlled tool use, execution feedback,
   and iterative repair outperform one-shot generation in this setting?

Correctness is defined strictly: a translation passes only if it is valid
Python, uses valid target-side APIs, executes, **and** matches the PyTorch
reference numerically under shared weights and the same input. The metric is
`max_diff = max_j |y_pytorch_j - y_target_j|` on the output tensor.

## Scope of this repository

This is the **subset of my thesis work that does not depend on Tensordyne's
proprietary infrastructure** and that can be shared externally. Specifically,
it contains:

- The LangGraph-based agent that I designed and implemented (`langgraph_agent/`)
- The three approach implementations — prompting, RAG, agentic — and their
  evaluation harnesses (`benchmark_eval.py`, `benchmark_vpu_eval.py`,
  `benchmark_kb_ablation.py`, `benchmark_report.py`)
- The PyTorch test models and Triton test kernels used as benchmark inputs
  (`testing_layers/`, `triton_testing_layers/`)
- The complete raw and aggregated benchmark results (`benchmark_results.json`,
  `benchmark_vpu_results.json`, the matching `.csv` summaries, and per-run
  generated code under `benchmark_outputs/` and `benchmark_vpu_outputs/`)
- Methodological documentation written in this work (`docs/`)

**Not included** (proprietary and excluded with the company's permission):

- The Tensordyne SDK (`tensordyne.nn`, `tensordyne.ir`, `tensordyne.runner`)
- The Pyxis VPU functional specification (internal hardware document)
- The hand-curated FAISS knowledge bases (built from those specs) and the
  scripts that construct them (`build_api_kb.py`, `build_vpu_kb.py`)
- The internal runner-op additions made to the Tensordyne SDK

Because the SDK and the KBs are not included, **the code in this repository
will not run end to end**. It is intended as documentation of methodology and
as evidence of what I authored. The benchmark results were produced on the
company's internal monorepo and saved here as-is.

## Repository layout

```
.
├── README.md                              this file
├── langgraph_agent/
│   ├── agent.py                           the LangGraph ReAct agent (both modes)
│   ├── tools.py                           the 5 agent tools (lookup_kb, think, etc.)
│   ├── ARCHITECTURE.md                    agent architecture overview
│   ├── demo.ipynb                         interactive demo on Exp 1 inputs
│   └── explore_vpu.ipynb                  exploration of the VPU lowering pipeline
├── benchmark_eval.py                      Exp 1 benchmark — PyTorch → Tensordyne.nn
├── benchmark_vpu_eval.py                  Exp 2 benchmark — Triton → VPU IR
├── benchmark_report.py                    aggregator → summary tables
├── benchmark_kb_ablation.py               KB-construction ablation harness
├── transpile_evaluation_helper.py         numeric comparison machinery (Exp 1)
├── build_kb_ast.py                        AST-based KB chunker
├── build_kb_token.py                      token-window KB chunker
├── src/vector_search/core/ast_chunking.py CASTChunker used by build_kb_ast.py
├── testing_layers/                        PyTorch input models (Exp 1)
├── triton_testing_layers/                 Triton input kernels + PyTorch refs (Exp 2)
├── docs/
│   ├── system_prompts.md                  all prompts & tool docstrings (both exps)
│   ├── benchmark_results.md               full results write-up
│   ├── failure_categories.md              failure-mode analysis
│   ├── kb_ablation_PLACEHOLDER.md         ablation methodology (illustrative table)
│   └── categorize_failures.py             script that produces the failure analysis
├── benchmark_results.json + .csv          Exp 1 raw + summary results
├── benchmark_vpu_results.json + .csv      Exp 2 raw + summary results
├── benchmark_outputs/                     Exp 1 generated code per run
└── benchmark_vpu_outputs/                 Exp 2 generated code per run
```

## The two experiments at a glance

### Experiment 1 — PyTorch `nn.Module` → Tensordyne.nn DSL

A six-level difficulty ladder (L0 single Linear → L5 Llama Decoder Layer),
with each level translated three times by each of three approaches and
verified numerically against a PyTorch reference on three input shapes.

| Metric                       | Prompting | RAG     | Agentic   |
|------------------------------|:---------:|:-------:|:---------:|
| Difficulty Score (D)         | 1         | 2       | **5**     |
| First-Pass Rate              | 33 %      | 50 %    | **89 %**  |
| Self-Correction (avg tools)  | —         | —       | 15.2      |
| Numeric Accuracy (mean diff) | 0.0       | 0.0     | 1.32 × 10⁻⁷|

### Experiment 2 — Triton GPU kernel → Tensordyne IR with `VpuKernel` subgraphs

A separate, lower-level extension that asks whether the same agentic approach
can also operate at the IR level, with the agent directly emitting an
`ir.RootGraph` containing one or more `VpuKernel` subgraphs.

| Metric                       | Prompting | RAG     | Agentic   |
|------------------------------|:---------:|:-------:|:---------:|
| Difficulty Score (D)         | 1         | 1       | **4**     |
| First-Pass Rate              | 29 %      | 43 %    | **86 %**  |
| Self-Correction (avg tools)  | —         | —       | 13.8      |
| Numeric Accuracy (mean diff) | 0.0       | 0.0     | 1.04 × 10⁻⁵|

A detailed breakdown — per level, per approach, with cost and failure
analysis — lives in [docs/benchmark_results.md](docs/benchmark_results.md)
and [docs/failure_categories.md](docs/failure_categories.md).

## Key findings (short summary)

- **The agent dominates both experiments.** It reaches the top of the Exp 1
  ladder and passes 6 of 7 levels in Exp 2 (the unpassed level is an
  infrastructure failure of the PyTorch reference itself, not a translation
  failure — see [docs/benchmark_results.md](docs/benchmark_results.md)).
- **Failure modes differ qualitatively across approaches.** Across 117 shape
  evaluations per approach, the agent produced **zero hallucinated imports**,
  while RAG produced 33. This is direct evidence that the agent's interactive
  `lookup_kb` queries prevent the "I've seen this name in retrieved docs, so
  it must be importable" failure mode that single-shot RAG suffers from.
- **The cost gap is two orders of magnitude.** Agentic uses ~100 × the input
  tokens of prompting in Exp 1 and ~375 × in Exp 2 — this is the price of
  letting the agent iterate via tool calls.
- **No sandbagging events** were recorded across all 63 Exp 2 runs (sandbagging
  detection — agent output ≈ 0 against a non-zero reference — was implemented
  as a guard rail).

The full thesis writeup discusses these findings in detail.

## Configuration used for the reported results

- **Language model (all three approaches)**: Anthropic `claude-sonnet-4-6`
- **Embedding model (KB and retrieval)**: OpenAI `text-embedding-3-large` (3072-d)
- **FAISS index type**: `IndexFlatL2` (exact L2 search)
- **Retrieval `k`** (per query, both `lookup_kb` and RAG): 3
- **RAG context size**: up to 12 queries × top-3 chunks, deduplicated, kept ≤ 20 chunks
- **Number of runs per cell**: 3 (≥ 2 of 3 = level passes)
- **Number of shapes per run**: 3 (all 3 must pass)
- **Pass tolerance**: `max_diff < 1.0` against PyTorch fp32 reference

All exact prompts (system prompts and tool docstrings) are reproduced verbatim
in [docs/system_prompts.md](docs/system_prompts.md) for citation in the thesis.

## How to read the repository

1. Start with [docs/benchmark_results.md](docs/benchmark_results.md) — the
   results document; it includes the headline tables, per-level breakdowns,
   cost analysis, sandbagging report, and anomalies worth flagging.
2. Open [docs/system_prompts.md](docs/system_prompts.md) for the exact
   prompts used in each approach.
3. [docs/failure_categories.md](docs/failure_categories.md) contains the
   categorisation of every failed shape evaluation across both experiments.
4. [langgraph_agent/agent.py](langgraph_agent/agent.py) is the agent itself;
   [langgraph_agent/tools.py](langgraph_agent/tools.py) defines the five
   tools the agent uses.
5. [benchmark_eval.py](benchmark_eval.py) is the Exp 1 protocol;
   [benchmark_vpu_eval.py](benchmark_vpu_eval.py) is the Exp 2 protocol.
   Both contain the prompting and RAG system prompts as in-file constants
   (`_PROMPTING_SYSTEM`, `_RAG_SYSTEM`, `_PROMPTING_VPU_SYSTEM`, `_RAG_VPU_SYSTEM`).
6. The two notebooks under [langgraph_agent/](langgraph_agent/) — `demo.ipynb`
   and `explore_vpu.ipynb` — document the interactive exploration that
   informed the system design. They reference internal Tensordyne APIs and
   will not execute in this repository, but their content (markdown, code,
   and saved outputs) is preserved as part of the development record.

## Reproducibility note

Because this repository excludes the Tensordyne SDK and the proprietary
knowledge bases, the code as published is **not directly runnable**.
Reproducing the benchmarks requires access to the internal Tensordyne
monorepo. The complete experimental setup, including the proprietary parts,
lives in Tensordyne's internal repository on branch
`anshul/thesis-vpu-snapshot`, tagged at the submission commit.

## License and attribution

All code and documentation in this repository was authored by me as part of
my master's thesis. The Tensordyne SDK and Pyxis VPU specifications referenced
in the thesis are proprietary to Tensordyne AI and are not included here.

The use of this repository for academic review (thesis examination) is
explicitly permitted. Any other use of the code or its derivatives requires
prior written permission from Tensordyne AI.
