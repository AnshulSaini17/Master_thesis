# Master's Thesis — Automatic Transpilation of PyTorch into a Proprietary Accelerator Stack

This repository accompanies my master's thesis on **automatic transpilation
from PyTorch `nn.Module` implementations into the Tensordyne accelerator
software stack** using large-language-model-based systems. It contains the
non-proprietary part of the code, the benchmark results, and the generated
outputs.

The thesis PDF is the canonical document for methodology, system design,
discussion, and conclusions. This repository is a companion to the PDF — it
holds the code I authored and the raw experimental data behind the tables
in the evaluation chapter.

**Author**: Anshul Saini · **Affiliation**: Tensordyne AI (industrial thesis)

## What this thesis is about

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

- The LangGraph-based agent that I designed and implemented, in **two
  versions** corresponding to the two experiments — `langgraph_agent_exp1/`
  contains the agent that produced the Experiment 1 results, and
  `langgraph_agent_exp2/` contains the agent that produced the Experiment 2
  results (see the note in the Repository layout below).
- The three approach implementations — prompting, RAG, agentic — and their
  evaluation harnesses (`benchmark_eval.py`, `benchmark_vpu_eval.py`,
  `benchmark_kb_ablation.py`, `benchmark_report.py`)
- The PyTorch test models and Triton test kernels used as benchmark inputs
  (`testing_layers/`, `triton_testing_layers/`)
- The complete raw and aggregated benchmark results (`benchmark_results.json`,
  `benchmark_vpu_results.json`, and the matching `.csv` summaries)

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
├── langgraph_agent_exp1/                  agent that ran Experiment 1
│   ├── agent.py                           nn-mode agent with the original PyTorch-focused _SYSTEM_PROMPT
│   ├── tools.py                           the 5 agent tools (no run_vpu_evaluation)
│   └── __init__.py
├── langgraph_agent_exp2/                  agent that ran Experiment 2
│   ├── agent.py                           adds vpu_lowering mode + _VPU_LOWERING_SYSTEM_PROMPT
│   ├── tools.py                           adds run_vpu_evaluation tool
│   ├── ARCHITECTURE.md                    agent architecture overview
│   ├── demo.ipynb                         interactive demo on Exp 1-style inputs
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
├── benchmark_results.json + .csv          Exp 1 raw + summary results
└── benchmark_vpu_results.json + .csv      Exp 2 raw + summary results
```

The per-run generated `output.py` files produced during the experiments
are not included here for IP reasons. The per-run records — including
input shape, max_diff per shape, error category, token cost, and elapsed
time — remain in the JSON result files.

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

The thesis PDF contains the full per-level breakdowns, cost analysis,
sandbagging report, and failure-mode discussion.

## Configuration used for the reported results

- **Language model (all three approaches)**: Anthropic `claude-sonnet-4-6`
- **Embedding model (KB and retrieval)**: OpenAI `text-embedding-3-large` (3072-d)
- **FAISS index type**: `IndexFlatL2` (exact L2 search)
- **Retrieval `k`** (per query, both `lookup_kb` and RAG): 3
- **RAG context size**: up to 12 queries × top-3 chunks, deduplicated, kept ≤ 20 chunks
- **Number of runs per cell**: 3 (≥ 2 of 3 = level passes)
- **Number of shapes per run**: 3 (all 3 must pass)
- **Pass tolerance**: `max_diff < 1.0` against PyTorch fp32 reference

The verbatim system prompts and tool docstrings are defined as constants in
the source files:

- `_SYSTEM_PROMPT` in [langgraph_agent_exp1/agent.py](langgraph_agent_exp1/agent.py) — the agent prompt used for Experiment 1
- `_VPU_LOWERING_SYSTEM_PROMPT` in [langgraph_agent_exp2/agent.py](langgraph_agent_exp2/agent.py) — the agent prompt used for Experiment 2
- `_PROMPTING_SYSTEM` / `_RAG_SYSTEM` in [benchmark_eval.py](benchmark_eval.py) — Exp 1 prompting and RAG baselines
- `_PROMPTING_VPU_SYSTEM` / `_RAG_VPU_SYSTEM` in [benchmark_vpu_eval.py](benchmark_vpu_eval.py) — Exp 2 prompting and RAG baselines
- Tool docstrings in [langgraph_agent_exp1/tools.py](langgraph_agent_exp1/tools.py) and [langgraph_agent_exp2/tools.py](langgraph_agent_exp2/tools.py)

**Why two agent directories?** The two experiments were run on two
different branches of the internal monorepo at different points in time.
Experiment 1 used the original PyTorch-focused agent (`anshul/thesis-bench`
branch). Experiment 2 was added later on a separate branch
(`anshul/thesis-vpu-snapshot`) where the agent was extended with a second
mode (`vpu_lowering`) for emitting IR directly, plus a new
`run_vpu_evaluation` tool. To preserve provenance, the repository keeps
both versions intact rather than merging them into one composite.

## How to read the repository

1. **[langgraph_agent_exp1/agent.py](langgraph_agent_exp1/agent.py)** — the
   Experiment 1 agent: nn-mode only, with the original PyTorch-focused
   `_SYSTEM_PROMPT` and outer-loop logic.
2. **[langgraph_agent_exp2/agent.py](langgraph_agent_exp2/agent.py)** — the
   Experiment 2 agent: adds `vpu_lowering` mode and `_VPU_LOWERING_SYSTEM_PROMPT`
   for emitting IR directly.
3. **[langgraph_agent_exp1/tools.py](langgraph_agent_exp1/tools.py) /
   [langgraph_agent_exp2/tools.py](langgraph_agent_exp2/tools.py)** — the
   four (Exp 1) or five (Exp 2) agent tools. Exp 2's `tools.py` adds
   `run_vpu_evaluation`.
4. **[benchmark_eval.py](benchmark_eval.py)** — the Experiment 1 protocol and
   the prompting/RAG system prompts as in-file constants.
5. **[benchmark_vpu_eval.py](benchmark_vpu_eval.py)** — the Experiment 2
   protocol and prompts.
6. **[benchmark_results.csv](benchmark_results.csv) /
   [benchmark_vpu_results.csv](benchmark_vpu_results.csv)** — the per-level
   summary tables the thesis quotes. The full per-run records (input shape,
   max_diff, error categories, tokens, wall time, etc.) live in the matching
   `.json` files.
7. **[langgraph_agent_exp2/demo.ipynb](langgraph_agent_exp2/demo.ipynb) /
   [langgraph_agent_exp2/explore_vpu.ipynb](langgraph_agent_exp2/explore_vpu.ipynb)** —
   exploratory notebooks documenting the system design and the VPU lowering
   pipeline. They reference internal Tensordyne APIs and will not execute in
   this repository.

## Reproducibility note

Because this repository excludes the Tensordyne SDK and the proprietary
knowledge bases, the code as published is **not directly runnable**.
Reproducing the benchmarks requires access to the internal Tensordyne
monorepo. The complete experimental setup, including the proprietary parts,
lives in Tensordyne's internal repository on branch
`anshul/thesis-vpu-snapshot`, tagged at the submission commit.

## License and attribution

All code in this repository was authored by me as part of my master's
thesis. The Tensordyne SDK and Pyxis VPU specifications referenced in the
thesis are proprietary to Tensordyne AI and are not included here.

The use of this repository for academic review (thesis examination) is
explicitly permitted. Any other use of the code or its derivatives requires
prior written permission from Tensordyne AI.
