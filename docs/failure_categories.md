# Failure Categories

Every passed run already passed — what's interesting are the failures. This
document categorises every failed `(run, shape)` pair across both experiments
by the underlying root cause, then aggregates them per approach.

- **Exp 1 (PyTorch → Tensordyne.nn):** 54 runs × 3 shapes = 162 shape-evaluations total, **69 failures**.
- **Exp 2 (Triton → VPU IR):** 63 runs × 3 shapes = 189 shape-evaluations total, **87 failures**.

The categorisation was produced by [tests/docs/categorize_failures.py](tests/docs/categorize_failures.py),
which combines the JSON-recorded error strings with a re-import of each saved
`output.py` to recover the true Python exception (the JSON truncates at 300
characters, which often cuts off the actual cause).

---

## Failure categories used

| Category                       | Meaning                                                                                       |
|--------------------------------|-----------------------------------------------------------------------------------------------|
| **api_hallucination_import**   | Generated code imports a module/class/symbol that doesn't exist in Tensordyne (the LLM made it up). |
| **api_hallucination_attr**     | Code references a method or attribute of a real module that the API doesn't expose.           |
| **api_misuse_signature**       | A real API was called with the wrong number / name / type of arguments.                       |
| **output_format_wrong**        | Output didn't follow the required schema — for Exp 2, no `VpuKernel` or no `build_graph(input_type)`. |
| **weight_load_mismatch**       | Trace succeeded, but PyTorch weights couldn't be copied into the generated module by attribute name. |
| **shape_mismatch**             | Generated code ran, but produced a tensor of a different shape than the reference.            |
| **numeric_mismatch**           | Shapes matched, but `max_diff` exceeded tolerance.                                            |
| **sandbagging**                | Agent produced all-zero output against a non-zero reference (VPU-only metric).                |
| **infra_reference_crashed**    | The PyTorch reference itself raised an exception, so we never got to compare against the agent's output. |
| **infra_runner_crashed**       | The Tensordyne runner crashed on an otherwise-valid graph.                                    |
| **syntax_error**               | Pure Python syntax error in the generated code.                                               |

Two categories that turn out to be empty in the data and so don't appear in
the tables below: `api_hallucination_attr` (no NameError-on-tensordyne
patterns survived once import-time hallucinations were caught) and
`numeric_mismatch` / `shape_mismatch` (no passing-trace-but-wrong-output
failures, in either experiment).

---

## Experiment 1 — PyTorch → Tensordyne.nn

### Aggregate failure counts (69 total)

| Category                  | Count | %    |
|---------------------------|------:|-----:|
| api_misuse_signature      | 48    | 70 % |
| api_hallucination_import  | 18    | 26 % |
| weight_load_mismatch      | 3     |  4 % |

### Breakdown per approach

| Category                  | Prompting | RAG | Agentic | TOTAL |
|---------------------------|----------:|----:|--------:|------:|
| api_hallucination_import  |     —     |  18 |    —    |  18   |
| api_misuse_signature      |     33    |   9 |    6    |  48   |
| weight_load_mismatch      |      3    |   — |    —    |   3   |
| **TOTAL failed shapes**   |   **36**  |**27**|  **6**| **69**|

### What this says, in plain English

- **Agentic has the fewest failures by a wide margin (6 of 69 — 9 %).** Its
  six failures are all "wrong API signature" — it called a real Tensordyne
  function with the wrong number of arguments. It never hallucinated an
  import that didn't exist and never failed weight loading.
- **Prompting fails by mis-using real APIs.** With no retrieval mechanism,
  the model has memorised some Tensordyne names but doesn't know exact
  signatures, so it passes the wrong argument count (e.g. `BufferizedRMSNorm`
  without `normalized_shape`).
- **RAG fails primarily by hallucinating imports (18 of 27 failures —
  66 %).** Concretely, RAG returns chunks that *describe* helpers like
  `_build_rms_norm`, and the LLM tries to `from tensordyne.nn.modules.normalization
  import _build_rms_norm` — but those helpers are private and not actually
  exported. RAG sees the name in a retrieved snippet and incorrectly assumes
  it's importable.
- **Weight-loading failures are limited to prompting (3/3).** When prompting
  invents an attribute name like `self.norm = nn.RMSNorm(...)` instead of
  `self.weight = nn.Parameter(...)`, the evaluator can't copy the PyTorch
  weights by name path. RAG and the agent avoid this because the KB
  explicitly warns about attribute naming.

### Representative error per category

- **api_misuse_signature** — `too many positional arguments` (e.g. for
  `nn.softmax(x, dim=-1)` called as `nn.softmax(x, -1)`).
- **api_hallucination_import** — `cannot import name '_build_rms_norm' from
  'tensordyne.nn.modules.normalization'`. The KB documents the helper, but
  the LLM treats it as an importable name.
- **weight_load_mismatch** — `'url_dict' was provided but resource
  information for parameter 'attn_norm.weight' is missing` — the evaluator
  expected the parameter named `weight` under one path, generated code put
  it under a different path.

---

## Experiment 2 — Triton → VPU IR

### Aggregate failure counts (87 total)

| Category                  | Count | %    |
|---------------------------|------:|-----:|
| output_format_wrong       | 54    | 62 % |
| api_hallucination_import  | 15    | 17 % |
| infra_reference_crashed   | 15    | 17 % |
| infra_runner_crashed      | 3     | 3 % |

### Breakdown per approach

| Category                  | Prompting | RAG | Agentic | TOTAL |
|---------------------------|----------:|----:|--------:|------:|
| api_hallucination_import  |     —     |  15 |    —    |  15   |
| output_format_wrong       |     45    |   9 |    —    |  54   |
| infra_reference_crashed   |     —     |   6 |    9    |  15   |
| infra_runner_crashed      |     —     |   3 |    —    |   3   |
| **TOTAL failed shapes**   |   **45**  |**33**|  **9**| **87**|

### What this says, in plain English

- **Agentic has the fewest failures (9 of 87 — 10 %), and every single one
  is an infrastructure failure, not a translation failure.** All 9 failed
  agentic shapes are the L5 (AbsMaxNorm) cell where the PyTorch reference
  itself broadcast-crashes before any comparison is possible. The agent
  may well have produced correct code; the evaluation pipeline simply
  couldn't tell.
- **Prompting fails by producing the wrong output schema.** 45 of its 45
  failures are `output_format_wrong` — the LLM never produced a valid
  `build_graph(input_type)` returning an `ir.RootGraph` with `VpuKernel`
  subgraphs. Single-shot prompting without iterative correction simply
  doesn't internalise the strict skeleton requirement.
- **RAG fails in three distinct ways**: it hallucinates submodule imports
  (15 — paths like `tensordyne.ir.instructions.pyxis.vpu` that don't
  exist), produces the wrong output skeleton (9), or its generated code
  causes the runner to crash (3). Plus 6 reference-crashes which it shares
  with the agent.
- **No sandbagging events.** Across all 189 shape evaluations the agent
  never produced all-zero output against a non-zero reference — the
  detector did its job as a guard rail; the rfp16-vs-fp32 comparison
  caught any would-be sandbagger by simply failing the `max_diff <
  tolerance` check.

### Representative error per category

- **output_format_wrong** — `build_graph raised: ...` (prompting / RAG
  produced code that defined `build_graph` but raised inside it, or
  didn't define it at all).
- **api_hallucination_import** — `ModuleNotFoundError: No module named
  'tensordyne.ir.instructions.pyxis'` (the LLM invented a submodule path
  that looks plausible but doesn't exist).
- **infra_reference_crashed** — `reference model failed: ... torch ...`
  followed by a PyTorch traceback. This is L5 AbsMaxNorm with default
  `num_features = 64` vs the benchmark's input shape `(4, 16)`.
- **infra_runner_crashed** — `runner execution failed: ...
  transpile_to_torch_runnable...` — the runner didn't have a builder for
  some op produced by RAG's generated graph.

---

## Cross-experiment headline finding

> The single most defensible thesis claim from the failure analysis is:
>
> **The agent's failure mode is qualitatively different from prompting and
> RAG's failure mode.** Prompting fails by misusing real APIs (Exp 1) or
> producing the wrong output schema (Exp 2). RAG fails by importing names
> it has seen described in the KB but that aren't actually exported.
> The agent — when it fails at all — fails by misusing a signature in Exp 1
> and *only* due to infrastructure issues (broken PyTorch references)
> in Exp 2. **None of the agent's failures are due to hallucinated
> imports or hallucinated symbols.**
>
> This is direct evidence that interactive `lookup_kb` queries prevent the
> "I've seen this name in the docs, so it must be importable" hallucination
> mode that single-shot RAG suffers from — because the agent can verify
> a name exists by querying it explicitly before using it.

### Combined table for thesis use

If you want a single combined table across both experiments:

| Failure category          | Prompting | RAG     | Agentic | Combined |
|---------------------------|----------:|--------:|--------:|---------:|
| api_hallucination_import  |     0     |   33    |    0    |    33    |
| api_misuse_signature      |    33     |    9    |    6    |    48    |
| output_format_wrong       |    45     |    9    |    0    |    54    |
| weight_load_mismatch      |     3     |    0    |    0    |     3    |
| infra_reference_crashed   |     0     |    6    |    9    |    15    |
| infra_runner_crashed      |     0     |    3    |    0    |     3    |
| **TOTAL failures**        |  **81**   | **60**  |  **15** |  **156** |
| Total shape evaluations   |   117     |   117   |   117   |   351    |
| **Failure rate**          | **69 %**  | **51 %**|  **13 %**| **44 %**|

(Total shape evaluations per approach = 6 levels × 3 runs × 3 shapes +
7 levels × 3 runs × 3 shapes = 54 + 63 = 117.)

---

## How to reproduce / refresh

The categorisation re-runs in a few seconds:

```bash
cd /workspaces/rock
python tests/docs/categorize_failures.py
```

It reads the two `benchmark_*_results.json` files plus the saved
`benchmark_outputs/` and `benchmark_vpu_outputs/` directories, and prints
the same tables this document quotes. If you ever rerun a benchmark, run
the categoriser afterwards to refresh the numbers.

---

## Caveats worth noting in the thesis

1. **The categoriser is keyword-based** and may misclassify edge cases. For
   any number you quote in the thesis, it's worth spot-checking 2–3 raw
   error strings in the JSON to be sure.
2. **The L5 (AbsMaxNorm) reference crashes** (15 of the 87 Exp 2 failures,
   17 %) inflate the failure count for both the agent and RAG. Subtracting
   those 15 leaves 72 "real" failures across the two approaches. This is
   already noted in [tests/docs/benchmark_results.md](tests/docs/benchmark_results.md)
   as a methodological limitation.
3. **Agent failures are likely under-counted relative to its true error
   rate** because the agent's stopping rule short-circuits any run where
   `run_evaluation` already returned `PASSED` inside the agent's stream.
   So the 6 agentic Exp 1 failures and the 9 agentic Exp 2 failures are
   "errors that survived the agent's own validation step" — a strict
   subset of the agent's total mistakes during the loop.
