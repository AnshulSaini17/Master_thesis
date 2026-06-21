#!/usr/bin/env python3
"""Thesis Benchmark (VPU branch): Prompting vs RAG vs Agentic
Triton-kernel → Tensordyne VPU IR translation.

Companion to ``benchmark_eval.py`` (the nn-DSL benchmark). Same evaluation
protocol so the two halves of the thesis can be reported side-by-side.

Evaluation protocol:
  - 7 difficulty tiers (L0 unary elementwise → L6 softmax with LUT exp)
  - 3 independent runs per tier per approach (handles LLM non-determinism)
  - 3 input shapes per run (robustness)
  - A run PASSES if the generated `build_graph(input_type)` produces an
    ir.RootGraph whose runner output matches a PyTorch fp32 reference within
    tolerance on ALL shapes AND does not exhibit sandbagging (all-zero output
    against a non-zero reference).
  - A tier PASSES for an approach if ≥ 2/3 runs pass.

Pass criterion: runner + fp32 reference (see ``evaluate_vpu_kernel``). TLM
build/harness is NOT in this loop — score the top kernels separately.

Usage:
  python benchmark_vpu_eval.py                          # all approaches, all tiers
  python benchmark_vpu_eval.py --approach agentic       # one approach only
  python benchmark_vpu_eval.py --levels 0 1 2           # specific tiers
  python benchmark_vpu_eval.py --no-resume              # rerun from scratch
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

# Load ANTHROPIC_API_KEY / OPENAI_API_KEY from tests/config/.env if present, so
# the script works without the caller exporting them into the shell first.
try:
    from dotenv import load_dotenv

    load_dotenv(TESTS_DIR / "config" / ".env")
except ImportError:
    pass

# ─── Settings ────────────────────────────────────────────────────────────────

N_RUNS = 3
PASS_THRESHOLD = 2
RESULTS_FILE = TESTS_DIR / "benchmark_vpu_results.json"
OUTPUTS_DIR = TESTS_DIR / "benchmark_vpu_outputs"
VPU_KB_PATH = TESTS_DIR / "data" / "databases" / "vpu_kb"
MODEL = "claude-sonnet-4-6"
EVAL_DTYPE = "rfp16"  # fp32 reference comparison — see evaluate_vpu_kernel
PASS_TOL = 1.0  # max_diff tolerance vs fp32 reference (matches run_vpu_evaluation)
SANDBAG_REF_MIN = 1e-2  # |ref|.max() below this — ref itself is ~zero, can't detect sandbagging
SANDBAG_AGENT_MAX = 1e-6  # |agent|.max() below this with non-trivial ref → sandbagging

# ─── Difficulty Tiers ─────────────────────────────────────────────────────────


class Level:
    def __init__(self, id, name, kernel_file, ref_file, shapes, family):
        self.id = id
        self.name = name
        self.kernel_file = kernel_file  # Triton source — agent reads this
        self.ref_file = ref_file  # PyTorch nn.Module — eval compares against this
        self.shapes = shapes  # list of 3 shapes (tuples of (n_rows, n_cols))
        self.family = family  # short tag for the thesis table


LEVELS = [
    Level(
        id=0,
        name="Negate (elementwise multi-row)",
        kernel_file="triton_testing_layers/triton_negate_kernel.py",
        ref_file="triton_testing_layers/triton_negate_ref.py",
        shapes=[(4, 16), (1, 16), (8, 16)],
        family="elementwise",
    ),
    Level(
        id=1,
        name="Abs (HW-native unary)",
        kernel_file="triton_testing_layers/triton_abs_kernel.py",
        ref_file="triton_testing_layers/triton_abs_ref.py",
        shapes=[(4, 16), (1, 16), (8, 16)],
        family="elementwise",
    ),
    Level(
        id=2,
        name="ReLU (predicate vs scalar zero)",
        kernel_file="triton_testing_layers/triton_relu_kernel.py",
        ref_file="triton_testing_layers/triton_relu_ref.py",
        shapes=[(4, 16), (1, 16), (8, 16)],
        family="elementwise",
    ),
    Level(
        id=3,
        name="Clip (parametric min/max)",
        kernel_file="triton_testing_layers/triton_clip_kernel.py",
        ref_file="triton_testing_layers/triton_clip_ref.py",
        shapes=[(4, 16), (1, 16), (8, 16)],
        family="elementwise",
    ),
    Level(
        id=4,
        name="Row-max reduction (TLM-verified pattern)",
        kernel_file="triton_testing_layers/triton_max_reduction_kernel.py",
        ref_file="triton_testing_layers/triton_max_reduction_ref.py",
        shapes=[(4, 16), (2, 16), (8, 16)],
        family="reduction",
    ),
    Level(
        id=5,
        name="AbsMaxNorm (reduce-then-broadcast)",
        kernel_file="triton_testing_layers/absmax_norm_kernel.py",
        ref_file="triton_testing_layers/absmax_norm_ref.py",
        shapes=[(4, 16), (2, 16), (8, 16)],
        family="norm",
    ),
    Level(
        id=6,
        name="Softmax (reduce + LUT exp + scalar broadcast)",
        kernel_file="triton_testing_layers/triton_softmax_kernel.py",
        ref_file="triton_testing_layers/triton_softmax_ref.py",
        shapes=[(4, 16), (2, 16), (8, 16)],
        family="norm",
    ),
]

# ─── Shared helpers ───────────────────────────────────────────────────────────


def _extract_code(text: str) -> str | None:
    """Extract the last ```python ... ``` block from an LLM response."""
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def _response_text(response) -> str:
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── VPU evaluation (structured) ──────────────────────────────────────────────
# This is the core pass/fail check. Mirrors ``run_vpu_evaluation`` in
# langgraph_agent/tools.py but returns a dict (max_diff, n_kernels, sandbagging
# flag) instead of a string, so the benchmark can record proper metrics.


def evaluate_vpu_kernel(
    output_code: str,
    ref_path: Path,
    shape: tuple,
    dtype: str = EVAL_DTYPE,
) -> dict:
    """Run a generated VpuKernel against a PyTorch fp32 reference.

    Returns a dict:
      {
        "passed": bool,
        "max_diff": float | None,
        "n_kernels": int | None,
        "agent_abs_max": float | None,   # for sandbagging detection
        "ref_abs_max": float | None,
        "sandbagging": bool,             # all-zero agent output, non-zero ref
        "error": str | None,
      }
    """
    import ast as _ast
    import importlib.util
    import tempfile

    result: dict = {
        "passed": False,
        "max_diff": None,
        "n_kernels": None,
        "agent_abs_max": None,
        "ref_abs_max": None,
        "sandbagging": False,
        "error": None,
    }

    try:
        _ast.parse(output_code)
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        return result

    if "VpuKernel" not in output_code:
        result["error"] = "no VpuKernel in output"
        return result
    if "build_graph" not in output_code:
        result["error"] = "no build_graph(input_type) defined"
        return result

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(output_code)
        agent_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_vpu_bench_kernel", agent_path)
        if spec is None or spec.loader is None:
            result["error"] = "could not load agent module"
            return result
        agent_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_mod)  # type: ignore[union-attr]

        if not hasattr(agent_mod, "build_graph"):
            result["error"] = "build_graph not defined at module level"
            return result

        import tensordyne.ir as ir
        from tensordyne.ir.vpu import VpuKernel

        if dtype == "rfp16":
            ir_dtype = ir.Rfp16(-15)
        elif dtype == "int16":
            ir_dtype = ir.Int16()
        else:
            result["error"] = f"unsupported dtype '{dtype}'"
            return result

        input_type = ir.Type(ir_dtype, shape)

        try:
            graph = agent_mod.build_graph(input_type)
        except Exception:
            result["error"] = f"build_graph raised:\n{traceback.format_exc()[-1500:]}"
            return result

        if not isinstance(graph, ir.RootGraph):
            result["error"] = f"build_graph returned {type(graph).__name__}, expected ir.RootGraph"
            return result

        kernels = [sg for sg in graph.iterate_subgraphs() if isinstance(sg, VpuKernel)]
        if not kernels:
            result["error"] = "graph contains no VpuKernel subgraphs"
            return result
        result["n_kernels"] = len(kernels)

        from tensordyne.ir.op import Op as _IROp

        input_ops = []
        for node in graph:
            if isinstance(node, _IROp) and type(node.instruction).__name__ == "Input":
                input_ops.append(node)
        if not input_ops:
            result["error"] = "graph has no Input nodes"
            return result
        input_types = [op.output_type for op in input_ops]

        import torch

        from tensordyne import runner
        from tensordyne.runner.testing.numeric import create_random_tensors

        try:
            inputs = create_random_tensors(input_types, seed=42)
        except Exception:
            result["error"] = f"create_random_tensors failed:\n{traceback.format_exc()[-1500:]}"
            return result

        try:
            runnable = runner.transpile_to_torch_runnable(graph)
            agent_out = runnable(*inputs)
            if isinstance(agent_out, tuple):
                agent_out = agent_out[0]
        except Exception:
            result["error"] = f"runner execution failed:\n{traceback.format_exc()[-1500:]}"
            return result

        ref_spec = importlib.util.spec_from_file_location("_vpu_bench_ref", str(ref_path))
        if ref_spec is None or ref_spec.loader is None:
            result["error"] = f"could not load reference at {ref_path}"
            return result
        ref_mod = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_mod)  # type: ignore[union-attr]

        ref_class = None
        for attr in vars(ref_mod).values():
            if isinstance(attr, type) and issubclass(attr, torch.nn.Module) and attr is not torch.nn.Module:
                ref_class = attr
                break
        if ref_class is None:
            result["error"] = f"no torch.nn.Module subclass found in {ref_path}"
            return result

        try:
            ref_model = ref_class()
            ref_inputs = [t.to(torch.float32) for t in inputs]
            ref_out = ref_model(*ref_inputs)
        except Exception:
            result["error"] = f"reference model failed:\n{traceback.format_exc()[-1500:]}"
            return result

        agent_f = agent_out.to(torch.float32)
        if agent_f.shape != ref_out.shape:
            result["error"] = f"shape mismatch — agent={tuple(agent_f.shape)}, ref={tuple(ref_out.shape)}"
            return result
        max_diff = float((agent_f - ref_out).abs().max().item())
        agent_abs_max = float(agent_f.abs().max().item())
        ref_abs_max = float(ref_out.abs().max().item())
        result["max_diff"] = max_diff
        result["agent_abs_max"] = agent_abs_max
        result["ref_abs_max"] = ref_abs_max

        # Sandbagging: kernel returns ~zero while reference is clearly non-zero.
        # See user memory project_vpu_agent.md (2026-05-12 softmax sandbagging case).
        sandbag = ref_abs_max > SANDBAG_REF_MIN and agent_abs_max < SANDBAG_AGENT_MAX
        result["sandbagging"] = sandbag

        if sandbag:
            result["error"] = f"sandbagging: agent_max={agent_abs_max:.2e} ref_max={ref_abs_max:.2e}"
            return result

        result["passed"] = max_diff < PASS_TOL
        if not result["passed"] and result["error"] is None:
            result["error"] = f"numeric mismatch: max_diff={max_diff:.2e}"
        return result

    except Exception:
        result["error"] = f"unexpected:\n{traceback.format_exc()[-1500:]}"
        return result
    finally:
        Path(agent_path).unlink(missing_ok=True)


def evaluate_shapes(level: Level, output_code: str) -> list[dict]:
    """Run the generated build_graph against all robustness shapes."""
    ref_path = TESTS_DIR / level.ref_file
    out = []
    for shape in level.shapes:
        r = evaluate_vpu_kernel(output_code, ref_path, tuple(shape), dtype=EVAL_DTYPE)
        out.append(
            {
                "shape": list(shape),
                "passed": bool(r["passed"]),
                "max_diff": r["max_diff"],
                "n_kernels": r["n_kernels"],
                "agent_abs_max": r["agent_abs_max"],
                "ref_abs_max": r["ref_abs_max"],
                "sandbagging": bool(r["sandbagging"]),
                "error": (r["error"] or "")[:400],
            }
        )
    return out


# ─── Prompting Approach (VPU) ─────────────────────────────────────────────────
# Single LLM call, no KB, no tools. Tight system prompt covering the IR skeleton
# and the most common pyxis.vpu.* ops.

_PROMPTING_VPU_SYSTEM = """\
You are an expert engineer writing low-level VPU kernels for the Tensordyne \
Pyxis chip. Given a Triton GPU kernel, write a Python function \
`build_graph(input_type)` returning an `ir.RootGraph` containing one or more \
`ir.VpuKernel` subgraphs that implement the same computation.

== Target hardware (the VPU) ==
- Processes tensors row-by-row through a FIFO.
- Inside a VpuKernel, every reference to an input pops a row from the FIFO —
  explicit Load() AND implicit references in op args.
- Loops over rows use `ir.RepeatGraph(parent=vpu_kernel, times=N)`.
- Reductions use an accumulator pattern: LoadZero → RepeatGraph → Add/Max + Assign.
- A second sequential VpuKernel is needed when a later step depends on a value
  produced by reducing an earlier step.

== Critical IR rules ==

1. SHAPE RULE (no RepeatGraph): every op's `output_type` is `(1, n_cols)`,
   NOT the full input shape. The compiler enforces this even though the runner
   may be lenient.

2. SHAPE RULE (inside RepeatGraph): every op's `output_type` equals the
   full input shape. Opposite of rule 1.

3. PREFER HW-NATIVE OPS. Use pyxis.vpu.Absolute, pyxis.vpu.Max, pyxis.vpu.Min,
   pyxis.vpu.Exp, pyxis.vpu.Mul, pyxis.vpu.Add, pyxis.vpu.Sub directly. Do NOT
   compose abs() out of sqrt(x*x) or max(a,b) out of 0.5*(a+b+|a-b|).

4. SCALAR LITERALS use insert_literal: e.g. for clip(x, lo, hi) emit a row of
   `lo` and `hi` literals, then Min/Max against them.

== Required output skeleton ==

```python
import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    vpu_type = ir.Type(input_type.dtype, tuple(input_type.shape))
    row_type = ir.Type(input_type.dtype, (1, input_type.shape[-1]))

    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))

    with ir.VpuKernel(parent=graph) as vpu_kernel:
        # For multi-row elementwise: wrap ops in a RepeatGraph and use vpu_type.
        # For reduction-to-one-row: unrolled chain, use row_type.
        load_0 = vpu_kernel.insert(
            ir.instructions.pyxis.vpu.Load(),
            args=(input_0,),
            output_type=row_type,  # or vpu_type inside RepeatGraph
        )
        # ... more pyxis.vpu.* ops ...

    graph.insert(ir.instructions.Output(), args=(result,), output_type=...)
    return graph
```

== Common pyxis.vpu ops ==

  pyxis.vpu.Load()                       # pops a FIFO row
  pyxis.vpu.LoadZero()                   # accumulator initializer
  pyxis.vpu.Assign()                     # materialize a value for reuse
  pyxis.vpu.Add() / Sub() / Mul()        # binary elementwise
  pyxis.vpu.Max() / Min()                # binary elementwise (also for reductions)
  pyxis.vpu.Absolute()                   # |x|
  pyxis.vpu.Negate()                     # -x
  pyxis.vpu.Exp()                        # LUT-backed; use for softmax
  pyxis.vpu.Mac()                        # multiply-accumulate

== Anti-patterns ==
- DO NOT write nn.Module — write `build_graph` returning an ir.RootGraph.
- DO NOT use @register_lowering — this is direct IR emission.
- DO NOT import torch in the output.
- DO NOT compose ops to avoid using a HW-native primitive.

Output ONLY a single ```python ... ``` code block containing imports and \
`build_graph(input_type)`.\
"""


def run_prompting(level: Level, output_path: Path) -> tuple[bool, int, int, str]:
    """Returns (code_generated, input_tokens, output_tokens, code_or_empty)."""
    import anthropic

    kernel_src = (TESTS_DIR / level.kernel_file).read_text()
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=_PROMPTING_VPU_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate this Triton kernel to a Tensordyne VPU "
                    f"`build_graph(input_type)`:\n\n```python\n{kernel_src}\n```"
                ),
            }
        ],
    )
    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    code = _extract_code(_response_text(response))
    if not code:
        return False, in_tok, out_tok, ""
    output_path.write_text(code)
    return True, in_tok, out_tok, code


# ─── RAG Approach (VPU) ───────────────────────────────────────────────────────
# Single generation call, with retrieved context from vpu_kb. No tools, no loop.

_RAG_VPU_SYSTEM = """\
You are an expert engineer writing low-level VPU kernels for the Tensordyne \
Pyxis chip. Given a Triton GPU kernel, write `build_graph(input_type)` \
returning an `ir.RootGraph` with `ir.VpuKernel` subgraphs that implement the \
same computation.

You do NOT write an nn.Module. You write the IR graph directly using \
pyxis.vpu.* instructions inside a VpuKernel.

== Required skeleton ==
```python
import tensordyne.ir as ir

def build_graph(input_type: ir.Type) -> ir.RootGraph:
    graph = ir.RootGraph()
    input_0 = graph.insert(ir.instructions.Input(output_type=input_type))
    with ir.VpuKernel(parent=graph) as vpu_kernel:
        ...
    graph.insert(ir.instructions.Output(), args=(result,), output_type=...)
    return graph
```

== Retrieved VPU knowledge base entries ==

{kb_context}

== Anti-patterns ==
- DO NOT compose abs out of sqrt(x*x) or max out of 0.5*(a+b+|a-b|). Use HW-native ops.
- DO NOT import torch.
- DO NOT write nn.Module or @register_lowering.

Output ONLY one ```python ... ``` block with imports + `build_graph(input_type)`.\
"""


def _load_vpu_db():
    """Load the vpu_kb FAISS DB (same embedding model as build_vpu_kb.py)."""
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    emb = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(str(VPU_KB_PATH), embeddings=emb, allow_dangerous_deserialization=True)


def run_rag(level: Level, output_path: Path, db) -> tuple[bool, int, int, str]:
    """Returns (code_generated, input_tokens, output_tokens, code_or_empty)."""
    import anthropic

    kernel_src = (TESTS_DIR / level.kernel_file).read_text()
    client = anthropic.Anthropic()
    total_in, total_out = 0, 0

    # Step 1 — extract query terms (pyxis.vpu ops + pattern names) from the kernel
    extract_resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "Inspect this Triton kernel and list the pyxis.vpu primitives and "
                    "pattern names that would be needed to translate it for the "
                    "Tensordyne VPU. Use names like 'pyxis.vpu.Max', 'pyxis.vpu.Exp', "
                    "'vpu_elementwise_multi_row', 'vpu_unrolled_reduction', "
                    "'vpu_rule_fifo_pop', 'vpu_horizontal_reduce_via_axis_transpose'. "
                    "Return ONLY a JSON array of short strings.\n\n"
                    f"```python\n{kernel_src}\n```"
                ),
            }
        ],
    )
    total_in += extract_resp.usage.input_tokens
    total_out += extract_resp.usage.output_tokens

    raw = _response_text(extract_resp).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        queries = json.loads(raw)
    except Exception:
        queries = []

    # Step 2 — retrieve KB chunks (dedup by first 150 chars)
    seen, chunks = set(), []
    for q in queries[:12]:
        for doc in db.similarity_search(q, k=3):
            text = doc.page_content.strip()
            sig = text[:150]
            if sig not in seen and len(text) >= 40:
                seen.add(sig)
                chunks.append(text)
    kb_context = "\n\n---\n\n".join(chunks[:20])

    # Step 3 — generate
    gen_resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=_RAG_VPU_SYSTEM.replace("{kb_context}", kb_context),
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate this Triton kernel to a Tensordyne VPU "
                    f"`build_graph(input_type)`:\n\n```python\n{kernel_src}\n```"
                ),
            }
        ],
    )
    total_in += gen_resp.usage.input_tokens
    total_out += gen_resp.usage.output_tokens

    code = _extract_code(_response_text(gen_resp))
    if not code:
        return False, total_in, total_out, ""
    output_path.write_text(code)
    return True, total_in, total_out, code


# ─── Agentic Approach (VPU) ───────────────────────────────────────────────────


def run_agentic(level: Level, output_path: Path, agent) -> tuple[bool, int | None, int | None, int, int, str]:
    """Returns (code_generated, outer_iterations, total_tool_calls, in_tok, out_tok, code)."""
    code_generated = agent.run(
        input_path=str(TESTS_DIR / level.kernel_file),
        output_path=str(output_path),
        input_shape=level.shapes[0],  # training shape; shapes[1:] are robustness
        eval_input_file=str(TESTS_DIR / level.ref_file),
        mode="vpu_lowering",
        kb_path=str(VPU_KB_PATH),
    )
    iterations = getattr(agent, "last_iteration_count", None)
    tool_calls = getattr(agent, "last_tool_calls", None)
    in_tok = getattr(agent, "last_input_tokens", 0)
    out_tok = getattr(agent, "last_output_tokens", 0)

    # The agent may have written its output even on failure (last attempt).
    code = output_path.read_text() if output_path.exists() else ""
    return code_generated, iterations, tool_calls, in_tok, out_tok, code


# ─── Main Runner ──────────────────────────────────────────────────────────────


def run_benchmark(approaches: list[str], level_ids: list[int], resume: bool = True):
    OUTPUTS_DIR.mkdir(exist_ok=True)

    existing: list[dict] = []
    if resume and RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
        print(f"Resuming: {len(existing)} results already saved.\n")

    def already_done(approach, level_id, run_idx):
        return any(
            r["approach"] == approach and r["level_id"] == level_id and r["run_idx"] == run_idx for r in existing
        )

    # Heavy resources: load once
    agent = None
    rag_db = None
    if "agentic" in approaches:
        from langgraph_agent.agent import LangGraphAgent

        agent = LangGraphAgent(max_iterations=4, verbose=True)
    if "rag" in approaches:
        print(f"Loading VPU FAISS KB from {VPU_KB_PATH} ...")
        rag_db = _load_vpu_db()

    levels = [lv for lv in LEVELS if lv.id in level_ids]

    total_runs = len(approaches) * len(levels) * N_RUNS
    done_runs = sum(1 for a in approaches for lv in levels for r in range(N_RUNS) if already_done(a, lv.id, r))
    completed = done_runs
    run_times: list[float] = []
    benchmark_start = time.monotonic()

    for approach in approaches:
        for level in levels:
            print(f"\n{'=' * 60}")
            print(f"  {approach.upper()}  |  L{level.id}: {level.name}")
            print(f"{'=' * 60}")

            for run_idx in range(N_RUNS):
                if resume and already_done(approach, level.id, run_idx):
                    print(f"  Run {run_idx + 1}/{N_RUNS}: already done, skipping.")
                    completed += 1
                    continue

                remaining = total_runs - completed
                if run_times:
                    avg_s = sum(run_times) / len(run_times)
                    eta_s = int(avg_s * remaining)
                    eta_str = f"  ETA ~{eta_s // 60}m{eta_s % 60:02d}s"
                else:
                    eta_str = ""

                print(f"\n  Run {run_idx + 1}/{N_RUNS}  [{completed + 1}/{total_runs} total{eta_str}]  {_now()}")

                out_dir = OUTPUTS_DIR / approach / f"level{level.id}" / f"run{run_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = out_dir / "output.py"

                record: dict = {
                    "approach": approach,
                    "model": MODEL,
                    "level_id": level.id,
                    "level_name": level.name,
                    "family": level.family,
                    "run_idx": run_idx,
                    "timestamp": _now(),
                    "code_generated": False,
                    "output_file": str(output_path.relative_to(TESTS_DIR)),
                    "shape_results": [],
                    "outer_iterations": None,
                    "tool_calls": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_s": 0.0,
                }

                t0 = time.monotonic()
                code = ""
                try:
                    if approach == "prompting":
                        ok, in_tok, out_tok, code = run_prompting(level, output_path)
                        record["code_generated"] = ok
                        record["input_tokens"] = in_tok
                        record["output_tokens"] = out_tok

                    elif approach == "rag":
                        ok, in_tok, out_tok, code = run_rag(level, output_path, rag_db)
                        record["code_generated"] = ok
                        record["input_tokens"] = in_tok
                        record["output_tokens"] = out_tok

                    elif approach == "agentic":
                        ok, iters, tool_calls, in_tok, out_tok, code = run_agentic(level, output_path, agent)
                        record["code_generated"] = ok
                        record["outer_iterations"] = iters
                        record["tool_calls"] = tool_calls
                        record["input_tokens"] = in_tok
                        record["output_tokens"] = out_tok

                    record["elapsed_s"] = round(time.monotonic() - t0, 1)

                    if code and output_path.exists():
                        print(f"     Evaluating {len(level.shapes)} shapes ...")
                        record["shape_results"] = evaluate_shapes(level, code)
                        n_passed = sum(1 for s in record["shape_results"] if s["passed"])
                        run_ok = n_passed == len(level.shapes)
                        sandbag = any(s["sandbagging"] for s in record["shape_results"])
                        sb_str = "  ⚠️ SANDBAG" if sandbag else ""

                        extra = ""
                        if approach == "agentic":
                            extra = f"  iters={record['outer_iterations']} tools={record['tool_calls']}"
                        tok_info = (
                            f"in={record['input_tokens']:,}  out={record['output_tokens']:,}  {record['elapsed_s']}s"
                        )
                        print(
                            f"     Shapes: {n_passed}/{len(level.shapes)} passed  "
                            f"{'✅ PASS' if run_ok else '❌ FAIL'}{sb_str}{extra}  [{tok_info}]"
                        )
                    else:
                        print(f"     ❌ Code generation failed  [{record['elapsed_s']}s]")

                except Exception:
                    record["elapsed_s"] = round(time.monotonic() - t0, 1)
                    record["traceback"] = traceback.format_exc()[:2000]
                    print("     ❌ Unexpected error (traceback saved to results JSON)")

                run_times.append(record["elapsed_s"])
                completed += 1
                existing.append(record)
                RESULTS_FILE.write_text(json.dumps(existing, indent=2))

    total_elapsed = int(time.monotonic() - benchmark_start)
    print(f"\nBenchmark complete in {total_elapsed // 60}m{total_elapsed % 60:02d}s. Results → {RESULTS_FILE}")
    _print_token_summary(existing, approaches, level_ids)


def _print_token_summary(results: list[dict], approaches: list[str], level_ids: list[int]):
    print(f"\n{'=' * 50}")
    print("  TOKEN SUMMARY (VPU benchmark)")
    print(f"{'=' * 50}")
    for approach in approaches:
        runs = [r for r in results if r["approach"] == approach and r["level_id"] in level_ids]
        total_in = sum(r.get("input_tokens", 0) for r in runs)
        total_out = sum(r.get("output_tokens", 0) for r in runs)
        n_sandbag = sum(
            1 for r in runs for s in r.get("shape_results", []) if s.get("sandbagging")
        )
        print(f"  {approach:<12}  in={total_in:>10,}  out={total_out:>8,}  sandbag_shapes={n_sandbag}")
    print(f"{'=' * 50}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach",
        nargs="+",
        choices=["prompting", "rag", "agentic"],
        default=["prompting", "rag", "agentic"],
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=list(range(len(LEVELS))),
        help="Tier IDs to run (0-6)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore saved results and rerun from scratch",
    )
    args = parser.parse_args()

    run_benchmark(
        approaches=args.approach,
        level_ids=args.levels,
        resume=not args.no_resume,
    )
