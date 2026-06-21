#!/usr/bin/env python3
"""Thesis Benchmark: Prompting vs RAG vs Agentic transpilation.

Evaluation protocol:
  - 7 difficulty levels (Level 0: Single Linear → Level 6: LlamaForCausalLM)
  - 3 independent runs per level per approach (handles LLM non-determinism)
  - 3 input shapes per run (robustness check)
  - A run PASSES if the generated code passes all 3 shapes
  - A level PASSES for an approach if >= 2/3 runs pass

Usage:
  python benchmark_eval.py                          # all approaches, all levels
  python benchmark_eval.py --approach agentic       # one approach only
  python benchmark_eval.py --levels 0 1 2           # specific levels
  python benchmark_eval.py --no-resume              # rerun everything from scratch
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ─── NOTE ─────────────────────────────────────────────────────────────────────
# json is imported above and used throughout — do not re-import below.

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

# Load ANTHROPIC_API_KEY / OPENAI_API_KEY from tests/config/.env if present, so
# the script works without the caller exporting them into the shell first.
try:
    from dotenv import load_dotenv

    load_dotenv(TESTS_DIR / "config" / ".env")
except ImportError:
    pass

from transpile_evaluation_helper import run_transpiled_evaluation

# ─── Settings ────────────────────────────────────────────────────────────────

N_RUNS = 3
PASS_THRESHOLD = 2  # minimum passing runs to count a level as passed
RESULTS_FILE = TESTS_DIR / "benchmark_results.json"
OUTPUTS_DIR = TESTS_DIR / "benchmark_outputs"
MODEL = "claude-sonnet-4-6"  # same model for all three approaches

# ─── Difficulty Levels ────────────────────────────────────────────────────────


class Level:
    def __init__(self, id, name, input_file, shapes, config_file="", target_class=""):
        self.id = id
        self.name = name
        self.input_file = input_file  # relative to TESTS_DIR
        self.shapes = shapes  # list of 3 input shapes (tuples)
        self.config_file = config_file  # relative to TESTS_DIR
        self.target_class = target_class


LEVELS = [
    Level(
        id=0,
        name="Single Linear",
        input_file="testing_layers/single_linear.py",
        # shape[0] is the agent's training shape; shapes[1-2] are robustness checks
        shapes=[(4, 32), (1, 32), (8, 32)],  # typical, batch=1 edge, large batch
    ),
    Level(
        id=1,
        name="MLP (3-layer ReLU)",
        input_file="testing_layers/linear_layer.py",
        shapes=[(4, 64), (1, 64), (8, 64)],  # typical, batch=1 edge, large batch
    ),
    Level(
        id=2,
        name="Single-Head Attention",
        input_file="testing_layers/attention_layer.py",
        shapes=[(2, 10, 64), (2, 1, 64), (4, 7, 64)],  # typical, seq=1 edge, odd seq
    ),
    Level(
        id=3,
        name="Gated MLP + RMSNorm",
        input_file="testing_layers/torch_gated.py",
        shapes=[(2, 10, 64), (2, 1, 64), (4, 7, 64)],
    ),
    Level(
        id=4,
        name="Multi-Head Attention + RMSNorm",
        input_file="testing_layers/multiattention_layer.py",
        shapes=[(2, 10, 64), (2, 1, 64), (4, 7, 64)],
    ),
    Level(
        id=5,
        name="Llama Decoder Layer",
        input_file="testing_layers/llama_decoder.py",
        shapes=[(2, 10, 64), (2, 1, 64), (4, 7, 64)],
    ),
    # L6 (LlamaForCausalLM) intentionally dropped — too similar to L5 to be a
    # distinct difficulty tier (both are Llama; L6 just wraps L5 with embedding
    # + tied head + stacked layers). Removed for the thesis writeup.
]

# ─── Shared helpers ───────────────────────────────────────────────────────────


def _extract_code(text: str) -> str | None:
    """Extract the last ```python ... ``` block from an LLM response."""
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def _response_text(response) -> str:
    """Return the text content from an Anthropic response, skipping non-text blocks."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Prompting Approach ───────────────────────────────────────────────────────
# One well-crafted prompt — no retrieval, no self-correction, single LLM call.

_PROMPTING_SYSTEM = """\
You are an expert compiler engineer. Translate PyTorch nn.Module models to the \
Tensordyne neural network API. Tensordyne is a compiled tensor DSL — models are \
traced to a computation graph, so all shapes must be statically resolvable.

== Module structure ==

  import tensordyne.nn as nn

  class MyModel(nn.Module):
      def __init__(self, ...):
          super().__init__()
          self.fc = nn.Linear(64, 128)     # sub-modules declared here

      def build(self, x):                  # NOTE: build(), not forward()
          return self.fc.build(x)          # NOTE: .build(x), not self.fc(x)

== Available modules ==

  nn.Linear(in_features, out_features, bias=True)
  nn.RMSNorm(normalized_shape, *, eps=1e-6)
  nn.LayerNorm(normalized_shape, *, eps=1e-5)
  nn.Embedding(num_embeddings, embedding_dim)
  nn.Parameter(shape)                      # e.g. nn.Parameter((64,)) replaces torch.nn.Parameter(torch.ones(64))

== Available ops (functions, not methods) ==

  # Shape manipulation
  nn.reshape(x, shape)                     # shape is a tuple/list of ints
  nn.transpose(x, dim0, dim1)
  nn.permute(x, dims)                      # dims is a tuple/list
  nn.flatten(x)
  nn.squeeze(x, dim)
  nn.unsqueeze(x, dim)
  nn.chunk(x, chunks, dim)                 # returns list of tensors
  nn.split(x, split_size, dim)
  nn.cat(tensors, *, dim=0)                # tensors is a list; dim is keyword-only
  nn.pad(x, pad, *, mode="constant", value=0)

  # Elementwise / math
  nn.add(a, b)
  nn.sub(a, b)
  nn.mul(a, b)
  nn.div(a, b)
  nn.matmul(a, b)
  nn.sqrt(x)
  nn.rsqrt(x)
  nn.abs(x)
  nn.exp(x)
  nn.log(x)
  nn.pow(x, exponent)
  nn.where(condition, x, y)

  # Activations
  nn.relu(x)
  nn.gelu(x)
  nn.silu(x)
  nn.tanh(x)
  nn.sigmoid(x)
  nn.softmax(x, *, dim)                    # dim is keyword-only

  # Reductions
  nn.reduce_mean(x, *, dim, keepdim=False) # dim and keepdim are keyword-only
  nn.reduce_sum(x, *, dim, keepdim=False)
  nn.mean(x, *, dim, keepdim=False)

  # Attention / masking
  nn.scaled_dot_product_attention(q, k, v, *, is_causal=False)
  nn.triu(x, diagonal=0)
  nn.tril(x, diagonal=0)

  # Indexing / broadcasting
  nn.repeat_interleave(x, repeats, dim)
  nn.insert_literal(value, shape, dtype)   # creates a constant tensor (replaces torch.full / torch.arange)

== CRITICAL rules ==

1. SAME ATTRIBUTE NAMES: keep self.fc1, self.q_proj, etc. exactly as in PyTorch.
   Weight loading matches by attribute name — any rename breaks it.
2. NO torch.* or F.*: every tensor operation must use nn.* functions.
3. KEYWORD-ONLY ARGS: args after * in the signatures above MUST be passed by name.
   CORRECT:   nn.softmax(x, dim=-1)       ✓  (dim= named explicitly)
   CORRECT:   nn.cat(tensors, dim=0)      ✓
   WRONG:     nn.softmax(x, -1)           ✗  (positional — will error)
4. NO DROPOUT: Tensordyne does not support dropout at trace time — omit it entirely.
5. NO UNPACKING SHAPES from symbolic tensors: do not write `b, s, d = x.shape`.
   Use concrete constructor arguments or nn.reshape with known dimensions instead.
6. CUSTOM RMSNorm: do not implement RMSNorm manually — use nn.RMSNorm.

Output ONLY a single Python code block: ```python ... ```\
"""


def run_prompting(level: Level, output_path: Path) -> tuple[bool, int, int]:
    """Returns (success, input_tokens, output_tokens)."""
    import anthropic

    src = (TESTS_DIR / level.input_file).read_text()
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=_PROMPTING_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Translate this PyTorch model to Tensordyne:\n\n```python\n{src}\n```",
            }
        ],
    )
    input_tok = response.usage.input_tokens
    output_tok = response.usage.output_tokens
    code = _extract_code(_response_text(response))
    if not code:
        return False, input_tok, output_tok
    output_path.write_text(code)
    return True, input_tok, output_tok


# ─── RAG Approach ─────────────────────────────────────────────────────────────
# Same model. Pipeline: extract ops → FAISS KB lookup → single generation call.
# No self-correction — one shot with retrieved context.

_RAG_SYSTEM = """\
You are an expert compiler engineer. Translate PyTorch nn.Module models to the \
Tensordyne neural network API. Tensordyne is a compiled tensor DSL — models are \
traced to a computation graph, so all shapes must be statically resolvable.

== Module structure ==

  import tensordyne.nn as nn

  class MyModel(nn.Module):
      def __init__(self, ...):
          super().__init__()
          self.fc = nn.Linear(64, 128)

      def build(self, x):                  # NOTE: build(), not forward()
          return self.fc.build(x)          # NOTE: .build(x), not self.fc(x)

== CRITICAL rules ==

1. SAME ATTRIBUTE NAMES: keep self.fc1, self.q_proj, etc. exactly as in PyTorch.
2. NO torch.* or F.*: every tensor operation must use nn.* functions.
3. KEYWORD-ONLY ARGS: args after * in signatures MUST be passed by name.
4. NO DROPOUT: omit entirely.
5. NO UNPACKING SHAPES: do not write `b, s, d = x.shape`.
6. CUSTOM RMSNorm: use nn.RMSNorm, never implement manually.

== Retrieved Tensordyne API documentation ==

{kb_context}

Output ONLY a single Python code block: ```python ... ```\
"""


def run_rag(level: Level, output_path: Path) -> tuple[bool, int, int]:
    """Returns (success, input_tokens, output_tokens)."""
    import anthropic
    from langgraph_agent.tools import _get_db  # reuses the same FAISS KB as the agent

    src = (TESTS_DIR / level.input_file).read_text()
    db = _get_db()
    client = anthropic.Anthropic()
    total_in, total_out = 0, 0

    # Step 1 — extract op names as Tensordyne KB query terms (lightweight call)
    extract_resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "List the Tensordyne nn operations needed to translate this PyTorch model. "
                    "Use Tensordyne naming: nn.Linear, nn.RMSNorm, nn.softmax, nn.matmul, nn.silu, etc. "
                    "Return ONLY a JSON array of short op names. No explanation.\n\n"
                    f"```python\n{src}\n```"
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
        ops = json.loads(raw)
    except Exception:
        ops = []

    # Step 2 — retrieve KB docs per op (k=3, deduplicated)
    seen, chunks = set(), []
    for op in ops[:12]:
        for doc in db.similarity_search(op, k=3):
            text = doc.page_content.strip()
            sig = text[:150]
            if sig not in seen and len(text) >= 40:
                seen.add(sig)
                chunks.append(text)

    kb_context = "\n\n---\n\n".join(chunks[:20])

    # Step 3 — generate (single call with retrieved context injected into system prompt)
    gen_resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=_RAG_SYSTEM.replace("{kb_context}", kb_context),
        messages=[
            {
                "role": "user",
                "content": f"Translate this PyTorch model to Tensordyne:\n\n```python\n{src}\n```",
            }
        ],
    )
    total_in += gen_resp.usage.input_tokens
    total_out += gen_resp.usage.output_tokens

    code = _extract_code(_response_text(gen_resp))
    if not code:
        return False, total_in, total_out
    output_path.write_text(code)
    return True, total_in, total_out


# ─── Agentic Approach ─────────────────────────────────────────────────────────


def run_agentic(
    level: Level,
    output_path: Path,
    agent,
) -> tuple[bool, int | None, int | None, int, int]:
    """Returns (success, outer_iterations, total_tool_calls, input_tokens, output_tokens)."""
    config_file = str(TESTS_DIR / level.config_file) if level.config_file else ""
    success = agent.run(
        input_path=str(TESTS_DIR / level.input_file),
        output_path=str(output_path),
        input_shape=level.shapes[0],
        config_file=config_file,
        target_class=level.target_class,
    )
    iterations = getattr(agent, "last_iteration_count", None)
    tool_calls = getattr(agent, "last_tool_calls", None)
    input_tok = getattr(agent, "last_input_tokens", 0)
    output_tok = getattr(agent, "last_output_tokens", 0)
    return success, iterations, tool_calls, input_tok, output_tok


# ─── Shape Evaluation ─────────────────────────────────────────────────────────


def evaluate_shapes(level: Level, output_path: Path) -> list[dict]:
    """Run the generated file against all 3 robustness shapes."""
    config_file = str(TESTS_DIR / level.config_file) if level.config_file else ""
    results = []
    for shape in level.shapes:
        try:
            r = run_transpiled_evaluation(
                input_file=str(TESTS_DIR / level.input_file),
                output_file=str(output_path),
                input_shape=tuple(shape),
                verbose=False,
                config_file=config_file,
                target_class=level.target_class,
            )
            results.append(
                {
                    "shape": list(shape),
                    "passed": bool(r.passed),
                    "max_diff": float(r.max_diff) if r.max_diff is not None else None,
                    "error": (r.error or "")[:300],
                }
            )
        except Exception as e:
            results.append(
                {
                    "shape": list(shape),
                    "passed": False,
                    "max_diff": None,
                    "error": str(e)[:300],
                }
            )
    return results


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

    # Build the LangGraph agent once — loading embeddings and KB is expensive
    agent = None
    if "agentic" in approaches:
        from langgraph_agent.agent import LangGraphAgent

        agent = LangGraphAgent(max_iterations=3, verbose=True)

    levels = [l for l in LEVELS if l.id in level_ids]

    # Total work units for progress tracking
    total_runs = len(approaches) * len(levels) * N_RUNS
    done_runs = sum(1 for a in approaches for l in levels for r in range(N_RUNS) if already_done(a, l.id, r))
    completed = done_runs
    run_times: list[float] = []  # elapsed seconds per completed run (for ETA)
    benchmark_start = time.monotonic()

    for approach in approaches:
        for level in levels:
            print(f"\n{'=' * 60}")
            print(f"  {approach.upper()}  |  Level {level.id}: {level.name}")
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
                    "run_idx": run_idx,
                    "timestamp": _now(),
                    "code_generated": False,
                    "output_file": str(output_path.relative_to(TESTS_DIR)),
                    "shape_results": [],
                    # agentic-only fields (None for prompting/rag)
                    "outer_iterations": None,  # outer correction rounds used
                    "tool_calls": None,  # total inner tool calls (KB lookups, evals, etc.)
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_s": 0.0,
                }

                t0 = time.monotonic()
                try:
                    if approach == "prompting":
                        ok, in_tok, out_tok = run_prompting(level, output_path)
                        record["code_generated"] = ok
                        record["input_tokens"] = in_tok
                        record["output_tokens"] = out_tok

                    elif approach == "rag":
                        ok, in_tok, out_tok = run_rag(level, output_path)
                        record["code_generated"] = ok
                        record["input_tokens"] = in_tok
                        record["output_tokens"] = out_tok

                    elif approach == "agentic":
                        ok, iters, tool_calls, in_tok, out_tok = run_agentic(level, output_path, agent)
                        record["code_generated"] = ok
                        record["outer_iterations"] = iters
                        record["tool_calls"] = tool_calls
                        record["input_tokens"] = in_tok
                        record["output_tokens"] = out_tok

                    record["elapsed_s"] = round(time.monotonic() - t0, 1)

                    if record["code_generated"] and output_path.exists():
                        print(f"     Evaluating {len(level.shapes)} shapes...")
                        record["shape_results"] = evaluate_shapes(level, output_path)
                        n_passed = sum(1 for s in record["shape_results"] if s["passed"])
                        run_ok = n_passed == len(level.shapes)

                        extra = ""
                        if approach == "agentic":
                            extra = f"  outer_iters={record['outer_iterations']}  tool_calls={record['tool_calls']}"
                        tok_info = (
                            f"in={record['input_tokens']:,}  out={record['output_tokens']:,}  {record['elapsed_s']}s"
                        )
                        print(
                            f"     Shapes: {n_passed}/{len(level.shapes)} passed  "
                            f"{'✅ PASS' if run_ok else '❌ FAIL'}"
                            f"{extra}  [{tok_info}]"
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
    """Print a quick cost summary at the end of the run."""
    print(f"\n{'=' * 50}")
    print("  TOKEN SUMMARY")
    print(f"{'=' * 50}")
    for approach in approaches:
        runs = [r for r in results if r["approach"] == approach and r["level_id"] in level_ids]
        total_in = sum(r.get("input_tokens", 0) for r in runs)
        total_out = sum(r.get("output_tokens", 0) for r in runs)
        print(f"  {approach:<12}  in={total_in:>10,}  out={total_out:>8,}")
    print(f"{'=' * 50}")


# ─── CLI ─────────────────────────────────────────────────────────────────────


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
        help="Level IDs to run (0-6)",
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
