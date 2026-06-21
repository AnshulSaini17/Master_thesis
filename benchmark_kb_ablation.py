#!/usr/bin/env python3
"""
KB Ablation Study: how does the KB construction strategy affect transpilation quality?

Tests the RAG and Agentic approaches with three different knowledge bases:
  - api_kb        : manually curated (117 entries with examples and gotchas) [default]
  - api_kb_token_600 : token-based chunking of Tensordyne source
  - api_kb_ast    : AST-based chunking of Tensordyne source

Runs on a representative subset of levels (not all 7) to keep cost manageable.
Results are saved separately from the main benchmark (kb_ablation_results.json).

Usage:
  python benchmark_kb_ablation.py                      # all KBs, subset of levels
  python benchmark_kb_ablation.py --kb curated ast     # specific KBs only
  python benchmark_kb_ablation.py --levels 2 4 6       # specific levels
  python benchmark_kb_ablation.py --no-resume
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

from transpile_evaluation_helper import run_transpiled_evaluation

# ─── Settings ────────────────────────────────────────────────────────────────

N_RUNS         = 3
PASS_THRESHOLD = 2
RESULTS_FILE   = TESTS_DIR / "kb_ablation_results.json"
OUTPUTS_DIR    = TESTS_DIR / "benchmark_outputs" / "kb_ablation"
MODEL          = "claude-sonnet-4-6"

# KB variants — (display_name, kb_path_arg passed to configure_run / agent.run)
KB_VARIANTS = [
    ("curated",    None),          # api_kb — manually curated, 117 entries
    ("token_600",  "token_600"),   # api_kb_token_600 — 600-token chunking
    ("ast",        "ast"),         # api_kb_ast — AST-based chunking
]

# Approaches to test (prompting doesn't use KB so excluded here)
APPROACHES = ["rag", "agentic"]

# Representative subset of levels — pick ones that stress-test KB quality:
#   L2 needs attention ops, L4 needs multi-head + RMSNorm, L6 is the hardest
DEFAULT_LEVEL_IDS = [2, 4, 6]

# ─── Level definitions (copy from benchmark_eval.py) ─────────────────────────

class Level:
    def __init__(self, id, name, input_file, shapes, config_file="", target_class=""):
        self.id           = id
        self.name         = name
        self.input_file   = input_file
        self.shapes       = shapes
        self.config_file  = config_file
        self.target_class = target_class

ALL_LEVELS = [
    Level(
        id=0, name="Single Linear",
        input_file="testing_layers/single_linear.py",
        shapes=[(2, 32), (1, 32), (4, 32)],
    ),
    Level(
        id=1, name="MLP (3-layer ReLU)",
        input_file="testing_layers/linear_layer.py",
        shapes=[(2, 64), (1, 64), (4, 64)],
    ),
    Level(
        id=2, name="Single-Head Attention",
        input_file="testing_layers/attention_layer.py",
        shapes=[(2, 10, 64), (1, 50, 64), (4, 5, 64)],
    ),
    Level(
        id=3, name="Gated MLP + RMSNorm",
        input_file="testing_layers/torch_gated.py",
        shapes=[(2, 10, 64), (1, 50, 64), (4, 5, 64)],
    ),
    Level(
        id=4, name="Multi-Head Attention + RMSNorm",
        input_file="testing_layers/multiattention_layer.py",
        shapes=[(2, 10, 64), (1, 50, 64), (4, 5, 64)],
    ),
    Level(
        id=5, name="Llama Decoder Layer",
        input_file="testing_layers/llama_decoder.py",
        shapes=[(2, 10, 64), (1, 20, 64), (4, 5, 64)],
    ),
    Level(
        id=6, name="LlamaForCausalLM (HuggingFace)",
        input_file="testing_layers/modeling_llama.py",
        shapes=[(2, 10), (1, 20), (4, 5)],
        config_file="testing_layers/llama_config.json",
        target_class="LlamaForCausalLM",
    ),
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None

def _response_text(response) -> str:
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ─── RAG with configurable KB ─────────────────────────────────────────────────

_RAG_SYSTEM = """\
You are an expert compiler engineer. Translate a PyTorch nn.Module to Tensordyne API.
You will be given the source code and relevant Tensordyne API documentation.

RULES:
1. Use build(self, x) instead of forward(self, x)
2. Call sub-layers with self.layer.build(x) instead of self.layer(x)
3. Import: import tensordyne.nn as nn
4. Keep EXACTLY the same attribute names as PyTorch (weight loading depends on this)
5. Do not use any torch.* or F.* functions
6. Args marked keyword-only (after *) must always be passed by name
7. Do not add dropout

Output ONLY a single Python code block: ```python ... ```\
"""


def run_rag(level: Level, output_path: Path, kb_path_arg) -> tuple[bool, int, int]:
    """RAG with a specific KB. Returns (success, input_tokens, output_tokens)."""
    import anthropic
    from langgraph_agent.tools import configure_run, _get_db

    # Point the KB to the requested variant before loading
    configure_run(output_path=str(output_path), allowed_read_paths=[], kb_path=kb_path_arg)
    db = _get_db()
    src = (TESTS_DIR / level.input_file).read_text()
    client = anthropic.Anthropic()
    total_in, total_out = 0, 0

    # Step 1 — extract ops
    extract_resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "List every PyTorch operation used in this code. "
                "Return ONLY a JSON array of short op names, e.g. "
                "[\"Linear\", \"softmax\", \"matmul\"]. No explanation.\n\n"
                f"```python\n{src}\n```"
            ),
        }],
    )
    total_in  += extract_resp.usage.input_tokens
    total_out += extract_resp.usage.output_tokens
    raw = _response_text(extract_resp).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        ops = json.loads(raw)
    except Exception:
        ops = []

    # Step 2 — retrieve KB docs
    seen, chunks = set(), []
    for op in ops[:12]:
        for doc in db.similarity_search(op, k=3):
            text = doc.page_content.strip()
            sig  = text[:150]
            if sig not in seen and len(text) >= 40:
                seen.add(sig)
                chunks.append(text)
    kb_context = "\n\n---\n\n".join(chunks[:20])

    # Step 3 — generate
    gen_resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=_RAG_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"## Tensordyne API reference\n\n{kb_context}\n\n"
                f"## PyTorch source to translate\n\n```python\n{src}\n```"
            ),
        }],
    )
    total_in  += gen_resp.usage.input_tokens
    total_out += gen_resp.usage.output_tokens

    code = _extract_code(_response_text(gen_resp))
    if not code:
        return False, total_in, total_out
    output_path.write_text(code)
    return True, total_in, total_out


# ─── Agentic with configurable KB ─────────────────────────────────────────────

def run_agentic(
    level: Level,
    output_path: Path,
    agent,
    kb_path_arg,
) -> tuple[bool, int | None, int | None, int, int]:
    """Agentic with a specific KB. Returns (success, outer_iters, tool_calls, in_tok, out_tok)."""
    config_file = str(TESTS_DIR / level.config_file) if level.config_file else ""
    success = agent.run(
        input_path=str(TESTS_DIR / level.input_file),
        output_path=str(output_path),
        input_shape=level.shapes[0],
        config_file=config_file,
        target_class=level.target_class,
        kb_path=kb_path_arg,
    )
    return (
        success,
        getattr(agent, "last_iteration_count", None),
        getattr(agent, "last_tool_calls",      None),
        getattr(agent, "last_input_tokens",    0),
        getattr(agent, "last_output_tokens",   0),
    )


# ─── Shape Evaluation ─────────────────────────────────────────────────────────

def evaluate_shapes(level: Level, output_path: Path) -> list[dict]:
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
            results.append({
                "shape":    list(shape),
                "passed":   bool(r.passed),
                "max_diff": float(r.max_diff) if r.max_diff is not None else None,
                "error":    (r.error or "")[:300],
            })
        except Exception as e:
            results.append({
                "shape": list(shape), "passed": False,
                "max_diff": None, "error": str(e)[:300],
            })
    return results


# ─── Main Runner ──────────────────────────────────────────────────────────────

def run_ablation(
    kb_names: list[str],
    approaches: list[str],
    level_ids: list[int],
    resume: bool = True,
):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if resume and RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
        print(f"Resuming: {len(existing)} results already saved.\n")

    def already_done(kb, approach, level_id, run_idx):
        return any(
            r["kb_variant"] == kb
            and r["approach"] == approach
            and r["level_id"] == level_id
            and r["run_idx"] == run_idx
            for r in existing
        )

    kb_variants = [(n, p) for n, p in KB_VARIANTS if n in kb_names]
    levels      = [l for l in ALL_LEVELS if l.id in level_ids]
    total_runs  = len(kb_variants) * len(approaches) * len(levels) * N_RUNS
    completed   = 0
    run_times: list[float] = []

    # Build agent once if needed
    agent = None
    if "agentic" in approaches:
        from langgraph_agent.agent import LangGraphAgent
        agent = LangGraphAgent(max_iterations=8, verbose=True)

    for kb_name, kb_path_arg in kb_variants:
        for approach in approaches:
            for level in levels:
                print(f"\n{'='*62}")
                print(f"  KB={kb_name}  |  {approach.upper()}  |  L{level.id}: {level.name}")
                print(f"{'='*62}")

                for run_idx in range(N_RUNS):
                    if resume and already_done(kb_name, approach, level.id, run_idx):
                        print(f"  Run {run_idx+1}/{N_RUNS}: already done, skipping.")
                        completed += 1
                        continue

                    remaining = total_runs - completed
                    eta_str = ""
                    if run_times:
                        avg_s = sum(run_times) / len(run_times)
                        eta_s = int(avg_s * remaining)
                        eta_str = f"  ETA ~{eta_s//60}m{eta_s%60:02d}s"

                    print(
                        f"\n  Run {run_idx+1}/{N_RUNS}  "
                        f"[{completed+1}/{total_runs}{eta_str}]  {_now()}"
                    )

                    out_dir = (
                        OUTPUTS_DIR / kb_name / approach
                        / f"level{level.id}" / f"run{run_idx}"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_path = out_dir / "output.py"

                    record: dict = {
                        "kb_variant":     kb_name,
                        "approach":       approach,
                        "model":          MODEL,
                        "level_id":       level.id,
                        "level_name":     level.name,
                        "run_idx":        run_idx,
                        "timestamp":      _now(),
                        "code_generated": False,
                        "output_file":    str(output_path.relative_to(TESTS_DIR)),
                        "shape_results":  [],
                        "outer_iterations": None,
                        "tool_calls":       None,
                        "input_tokens":   0,
                        "output_tokens":  0,
                        "elapsed_s":      0.0,
                    }

                    t0 = time.monotonic()
                    try:
                        if approach == "rag":
                            ok, in_tok, out_tok = run_rag(level, output_path, kb_path_arg)
                            record["code_generated"] = ok
                            record["input_tokens"]   = in_tok
                            record["output_tokens"]  = out_tok

                        elif approach == "agentic":
                            ok, iters, tool_calls, in_tok, out_tok = run_agentic(
                                level, output_path, agent, kb_path_arg
                            )
                            record["code_generated"]   = ok
                            record["outer_iterations"] = iters
                            record["tool_calls"]       = tool_calls
                            record["input_tokens"]     = in_tok
                            record["output_tokens"]    = out_tok

                        record["elapsed_s"] = round(time.monotonic() - t0, 1)

                        if record["code_generated"] and output_path.exists():
                            print(f"     Evaluating {len(level.shapes)} shapes...")
                            record["shape_results"] = evaluate_shapes(level, output_path)
                            n_passed = sum(1 for s in record["shape_results"] if s["passed"])
                            run_ok   = n_passed == len(level.shapes)
                            extra = ""
                            if approach == "agentic":
                                extra = f"  tool_calls={record['tool_calls']}"
                            print(
                                f"     Shapes: {n_passed}/{len(level.shapes)} passed  "
                                f"{'✅ PASS' if run_ok else '❌ FAIL'}"
                                f"{extra}  "
                                f"[in={record['input_tokens']:,}  {record['elapsed_s']}s]"
                            )
                        else:
                            print(f"     ❌ Code generation failed  [{record['elapsed_s']}s]")

                    except Exception:
                        record["elapsed_s"] = round(time.monotonic() - t0, 1)
                        record["traceback"] = traceback.format_exc()[:2000]
                        print("     ❌ Unexpected error (traceback in results JSON)")

                    run_times.append(record["elapsed_s"])
                    completed += 1
                    existing.append(record)
                    RESULTS_FILE.write_text(json.dumps(existing, indent=2))

    print(f"\nAblation complete. Results → {RESULTS_FILE}")
    _print_kb_summary(existing, kb_variants, approaches, level_ids)


def _print_kb_summary(results, kb_variants, approaches, level_ids):
    print(f"\n{'='*62}")
    print("  KB ABLATION SUMMARY  (pass rate per KB × approach × level)")
    print(f"{'='*62}")

    for approach in approaches:
        print(f"\n  {approach.upper()}")
        header = f"  {'Level':<28}" + "".join(f"{n:>12}" for n, _ in kb_variants)
        print(header)
        print("  " + "-" * (28 + 12 * len(kb_variants)))
        for lid in level_ids:
            level_name = next((l.name for l in ALL_LEVELS if l.id == lid), f"L{lid}")
            row = f"  L{lid} {level_name:<26}"
            for kb_name, _ in kb_variants:
                runs = [
                    r for r in results
                    if r["kb_variant"] == kb_name
                    and r["approach"] == approach
                    and r["level_id"] == lid
                ]
                n_pass = sum(
                    1 for r in runs
                    if r.get("shape_results")
                    and all(s["passed"] for s in r["shape_results"])
                )
                row += f"  {n_pass}/{len(runs)} runs".rjust(12)
            print(row)

    print(f"{'='*62}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KB ablation study: curated vs token-chunked vs AST-chunked"
    )
    parser.add_argument(
        "--kb", nargs="+",
        choices=["curated", "token_600", "ast"],
        default=["curated", "token_600", "ast"],
        help="Which KB variants to test",
    )
    parser.add_argument(
        "--approach", nargs="+",
        choices=["rag", "agentic"],
        default=["rag", "agentic"],
    )
    parser.add_argument(
        "--levels", nargs="+", type=int,
        default=DEFAULT_LEVEL_IDS,
        help=f"Level IDs to include (default: {DEFAULT_LEVEL_IDS})",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore saved results and rerun from scratch",
    )
    args = parser.parse_args()

    run_ablation(
        kb_names=args.kb,
        approaches=args.approach,
        level_ids=args.levels,
        resume=not args.no_resume,
    )
