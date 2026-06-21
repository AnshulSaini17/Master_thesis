#!/usr/bin/env python3
# ruff: noqa: T201, ANN001, BLE001, PLC0415
"""Categorize failure modes across both experiments.

For each failed (run, shape) we look at the JSON error string AND re-import
the saved output.py to recover the actual Python exception. Then we bucket
into thesis-quoteable categories.

Usage:
  cd /workspaces/rock
  python tests/docs/categorize_failures.py
"""

import importlib.util
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))


CATEGORIES = {
    "api_hallucination_import":  "The generated code imports a module that doesn't exist in Tensordyne.",
    "api_hallucination_attr":    "The generated code references a class/attribute/method the API doesn't expose.",
    "api_misuse_signature":      "A real Tensordyne API was called with the wrong number / name of args.",
    "syntax_error":              "Python syntax error in the generated code.",
    "output_format_wrong":       "Output didn't follow the required schema (no VpuKernel / no build_graph / wrong skeleton).",
    "weight_load_mismatch":      "Trace OK, but PyTorch weights couldn't be copied into the generated module (attribute names diverged).",
    "shape_mismatch":            "Trace OK, but the output shape disagreed with the PyTorch reference.",
    "numeric_mismatch":          "Trace + shape OK, but max_diff exceeded tolerance.",
    "sandbagging":               "Agent produced all-zero output against a non-zero reference (VPU only).",
    "infra_reference_crashed":   "PyTorch reference itself failed before comparison (e.g. broadcast bug in ref code).",
    "infra_runner_crashed":      "Tensordyne runner crashed on otherwise-valid graph.",
    "no_code_generated":         "LLM never produced a code block.",
    "other":                     "Uncategorised — needs manual review.",
}


def categorize_error_text(err_text: str, exec_result: str) -> str:
    """Pick a category from the raw error text + the live exec result if any."""
    text = (err_text or "") + "\n" + (exec_result or "")
    t = text.lower()

    if "sandbag" in t:
        return "sandbagging"
    if "reference model failed" in t:
        return "infra_reference_crashed"
    if "runner execution failed" in t or "transpile_to_torch_runnable" in t:
        return "infra_runner_crashed"
    if "no vpukernel" in t or "build_graph" in t or "must define" in t:
        return "output_format_wrong"
    if "syntaxerror" in t:
        return "syntax_error"
    if "modulenotfounderror" in t or "no module named" in t or "importerror" in t or "cannot import" in t:
        return "api_hallucination_import"
    if "has no attribute" in t or "object has no attribute" in t or "module 'tensordyne" in t:
        return "api_hallucination_attr"
    if "missing 1 required positional argument" in t or "missing 1 required keyword-only" in t \
       or "unexpected keyword argument" in t or "too many positional arguments" in t \
       or "missing 2 required" in t or "positional argument" in t:
        return "api_misuse_signature"
    if "shape mismatch" in t:
        return "shape_mismatch"
    if "numeric mismatch" in t or "max_diff=" in t and "passed" not in t:
        return "numeric_mismatch"
    if "weight" in t and ("missing" in t or "shape" in t or "loading" in t or "size mismatch" in t):
        return "weight_load_mismatch"
    if "nameerror" in t:
        # NameError on a tensordyne identifier is a hallucination; on a user-defined name is a code bug.
        if "tensordyne" in t or "nn." in t or "ir." in t or "vpu" in t:
            return "api_hallucination_attr"
    if "typeerror" in t:
        return "api_misuse_signature"
    return "other"


def reimport(path: Path) -> str:
    """Try to import the file; return a short error description, or empty string on success."""
    if not path.exists():
        return ""
    try:
        spec = importlib.util.spec_from_file_location("_x", path)
        if spec is None or spec.loader is None:
            return "could not create spec"
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return ""
    except Exception:
        tb = traceback.format_exc()
        # Take the last few lines that aren't importlib boilerplate
        lines = [ln for ln in tb.splitlines() if ln.strip()
                 and "_bootstrap" not in ln and "importlib._" not in ln]
        return "\n".join(lines[-4:])


def process_file(label: str, results_path: Path, outputs_root: Path):
    print(f"\n{'=' * 64}")
    print(f"  {label}  ({results_path.name})")
    print(f"{'=' * 64}")
    data = json.loads(results_path.read_text())
    by_category: Counter = Counter()
    by_cell: dict[tuple, Counter] = defaultdict(Counter)
    examples: dict[str, str] = {}

    for r in data:
        approach = r["approach"]
        level_id = r["level_id"]
        run_idx = r["run_idx"]

        out_path = TESTS / r["output_file"] if "output_file" in r else None

        for s in r.get("shape_results", []):
            if s.get("passed"):
                continue
            err_text = s.get("error", "")
            shape = tuple(s.get("shape", []))

            exec_result = ""
            # For VPU-eval failures that look like "unexpected:" or "exec_module",
            # re-import the saved code to get the real exception.
            if out_path is not None and out_path.exists():
                if "unexpected" in err_text.lower() or "exec_module" in err_text.lower():
                    exec_result = reimport(out_path)

            cat = categorize_error_text(err_text, exec_result)
            by_category[cat] += 1
            by_cell[(approach, level_id)][cat] += 1
            if cat not in examples:
                examples[cat] = (err_text + " || " + exec_result)[:200]

    total = sum(by_category.values())
    print(f"\nTotal failed (run, shape) pairs: {total}\n")
    print(f"{'Category':<32} {'Count':>6} {'%':>6}")
    print("-" * 46)
    for cat, n in by_category.most_common():
        pct = round(100 * n / total) if total else 0
        print(f"{cat:<32} {n:>6} {pct:>5}%")

    print("\n--- one representative error per category (first 200 chars) ---")
    for cat in by_category:
        ex = examples.get(cat, "").replace("\n", " | ")[:200]
        print(f"  [{cat}] {ex}")

    print("\n--- per-approach breakdown ---")
    approaches = sorted({k[0] for k in by_cell})
    cats_present = [c for c in CATEGORIES if any(by_cell[k].get(c) for k in by_cell)]
    header = f"  {'cat':<32} " + "".join(f"{a:>10}" for a in approaches) + f"  {'TOTAL':>7}"
    print(header)
    for cat in cats_present:
        row = f"  {cat:<32} "
        total_in_cat = 0
        for a in approaches:
            n = sum(by_cell[(a, lid)].get(cat, 0) for lid in {k[1] for k in by_cell})
            row += f"{(n if n else '-'):>10}"
            total_in_cat += n
        row += f"  {total_in_cat:>7}"
        print(row)


if __name__ == "__main__":
    process_file(
        "EXP 1 — PyTorch → Tensordyne.nn",
        TESTS / "benchmark_results.json",
        TESTS / "benchmark_outputs",
    )
    process_file(
        "EXP 2 — Triton → VPU IR",
        TESTS / "benchmark_vpu_results.json",
        TESTS / "benchmark_vpu_outputs",
    )
