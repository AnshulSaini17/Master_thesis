#!/usr/bin/env python3
"""
Generate a thesis summary table from a benchmark results JSON.

Works with both:
  - benchmark_results.json     (nn-DSL benchmark, from benchmark_eval.py)
  - benchmark_vpu_results.json (VPU benchmark, from benchmark_vpu_eval.py)

Usage:
  python benchmark_report.py                                # nn benchmark
  python benchmark_report.py --results benchmark_vpu_results.json
  python benchmark_report.py --detail                       # per-level breakdown
  python benchmark_report.py --csv                          # also write CSV
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

TESTS_DIR = Path(__file__).parent
DEFAULT_RESULTS_FILE = TESTS_DIR / "benchmark_results.json"

N_RUNS         = 3
PASS_THRESHOLD = 2   # runs out of N_RUNS that must pass for a level to count
APPROACHES     = ["prompting", "rag", "agentic"]

# ─── Aggregation ─────────────────────────────────────────────────────────────

def run_passed(record: dict) -> bool:
    """A single run passes if all 3 shapes passed."""
    shapes = record.get("shape_results", [])
    return bool(shapes) and all(s["passed"] for s in shapes)


def level_summary(records: list[dict]) -> dict:
    """Summarise N_RUNS records for one (approach, level) pair."""
    runs_passed    = [r for r in records if run_passed(r)]
    level_ok       = len(runs_passed) >= PASS_THRESHOLD

    # max_diff: mean over passed runs (all shapes)
    diffs = [
        s["max_diff"]
        for r in runs_passed
        for s in r.get("shape_results", [])
        if s.get("max_diff") is not None
    ]
    mean_diff = mean(diffs) if diffs else None

    # Tool calls: agentic only. This is the meaningful self-correction metric —
    # the agent's own run_evaluation retries inside one outer turn show up here,
    # not in outer_iterations (which is ~always 1 because the agent self-validates).
    tool_calls: list[int] = []
    for r in runs_passed:
        v = r.get("tool_calls")
        if v is not None:
            tool_calls.append(v)
    mean_tool_calls = round(mean(tool_calls), 1) if tool_calls else None

    # first-pass rate: fraction of runs where code was generated and all shapes passed
    first_pass_rate = len(runs_passed) / N_RUNS if records else 0.0

    # sandbagging: VPU benchmark only — count shape-results flagged as sandbagging
    sandbag_shapes = sum(
        1 for r in records for s in r.get("shape_results", []) if s.get("sandbagging")
    )

    return {
        "level_ok":        level_ok,
        "runs_passed":     len(runs_passed),
        "total_runs":      len(records),
        "first_pass_rate": first_pass_rate,
        "mean_max_diff":   mean_diff,
        "mean_tool_calls": mean_tool_calls,
        "sandbag_shapes":  sandbag_shapes,
    }


def build_table(results: list[dict]) -> dict:
    """
    Returns:
      {approach: {level_id: summary_dict}}
    """
    # Group by (approach, level_id)
    grouped: dict[tuple, list] = {}
    for r in results:
        key = (r["approach"], r["level_id"])
        grouped.setdefault(key, []).append(r)

    table = {}
    for approach in APPROACHES:
        table[approach] = {}
        level_ids = sorted({r["level_id"] for r in results})
        for lid in level_ids:
            recs = grouped.get((approach, lid), [])
            table[approach][lid] = level_summary(recs)

    return table


def difficulty_score(approach_row: dict) -> int:
    """Highest consecutive level passed (0 if none)."""
    passed_levels = [lid for lid, s in sorted(approach_row.items()) if s["level_ok"]]
    if not passed_levels:
        return 0
    score = 0
    for lid in sorted(approach_row.keys()):
        if approach_row[lid]["level_ok"]:
            score = lid
        else:
            break
    return score


# ─── Printing ─────────────────────────────────────────────────────────────────

# LEVEL_NAMES is derived from the records' "level_name" field at runtime so the
# same report code prints both the nn benchmark and the VPU benchmark.
LEVEL_NAMES: dict[int, str] = {}


def populate_level_names(results: list[dict]) -> None:
    """Fill LEVEL_NAMES from the records' embedded level_name."""
    LEVEL_NAMES.clear()
    for r in results:
        lid = r.get("level_id")
        name = r.get("level_name")
        if lid is not None and name and lid not in LEVEL_NAMES:
            LEVEL_NAMES[lid] = f"L{lid} {name}"


def fmt_diff(v) -> str:
    if v is None:
        return "—"
    if v == 0.0:
        return "0.0"
    return f"{v:.2e}"


def fmt_tool_calls(v) -> str:
    return f"{v:.1f}" if v is not None else "—"


def fmt_rate(v) -> str:
    return f"{v:.0%}"


def print_summary_table(table: dict):
    print("\n" + "=" * 72)
    print("  THESIS BENCHMARK SUMMARY")
    print("=" * 72)
    header = f"{'Metric':<30}" + "".join(f"{a.capitalize():>14}" for a in APPROACHES)
    print(header)
    print("-" * 72)

    # Difficulty score
    d_scores = {a: difficulty_score(table[a]) for a in APPROACHES}
    row = f"{'Difficulty Score (D)':<30}"
    for a in APPROACHES:
        row += f"{d_scores[a]:>14}"
    print(row)

    # First-pass rate (averaged across levels that were run)
    row = f"{'First-Pass Rate':<30}"
    for a in APPROACHES:
        rates = [s["first_pass_rate"] for s in table[a].values() if s["total_runs"] > 0]
        avg = mean(rates) if rates else 0.0
        row += f"{fmt_rate(avg):>14}"
    print(row)

    # Self-correction effort (agentic only): mean tool calls per passed run.
    # The outer-iteration counter is uninformative — see level_summary comment.
    row = f"{'Self-Correction (avg tools)':<30}"
    for a in APPROACHES:
        all_tc = [
            s["mean_tool_calls"]
            for s in table[a].values()
            if s.get("mean_tool_calls") is not None
        ]
        avg = round(mean(all_tc), 1) if all_tc else None
        row += f"{fmt_tool_calls(avg):>14}"
    print(row)

    # Numeric accuracy (mean max_diff across all passed runs)
    row = f"{'Numeric Accuracy (mean max_diff)':<30}"
    for a in APPROACHES:
        diffs = [
            s["mean_max_diff"]
            for s in table[a].values()
            if s.get("mean_max_diff") is not None
        ]
        avg = mean(diffs) if diffs else None
        row += f"{fmt_diff(avg):>14}"
    print(row)

    print("=" * 72)


def print_detail_table(table: dict):
    all_level_ids = sorted({lid for a in APPROACHES for lid in table[a]})

    print("\n" + "=" * 72)
    print("  PER-LEVEL BREAKDOWN")
    print("=" * 72)

    col = 16
    head = f"{'Level':<24}" + "".join(f"{a.capitalize():>{col}}" for a in APPROACHES)
    print(head)

    for lid in all_level_ids:
        name = LEVEL_NAMES.get(lid, f"Level {lid}")
        print(f"\n  {name}")

        # Passed?
        row = f"    {'Passed (≥2/3 runs)':<20}"
        for a in APPROACHES:
            s = table[a].get(lid, {})
            ok = s.get("level_ok")
            val = ("✅" if ok else "❌") if ok is not None else "—"
            row += f"{val:>{col}}"
        print(row)

        # Pass rate
        row = f"    {'Pass rate':<20}"
        for a in APPROACHES:
            s = table[a].get(lid, {})
            rp = s.get("runs_passed")
            rt = s.get("total_runs")
            val = f"{rp}/{rt}" if rt else "—"
            row += f"{val:>{col}}"
        print(row)

        # max_diff
        row = f"    {'mean max_diff':<20}"
        for a in APPROACHES:
            s = table[a].get(lid, {})
            row += f"{fmt_diff(s.get('mean_max_diff')):>{col}}"
        print(row)

        # tool calls (agentic): proxy for self-correction effort
        row = f"    {'avg tool calls':<20}"
        for a in APPROACHES:
            s = table[a].get(lid, {})
            row += f"{fmt_tool_calls(s.get('mean_tool_calls')):>{col}}"
        print(row)

        # sandbagging (VPU only — non-zero means agent emitted all-zero outputs)
        sandbag_total = sum(table[a].get(lid, {}).get("sandbag_shapes", 0) for a in APPROACHES)
        if sandbag_total > 0:
            row = f"    {'sandbagging shapes':<20}"
            for a in APPROACHES:
                s = table[a].get(lid, {})
                val = s.get("sandbag_shapes", 0)
                row += f"{(str(val) if val else '—'):>{col}}"
            print(row)

    print("=" * 72)


def write_csv(table: dict, out_path: Path):
    all_level_ids = sorted({lid for a in APPROACHES for lid in table[a]})
    lines = [
        "approach,level_id,level_name,runs_passed,total_runs,first_pass_rate,"
        "mean_max_diff,mean_tool_calls,sandbag_shapes"
    ]
    for a in APPROACHES:
        for lid in all_level_ids:
            s = table[a].get(lid, {})
            lines.append(",".join([
                a,
                str(lid),
                LEVEL_NAMES.get(lid, f"Level {lid}"),
                str(s.get("runs_passed", "")),
                str(s.get("total_runs", "")),
                f"{s.get('first_pass_rate', ''):.4f}" if s.get("total_runs") else "",
                f"{s['mean_max_diff']:.6e}" if s.get("mean_max_diff") is not None else "",
                str(s.get("mean_tool_calls", "")),
                str(s.get("sandbag_shapes", 0)),
            ]))
    out_path.write_text("\n".join(lines))
    print(f"CSV written to {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Path to results JSON (default: benchmark_results.json; "
        "pass benchmark_vpu_results.json for the VPU benchmark).",
    )
    parser.add_argument("--detail", action="store_true", help="Print per-level breakdown")
    parser.add_argument("--csv", action="store_true", help="Write CSV next to results JSON")
    args = parser.parse_args()

    results_file: Path = args.results
    if not results_file.is_absolute():
        results_file = TESTS_DIR / results_file

    if not results_file.exists():
        print(f"No results file found at {results_file}. Run benchmark_eval.py first.")
        sys.exit(1)

    results = json.loads(results_file.read_text())
    populate_level_names(results)
    print(f"Loaded {len(results)} run records from {results_file.name}.")

    table = build_table(results)
    print_summary_table(table)

    if args.detail:
        print_detail_table(table)

    if args.csv:
        csv_path = results_file.with_suffix(".csv")
        write_csv(table, csv_path)
