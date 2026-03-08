#!/usr/bin/env python3
"""Test-Time Compute vs Multi-Agent Orchestration — Comparison Harness.

Compares GPT-5.4 with extended reasoning (single model, more thinking) against
a 3-agent GPT-5.4 pipeline (no reasoning) on real SWE-bench bugs.
Same model, different inference strategy — isolates the variable.

Usage:
    python harness.py              # Run all 9 tasks
    python harness.py --tasks 3    # Quick run with 3 tasks
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

import evaluate
from strategies import single_model, multi_agent

# 9 diverse SWE-bench_Lite instances across different repos
DEFAULT_INSTANCE_IDS = [
    "astropy__astropy-12907",
    "django__django-11039",
    "django__django-13230",
    "matplotlib__matplotlib-23314",
    "pylint-dev__pylint-7993",
    "pytest-dev__pytest-5692",
    "scikit-learn__scikit-learn-13496",
    "sympy__sympy-13146",
    "sympy__sympy-20442",
]


def load_tasks(n_tasks: int | None = None) -> list[dict]:
    """Load SWE-bench_Lite instances by ID."""
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    id_set = set(DEFAULT_INSTANCE_IDS[:n_tasks] if n_tasks else DEFAULT_INSTANCE_IDS)
    tasks = [row for row in ds if row["instance_id"] in id_set]
    tasks.sort(key=lambda t: DEFAULT_INSTANCE_IDS.index(t["instance_id"]))
    return tasks


def run_comparison(tasks: list[dict]) -> list[dict]:
    """Run both strategies on each task and collect results."""
    results = []
    for i, task in enumerate(tasks, 1):
        iid = task["instance_id"]
        print(f"\n[{i}/{len(tasks)}] {iid}")
        print(f"  Repo: {task['repo']}  |  Problem: {task['problem_statement'][:80]}...")

        ctx = {
            "problem_statement": task["problem_statement"],
            "repo": task["repo"],
            "hints": task.get("hints_text", ""),
        }

        # Strategy A: Single model (GPT-5.4 extreme reasoning)
        print("  Running GPT-5.4 (high reasoning)...", end=" ", flush=True)
        sm_result = single_model.run(**ctx)
        sm_score = evaluate.score(sm_result["patch"], task["patch"])
        if sm_result["error"]:
            print(f"ERROR: {sm_result['error']}")
        else:
            print(f"{sm_result['latency_s']}s | sim={sm_score['diff_similarity']}")

        # Strategy B: Multi-agent pipeline (GPT-5.4, no reasoning)
        print("  Running 3-agent GPT-5.4 pipeline...", end=" ", flush=True)
        ma_result = multi_agent.run(**ctx)
        ma_score = evaluate.score(ma_result["patch"], task["patch"])
        if ma_result["error"]:
            print(f"ERROR: {ma_result['error']}")
        else:
            print(f"{ma_result['latency_s']}s | sim={ma_score['diff_similarity']}")

        results.append({
            "instance_id": iid,
            "repo": task["repo"],
            "single_model": {**sm_result, "scores": sm_score},
            "multi_agent": {**ma_result, "scores": ma_score},
        })

    return results


def print_summary(results: list[dict]):
    """Print a comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Instance':<40} {'GPT-5.4 (single)':<24} {'GPT-5.4 (3-agent)':<24}")
    print(f"{'':40} {'sim   lat   files':<24} {'sim   lat   files':<24}")
    print("-" * 90)

    sm_wins = ma_wins = ties = 0
    for r in results:
        sm, ma = r["single_model"]["scores"], r["multi_agent"]["scores"]
        sm_lat = r["single_model"]["latency_s"]
        ma_lat = r["multi_agent"]["latency_s"]

        sm_col = f"{sm['diff_similarity']:<6}{sm_lat:<6}{'Y' if sm['files_correct'] else 'N'}"
        ma_col = f"{ma['diff_similarity']:<6}{ma_lat:<6}{'Y' if ma['files_correct'] else 'N'}"
        print(f"{r['instance_id']:<40} {sm_col:<24} {ma_col:<24}")

        if sm["diff_similarity"] > ma["diff_similarity"]:
            sm_wins += 1
        elif ma["diff_similarity"] > sm["diff_similarity"]:
            ma_wins += 1
        else:
            ties += 1

    print("-" * 90)
    sm_avg_lat = sum(r["single_model"]["latency_s"] for r in results) / len(results)
    ma_avg_lat = sum(r["multi_agent"]["latency_s"] for r in results) / len(results)
    sm_avg_sim = sum(r["single_model"]["scores"]["diff_similarity"] for r in results) / len(results)
    ma_avg_sim = sum(r["multi_agent"]["scores"]["diff_similarity"] for r in results) / len(results)

    print(f"{'AVERAGE':<40} {sm_avg_sim:<6.3f}{sm_avg_lat:<6.1f}{'':12} {ma_avg_sim:<6.3f}{ma_avg_lat:<6.1f}")
    print(f"\nSingle-model wins: {sm_wins} | Multi-agent wins: {ma_wins} | Ties: {ties}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Test-Time Compute vs Multi-Agent Orchestration")
    parser.add_argument("--tasks", type=int, default=None, help="Number of tasks to run (default: all 9)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print("Loading SWE-bench_Lite dataset...")
    tasks = load_tasks(args.tasks)
    print(f"Loaded {len(tasks)} tasks")

    results = run_comparison(tasks)
    print_summary(results)

    # Save results
    os.makedirs("results", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"results/run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
