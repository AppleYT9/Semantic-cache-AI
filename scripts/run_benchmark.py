"""
Benchmark Generation Script for SemanticCache AI.
Runs:
1. Baseline Comparisons (No Caching vs. Exact-Match vs. Semantic Cache vs. Routing Tiers)
2. Threshold Sensitivity Sweep (0.70 to 0.98)
Populates SQLite database data/experiments.db with empirical benchmark results.
"""

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import EmbeddingService
from src.database import ExperimentDB
from src.experiments import ExperimentRunner

def main():
    print("=" * 65)
    print("  SemanticCache AI - Benchmark & Sensitivity Experiment Runner")
    print("=" * 65)

    print("\n[1/3] Initializing Embedding Service and Database...")
    t0 = time.time()
    emb = EmbeddingService()
    db = ExperimentDB()
    runner = ExperimentRunner(embedding_service=emb, db=db)
    print(f"       Ready in {time.time() - t0:.2f}s")

    print("\n[2/3] Running Strategy Comparison Suite...")
    comp_results = runner.run_strategy_comparison_suite()
    print(f"       Successfully completed {len(comp_results)} baseline comparison experiments:")
    for name, s in comp_results.items():
        print(f"       -> {name:<46} | HitRate: {s['cache_hit_rate']*100:5.1f}% | Precision: {s['cache_precision']*100:5.1f}% | Savings: {s['cost_savings_pct']:5.1f}% | Latency: {s['avg_latency_ms']:5.1f}ms")

    print("\n[3/3] Running Threshold Sensitivity Sweep (0.70 to 0.98)...")
    sweep_results = runner.run_threshold_sensitivity_sweep()
    print(f"       Successfully generated {len(sweep_results)} threshold sweep points:")
    print("       Threshold | Hit Rate | Precision | False Reuse | Savings % | Avg Latency")
    print("       " + "-" * 60)
    for pt in sweep_results:
        print(f"         {pt['threshold']:<7.2f} | {pt['cache_hit_rate']*100:6.1f}%  | {pt['cache_precision']*100:7.1f}%  | {pt['incorrect_reuse_rate']*100:9.1f}%   | {pt['cost_savings_pct']:7.1f}%  | {pt['avg_latency_ms']:6.1f} ms")

    print("\n" + "=" * 65)
    print("  Benchmark Run Complete! All empirical data logged to SQLite.")
    print("=" * 65)

if __name__ == "__main__":
    main()
