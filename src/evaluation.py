"""
Evaluation and Metrics Engine for SemanticCache AI.
Calculates:
- Cache-Hit Rate
- Cache Precision & Incorrect Cache-Reuse Rate
- p50, p95, and Mean Latency distributions
- Baseline vs. Optimized Cost and Cost Savings %
- Response Quality scoring
Complies with Section 12 of the PRD.
"""

import numpy as np
from typing import List, Dict, Any, Optional

class EvaluationMetrics:
    """Calculates standardized data science metrics for caching and routing experiments."""

    @staticmethod
    def calculate_percentiles(values: List[float]) -> Dict[str, float]:
        """Compute mean, p50 (median), and p95 for a list of numbers."""
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2)
        }

    @classmethod
    def evaluate_experiment_results(
        cls,
        queries_evaluated: List[Dict[str, Any]],
        baseline_cost_usd: float
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation metrics across a suite of processed queries.
        
        Each item in queries_evaluated contains:
        - query_id, intent_id, is_seed
        - cache_hit (bool)
        - is_correct_reuse (bool or None)
        - latency_ms (float)
        - cost_usd (float)
        - quality_score (float)
        - model_tier (str)
        """
        total_queries = len(queries_evaluated)
        if total_queries == 0:
            return {
                "total_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
                "cache_precision": 0.0,
                "incorrect_reuse_rate": 0.0,
                "latency_overall": cls.calculate_percentiles([]),
                "latency_hits": cls.calculate_percentiles([]),
                "latency_misses": cls.calculate_percentiles([]),
                "baseline_cost_usd": 0.0,
                "actual_cost_usd": 0.0,
                "cost_savings_usd": 0.0,
                "cost_savings_pct": 0.0,
                "avg_quality_score": 0.0
            }

        hits = [q for q in queries_evaluated if q["cache_hit"]]
        misses = [q for q in queries_evaluated if not q["cache_hit"]]

        total_hits = len(hits)
        total_misses = len(misses)
        cache_hit_rate = total_hits / total_queries

        # Cache precision: proportion of cache hits that were acceptable (intent match)
        # Incorrect cache reuse: proportion of cache hits that were unacceptable
        if total_hits > 0:
            correct_hits = sum(1 for h in hits if h.get("is_correct_reuse") is True)
            incorrect_hits = sum(1 for h in hits if h.get("is_correct_reuse") is False)
            cache_precision = correct_hits / total_hits
            incorrect_reuse_rate = incorrect_hits / total_hits
        else:
            cache_precision = 1.0  # Vacuously true if zero hits
            incorrect_reuse_rate = 0.0

        # Latency statistics
        all_latencies = [q["latency_ms"] for q in queries_evaluated]
        hit_latencies = [q["latency_ms"] for q in hits]
        miss_latencies = [q["latency_ms"] for q in misses]

        latency_overall = cls.calculate_percentiles(all_latencies)
        latency_hits = cls.calculate_percentiles(hit_latencies)
        latency_misses = cls.calculate_percentiles(miss_latencies)

        # Cost calculations
        actual_cost_usd = sum(q.get("cost_usd", 0.0) for q in queries_evaluated)
        cost_savings_usd = max(0.0, baseline_cost_usd - actual_cost_usd)
        cost_savings_pct = (
            (cost_savings_usd / baseline_cost_usd) * 100.0 if baseline_cost_usd > 0 else 0.0
        )

        # Quality scoring
        quality_scores = [q.get("quality_score", 0.0) for q in queries_evaluated]
        avg_quality_score = float(np.mean(quality_scores)) if quality_scores else 0.0

        # Model routing tier distribution
        tier_counts = {}
        for q in queries_evaluated:
            tier = q.get("model_tier", "cached")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "total_queries": total_queries,
            "cache_hits": total_hits,
            "cache_misses": total_misses,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "cache_precision": round(cache_precision, 4),
            "incorrect_reuse_rate": round(incorrect_reuse_rate, 4),
            "latency_overall": latency_overall,
            "latency_hits": latency_hits,
            "latency_misses": latency_misses,
            "baseline_cost_usd": round(baseline_cost_usd, 6),
            "actual_cost_usd": round(actual_cost_usd, 6),
            "cost_savings_usd": round(cost_savings_usd, 6),
            "cost_savings_pct": round(cost_savings_pct, 2),
            "avg_quality_score": round(avg_quality_score, 4),
            "model_tier_distribution": tier_counts
        }
