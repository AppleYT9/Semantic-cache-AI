"""
Unit Tests for Evaluation Metrics Engine.
Verifies FR-11 and Section 12 metrics formulas.
"""

import pytest
from src.evaluation import EvaluationMetrics


def test_percentiles_calculation():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    p = EvaluationMetrics.calculate_percentiles(vals)
    assert p["mean"] == 30.0
    assert p["p50"] == 30.0
    assert p["p95"] >= 45.0
    assert p["min"] == 10.0
    assert p["max"] == 50.0


def test_evaluate_experiment_results():
    queries = [
        # Hit with correct reuse
        {
            "query_id": "q1",
            "intent_id": "intent_1",
            "is_seed": False,
            "cache_hit": True,
            "is_correct_reuse": True,
            "latency_ms": 12.0,
            "cost_usd": 0.000002,
            "quality_score": 0.98,
            "model_tier": "cached"
        },
        # Miss routed to Tier 1
        {
            "query_id": "q2",
            "intent_id": "intent_2",
            "is_seed": False,
            "cache_hit": False,
            "is_correct_reuse": None,
            "latency_ms": 130.0,
            "cost_usd": 0.00005,
            "quality_score": 0.85,
            "model_tier": "tier_1_low_cost"
        }
    ]

    baseline_cost = 0.0010  # Assume Tier 2 baseline cost

    res = EvaluationMetrics.evaluate_experiment_results(queries, baseline_cost)

    assert res["total_queries"] == 2
    assert res["cache_hits"] == 1
    assert res["cache_misses"] == 1
    assert res["cache_hit_rate"] == 0.5
    assert res["cache_precision"] == 1.0
    assert res["incorrect_reuse_rate"] == 0.0
    assert res["cost_savings_pct"] > 0
    assert res["actual_cost_usd"] < baseline_cost

