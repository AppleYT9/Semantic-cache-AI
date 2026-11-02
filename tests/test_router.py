"""
Unit Tests for Cost-Aware Model Router.
Verifies FR-07 and FR-08.
"""

import pytest
from src.router import CostAwareRouter


@pytest.fixture
def router():
    return CostAwareRouter()


def test_rule_based_simple_query(router):
    decision = router.route("What is a database?", strategy="rule_based")
    assert decision.selected_tier == "tier_1_low_cost"
    assert decision.estimated_complexity in ["simple", "moderate"]


def test_rule_based_complex_query(router):
    complex_q = "Design a distributed, fault-tolerant architecture with low-latency replication."
    decision = router.route(complex_q, strategy="rule_based")
    assert decision.selected_tier == "tier_2_high_capability"
    assert decision.estimated_complexity == "complex"


def test_forced_tier_override(router):
    decision = router.route("What is python?", strategy="rule_based", force_tier="tier_2_high_capability")
    assert decision.selected_tier == "tier_2_high_capability"


def test_always_strategies(router):
    dec1 = router.route("complex query", strategy="always_low_cost")
    assert dec1.selected_tier == "tier_1_low_cost"

    dec2 = router.route("simple query", strategy="always_high_capability")
    assert dec2.selected_tier == "tier_2_high_capability"


def test_ml_router_training(router):
    queries = [
        "what is python",
        "define microservice",
        "what is cache",
        "quick definition of api",
        "design scalable distributed system with high availability",
        "architect multi-region failover database with raft consensus",
        "optimize concurrent throughput in distributed message broker"
    ]
    labels = [
        "tier_1_low_cost",
        "tier_1_low_cost",
        "tier_1_low_cost",
        "tier_1_low_cost",
        "tier_2_high_capability",
        "tier_2_high_capability",
        "tier_2_high_capability"
    ]
    router.train_ml_router(queries, labels)
    assert router.is_ml_trained is True

    dec_pred = router.route("design distributed system", strategy="ml_based")
    assert dec_pred.selected_tier in ["tier_1_low_cost", "tier_2_high_capability"]

