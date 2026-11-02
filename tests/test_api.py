"""
Integration Tests for SemanticCache AI FastAPI endpoints.
Verifies FR-01, FR-02, FR-09, FR-10, FR-12, FR-13, and Section 14 REST contracts.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "embedding_model" in data


def test_query_flow():
    # 1. First query should be a cache miss (or hit if pre-seeded)
    req_body = {
        "query": "How do distributed hash tables implement fault tolerance?",
        "use_semantic_cache": True,
        "use_model_routing": True,
        "cache_strategy": "semantic",
        "routing_strategy": "rule_based"
    }
    res1 = client.post("/query", json=req_body)
    assert res1.status_code == 200
    data1 = res1.json()
    assert "request_id" in data1
    assert "answer" in data1
    assert "latency_ms" in data1
    assert "estimated_cost_usd" in data1

    # 2. Exact same query should now be a cache hit
    res2 = client.post("/query", json=req_body)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["selected_model"] == "cached_response"


def test_empty_query_rejected():
    res = client.post("/query", json={"query": "   "})
    assert res.status_code == 400


def test_cache_stats_and_clear():
    res = client.get("/cache/stats")
    assert res.status_code == 200
    data = res.json()
    assert "stats" in data
    assert "entries" in data

    res_del = client.delete("/cache")
    assert res_del.status_code == 200
    assert "cleared_count" in res_del.json()


def test_metrics_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "cache" in data
    assert "telemetry" in data


def test_experiments_list():
    res = client.get("/experiments")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

