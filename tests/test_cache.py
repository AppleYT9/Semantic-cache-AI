"""
Unit Tests for SemanticCache AI Caching Engine.
Verifies FR-03, FR-04, FR-05, and FR-10.
"""

import time
import numpy as np
import pytest

from src.cache import SemanticCache, CacheEntry
from src.embeddings import EmbeddingService


class MockEmbeddingService:
    """Mock embedding service producing deterministic vectors for fast unit testing."""
    def __init__(self):
        self.model_name = "mock-minilm"

    def normalize_text(self, text: str) -> str:
        return EmbeddingService.normalize_text(text)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return EmbeddingService.cosine_similarity(vec1, vec2)

    def embed_text(self, text: str) -> np.ndarray:
        norm = self.normalize_text(text)
        vec = np.zeros(16, dtype=np.float32)
        for i, ch in enumerate(norm[:16]):
            vec[i] = ord(ch) / 255.0
        norm_val = np.linalg.norm(vec)
        if norm_val > 0:
            vec /= norm_val
        return vec


@pytest.fixture
def mock_cache():
    mock_emb = MockEmbeddingService()
    return SemanticCache(embedding_service=mock_emb, default_threshold=0.88, default_ttl_seconds=3600)


def test_put_and_exact_match(mock_cache):
    entry = mock_cache.put(
        query="What is SSL?",
        response="SSL is Secure Sockets Layer.",
        response_model="tier_1_low_cost"
    )
    assert entry is not None
    assert entry.original_query == "What is SSL?"

    # Exact match hit
    res = mock_cache.lookup("what is ssl?", strategy="exact")
    assert res.is_hit is True
    assert res.matched_entry is not None
    assert res.matched_entry.response == "SSL is Secure Sockets Layer."

    # Exact match miss
    res_miss = mock_cache.lookup("what is TLS?", strategy="exact")
    assert res_miss.is_hit is False
    assert res_miss.matched_entry is None


def test_semantic_match_identical(mock_cache):
    mock_cache.put(
        query="Explain Python decorators",
        response="Decorators wrap functions to extend behavior.",
        response_model="tier_1_low_cost"
    )

    res = mock_cache.lookup("Explain Python decorators", strategy="semantic", threshold=0.85)
    assert res.is_hit is True
    assert res.similarity_score >= 0.99
    assert "Decorators wrap functions" in res.matched_entry.response


def test_bypass_cache(mock_cache):
    mock_cache.put(
        query="Test query",
        response="Test answer",
        response_model="tier_1_low_cost"
    )
    res = mock_cache.lookup("Test query", strategy="none")
    assert res.is_hit is False


def test_cache_ttl_expiration(mock_cache):
    mock_cache.put(
        query="Short lived query",
        response="Short lived answer",
        response_model="tier_1_low_cost",
        ttl_seconds=1
    )

    # Immediate lookup succeeds
    res1 = mock_cache.lookup("Short lived query", strategy="exact")
    assert res1.is_hit is True

    # After expiration
    time.sleep(1.1)
    res2 = mock_cache.lookup("Short lived query", strategy="exact")
    assert res2.is_hit is False

    cleaned = mock_cache.cleanup_expired()
    assert cleaned >= 1


def test_cache_invalidation_and_clear(mock_cache):
    e1 = mock_cache.put("Query A", "Ans A", "tier_1_low_cost")
    e2 = mock_cache.put("Query B", "Ans B", "tier_1_low_cost")

    assert len(mock_cache.list_entries()) == 2

    # Invalidate single
    ok = mock_cache.invalidate(e1.query_id)
    assert ok is True
    assert len(mock_cache.list_entries()) == 1

    # Clear all
    cleared = mock_cache.clear()
    assert cleared == 1
    assert len(mock_cache.list_entries()) == 0
