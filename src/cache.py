"""
Caching Engine for SemanticCache AI.
Implements:
1. Baseline A: No-Cache (always miss)
2. Baseline B: Exact-Match Cache (normalized query match)
3. Proposed: Semantic Cache (vector cosine similarity with configurable threshold & TTL)
Thread-safe with TTL expiration, inspection, and invalidation.
Complies with FR-03, FR-04, FR-05, and FR-10.
"""

import re
import time
import uuid
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_EMBEDDING_MODEL
)
from src.embeddings import EmbeddingService

@dataclass
class CacheEntry:
    """Represents a cached query and its associated response metadata."""
    query_id: str
    original_query: str
    normalized_query: str
    embedding: np.ndarray
    response: str
    created_at: float
    expires_at: float
    embedding_model: str
    response_model: str
    intent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0
    last_accessed_at: float = field(default_factory=time.time)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if entry has passed its TTL."""
        now = current_time or time.time()
        return now > self.expires_at

    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        """Convert entry to serializable dictionary for inspection."""
        data = {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "response": self.response,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "embedding_model": self.embedding_model,
            "response_model": self.response_model,
            "intent_id": self.intent_id,
            "hit_count": self.hit_count,
            "last_accessed_at": self.last_accessed_at,
            "metadata": self.metadata
        }
        if include_vector:
            data["embedding"] = self.embedding.tolist()
        return data

@dataclass
class CacheLookupResult:
    """Result of searching the cache."""
    is_hit: bool
    matched_entry: Optional[CacheEntry] = None
    similarity_score: float = 0.0
    strategy_used: str = "semantic"
    lookup_latency_ms: float = 0.0
    reason: str = ""

class SemanticCache:
    """Core caching engine supporting NoCache, ExactMatch, and Semantic similarity."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        default_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        default_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.threshold = default_threshold
        self.ttl_seconds = default_ttl_seconds
        self._entries: Dict[str, CacheEntry] = {}  # query_id -> CacheEntry
        self._lock = threading.Lock()

        # Telemetry counters
        self.total_lookups = 0
        self.total_hits = 0
        self.total_misses = 0

    def set_threshold(self, threshold: float) -> None:
        """Update similarity threshold dynamically."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold

    def set_ttl(self, ttl_seconds: int) -> None:
        """Update default time-to-live."""
        self.ttl_seconds = max(1, ttl_seconds)

    def clear(self) -> int:
        """Flush all cache entries."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def invalidate(self, query_id: str) -> bool:
        """Remove a specific entry by its query ID."""
        with self._lock:
            if query_id in self._entries:
                del self._entries[query_id]
                return True
            return False

    @staticmethod
    def extract_entity_tokens(text: str) -> set:
        """
        Extract numerical, ID, and entity tokens (e.g., #1024, order IDs, version numbers, quantities).
        Guards against false cache reuse on near-duplicate queries differing only by entity values.
        """
        tokens = set()
        # Numerical IDs with hashtags: #1234, #abc12
        for match in re.findall(r"#\w+", text):
            tokens.add(match.lower())
        # Standalone numbers, versions, or codes: 1024, 3.1, 404
        for match in re.findall(r"\b\d+(?:\.\d+)?\b", text):
            tokens.add(match)
        return tokens

    @classmethod
    def check_entity_guardrail(cls, query_a: str, query_b: str) -> bool:
        """
        Validate whether two queries share compatible entity/numerical slots.
        Returns True if entities match or if neither query contains entity tokens.
        Returns False if queries have conflicting numbers or entity IDs.
        """
        entities_a = cls.extract_entity_tokens(query_a)
        entities_b = cls.extract_entity_tokens(query_b)

        # If neither has specific numerical/ID entities, allow semantic matching
        if not entities_a and not entities_b:
            return True

        # If both have entities, they must match
        return entities_a == entities_b

    def cleanup_expired(self) -> int:
        """Remove any entries past expiration timestamp."""
        now = time.time()
        with self._lock:
            expired_keys = [k for k, v in self._entries.items() if v.is_expired(now)]
            for k in expired_keys:
                del self._entries[k]
            return len(expired_keys)

    def put(
        self,
        query: str,
        response: str,
        response_model: str,
        embedding: Optional[np.ndarray] = None,
        intent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None
    ) -> CacheEntry:
        """Store a new query and response into the cache."""
        norm_query = self.embedding_service.normalize_text(query)
        if embedding is None:
            embedding = self.embedding_service.embed_text(query)

        now = time.time()
        ttl = ttl_seconds or self.ttl_seconds
        entry = CacheEntry(
            query_id=str(uuid.uuid4()),
            original_query=query,
            normalized_query=norm_query,
            embedding=embedding,
            response=response,
            created_at=now,
            expires_at=now + ttl,
            embedding_model=self.embedding_service.model_name,
            response_model=response_model,
            intent_id=intent_id,
            metadata=metadata or {},
            hit_count=0,
            last_accessed_at=now
        )

        with self._lock:
            self._entries[entry.query_id] = entry

        return entry

    def lookup(
        self,
        query: str,
        strategy: str = "semantic",
        threshold: Optional[float] = None
    ) -> CacheLookupResult:
        """
        Query the cache using the designated strategy:
        - 'none': Baseline A (Always miss)
        - 'exact': Baseline B (Normalized exact string match)
        - 'semantic': Proposed (Embedding cosine similarity >= threshold)
        """
        t0 = time.time()
        active_threshold = threshold if threshold is not None else self.threshold

        with self._lock:
            self.total_lookups += 1

            # Strategy 1: No Cache (Baseline A)
            if strategy == "none":
                self.total_misses += 1
                return CacheLookupResult(
                    is_hit=False,
                    strategy_used="none",
                    lookup_latency_ms=(time.time() - t0) * 1000.0,
                    reason="No-cache strategy selected: direct inference required."
                )

            norm_query = self.embedding_service.normalize_text(query)
            now = time.time()

            # Filter valid (non-expired) entries
            valid_entries = [e for e in self._entries.values() if not e.is_expired(now)]

            # Strategy 2: Exact Match Baseline (Baseline B)
            if strategy == "exact":
                for entry in valid_entries:
                    if entry.normalized_query == norm_query:
                        entry.hit_count += 1
                        entry.last_accessed_at = now
                        self.total_hits += 1
                        return CacheLookupResult(
                            is_hit=True,
                            matched_entry=entry,
                            similarity_score=1.0,
                            strategy_used="exact",
                            lookup_latency_ms=(time.time() - t0) * 1000.0,
                            reason=f"Exact normalized string match found (ID: {entry.query_id[:8]})."
                        )
                self.total_misses += 1
                return CacheLookupResult(
                    is_hit=False,
                    strategy_used="exact",
                    lookup_latency_ms=(time.time() - t0) * 1000.0,
                    reason="No exact normalized string match found."
                )

            # Strategy 3: Semantic Cache (Proposed)
            if not valid_entries:
                self.total_misses += 1
                return CacheLookupResult(
                    is_hit=False,
                    strategy_used="semantic",
                    lookup_latency_ms=(time.time() - t0) * 1000.0,
                    reason="Cache is empty or all entries have expired."
                )

            # Calculate similarity against all valid entries
            query_vector = self.embedding_service.embed_text(query)
            best_entry: Optional[CacheEntry] = None
            best_sim = -1.0

            for entry in valid_entries:
                # Ensure embedding model compatibility
                if entry.embedding_model != self.embedding_service.model_name:
                    continue
                sim = self.embedding_service.cosine_similarity(query_vector, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            elapsed_ms = (time.time() - t0) * 1000.0

            if best_entry and best_sim >= active_threshold:
                # Entity & Numerical Guardrail Check
                if not self.check_entity_guardrail(query, best_entry.original_query):
                    self.total_misses += 1
                    return CacheLookupResult(
                        is_hit=False,
                        matched_entry=best_entry,
                        similarity_score=round(best_sim, 4),
                        strategy_used="semantic",
                        lookup_latency_ms=elapsed_ms,
                        reason=f"Similarity {best_sim:.4f} >= threshold {active_threshold:.2f}, but rejected by Entity/Numerical Guardrail (mismatched entity slots/identifiers)."
                    )

                best_entry.hit_count += 1
                best_entry.last_accessed_at = now
                self.total_hits += 1
                return CacheLookupResult(
                    is_hit=True,
                    matched_entry=best_entry,
                    similarity_score=round(best_sim, 4),
                    strategy_used="semantic",
                    lookup_latency_ms=elapsed_ms,
                    reason=f"Semantic match found (similarity {best_sim:.4f} >= threshold {active_threshold:.2f})."
                )
            else:
                self.total_misses += 1
                reason = (
                    f"Highest similarity {best_sim:.4f} below threshold {active_threshold:.2f}."
                    if best_entry else "No compatible cached entries."
                )
                return CacheLookupResult(
                    is_hit=False,
                    matched_entry=best_entry,
                    similarity_score=round(best_sim, 4) if best_sim > 0 else 0.0,
                    strategy_used="semantic",
                    lookup_latency_ms=elapsed_ms,
                    reason=reason
                )

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate cache statistics."""
        with self._lock:
            now = time.time()
            total_entries = len(self._entries)
            active_entries = sum(1 for e in self._entries.values() if not e.is_expired(now))
            expired_entries = total_entries - active_entries
            hit_rate = (self.total_hits / self.total_lookups) if self.total_lookups > 0 else 0.0

            return {
                "total_entries": total_entries,
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl_seconds,
                "embedding_model": self.embedding_service.model_name,
                "total_lookups": self.total_lookups,
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "hit_rate": round(hit_rate, 4)
            }

    def list_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List current cache entries for inspection."""
        with self._lock:
            # Sort by last accessed descending
            sorted_entries = sorted(
                self._entries.values(),
                key=lambda e: e.last_accessed_at,
                reverse=True
            )
            return [e.to_dict() for e in sorted_entries[:limit]]
