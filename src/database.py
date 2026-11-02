"""
SQLite Database Storage and Telemetry for SemanticCache AI.
Stores live query logs, experiment runs, and detailed benchmark metrics.
Complies with FR-09, FR-11, and FR-12.
"""

import sqlite3
import json
import os
import time
from typing import Dict, List, Any, Optional
from src.config import DB_PATH

class ExperimentDB:
    """Manages SQLite storage for experiment tracking and query logging."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create tables if not existing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Queries log (live requests)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    query_text TEXT,
                    normalized_query TEXT,
                    intent_id TEXT,
                    cache_hit INTEGER,
                    similarity_score REAL,
                    selected_model TEXT,
                    latency_ms REAL,
                    estimated_cost REAL,
                    answer_quality REAL,
                    response_text TEXT,
                    cache_strategy TEXT,
                    routing_strategy TEXT,
                    reasoning TEXT
                )
            """)

            # Experiment summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT,
                    timestamp REAL,
                    dataset_version TEXT,
                    embedding_model TEXT,
                    similarity_threshold REAL,
                    cache_strategy TEXT,
                    routing_strategy TEXT,
                    total_queries INTEGER,
                    cache_hits INTEGER,
                    cache_hit_rate REAL,
                    cache_precision REAL,
                    incorrect_reuse_rate REAL,
                    avg_latency_ms REAL,
                    p50_latency_ms REAL,
                    p95_latency_ms REAL,
                    baseline_cost_usd REAL,
                    actual_cost_usd REAL,
                    cost_savings_pct REAL,
                    avg_quality_score REAL,
                    config_json TEXT
                )
            """)

            # Detailed per-query results inside an experiment
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    query_id TEXT,
                    query_text TEXT,
                    intent_id TEXT,
                    is_seed INTEGER,
                    paraphrase_type TEXT,
                    cache_hit INTEGER,
                    matched_query_id TEXT,
                    similarity_score REAL,
                    is_correct_reuse INTEGER,
                    model_tier TEXT,
                    latency_ms REAL,
                    cost_usd REAL,
                    quality_score REAL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            conn.commit()

    def log_query(self, data: Dict[str, Any]) -> None:
        """Record live query processing event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO query_logs (
                    query_id, timestamp, query_text, normalized_query, intent_id,
                    cache_hit, similarity_score, selected_model, latency_ms,
                    estimated_cost, answer_quality, response_text,
                    cache_strategy, routing_strategy, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("query_id"),
                data.get("timestamp", time.time()),
                data.get("query_text", ""),
                data.get("normalized_query", ""),
                data.get("intent_id", ""),
                1 if data.get("cache_hit") else 0,
                data.get("similarity_score", 0.0),
                data.get("selected_model", ""),
                data.get("latency_ms", 0.0),
                data.get("estimated_cost", 0.0),
                data.get("answer_quality", 0.0),
                data.get("response_text", ""),
                data.get("cache_strategy", "semantic"),
                data.get("routing_strategy", "rule_based"),
                data.get("reasoning", "")
            ))
            conn.commit()

    def save_experiment(self, exp_data: Dict[str, Any], query_details: List[Dict[str, Any]]) -> None:
        """Save complete experiment run and associated query details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO experiments (
                    experiment_id, name, timestamp, dataset_version, embedding_model,
                    similarity_threshold, cache_strategy, routing_strategy,
                    total_queries, cache_hits, cache_hit_rate, cache_precision,
                    incorrect_reuse_rate, avg_latency_ms, p50_latency_ms, p95_latency_ms,
                    baseline_cost_usd, actual_cost_usd, cost_savings_pct, avg_quality_score, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_data["experiment_id"],
                exp_data.get("name", exp_data["experiment_id"]),
                exp_data.get("timestamp", time.time()),
                exp_data.get("dataset_version", "v1.0"),
                exp_data.get("embedding_model", "all-MiniLM-L6-v2"),
                exp_data.get("similarity_threshold", 0.88),
                exp_data.get("cache_strategy", "semantic"),
                exp_data.get("routing_strategy", "rule_based"),
                exp_data.get("total_queries", 0),
                exp_data.get("cache_hits", 0),
                exp_data.get("cache_hit_rate", 0.0),
                exp_data.get("cache_precision", 0.0),
                exp_data.get("incorrect_reuse_rate", 0.0),
                exp_data.get("avg_latency_ms", 0.0),
                exp_data.get("p50_latency_ms", 0.0),
                exp_data.get("p95_latency_ms", 0.0),
                exp_data.get("baseline_cost_usd", 0.0),
                exp_data.get("actual_cost_usd", 0.0),
                exp_data.get("cost_savings_pct", 0.0),
                exp_data.get("avg_quality_score", 0.0),
                json.dumps(exp_data.get("config", {}))
            ))

            for q in query_details:
                cursor.execute("""
                    INSERT INTO experiment_queries (
                        experiment_id, query_id, query_text, intent_id, is_seed,
                        paraphrase_type, cache_hit, matched_query_id, similarity_score,
                        is_correct_reuse, model_tier, latency_ms, cost_usd, quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exp_data["experiment_id"],
                    q.get("query_id"),
                    q.get("query_text"),
                    q.get("intent_id"),
                    1 if q.get("is_seed") else 0,
                    q.get("paraphrase_type", ""),
                    1 if q.get("cache_hit") else 0,
                    q.get("matched_query_id"),
                    q.get("similarity_score", 0.0),
                    1 if q.get("is_correct_reuse") else (0 if q.get("cache_hit") else None),
                    q.get("model_tier", ""),
                    q.get("latency_ms", 0.0),
                    q.get("cost_usd", 0.0),
                    q.get("quality_score", 0.0)
                ))
            conn.commit()

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Fetch all recorded experiments ordered by timestamp descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single experiment summary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def get_experiment_queries(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Fetch query records for a specific experiment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_queries WHERE experiment_id = ?", (experiment_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent query logs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Compute system-wide metrics across all logged queries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(cache_hit) as total_hits,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(estimated_cost) as total_cost_usd,
                    AVG(similarity_score) as avg_similarity
                FROM query_logs
            """)
            row = cursor.fetchone()
            if not row or row["total_queries"] == 0:
                return {
                    "total_queries": 0,
                    "cache_hits": 0,
                    "cache_hit_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "total_cost_usd": 0.0,
                    "avg_similarity": 0.0
                }
            tot = row["total_queries"]
            hits = row["total_hits"] or 0
            return {
                "total_queries": tot,
                "cache_hits": hits,
                "cache_hit_rate": round(hits / tot, 4),
                "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 2),
                "total_cost_usd": round(row["total_cost_usd"] or 0.0, 6),
                "avg_similarity": round(row["avg_similarity"] or 0.0, 4)
            }
