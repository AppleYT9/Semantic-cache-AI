"""
Experiment Runner and Threshold Sensitivity Analysis for SemanticCache AI.
Executes reproducible benchmarks:
- Baseline A (No Cache)
- Baseline B (Exact Match Cache)
- Proposed (Semantic Cache)
- Model Routing Comparisons (Always Tier 1, Always Tier 2, Rule-Based, ML-Based)
- Threshold Sensitivity Sweep (0.70 - 0.98)
Complies with Section 11, Section 12, and FR-11.
"""

import json
import os
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional

from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_CACHE_TTL_SECONDS,
    MODEL_PRICING,
    EMBEDDING_COST_PER_M_TOKENS
)
from src.embeddings import EmbeddingService
from src.cache import SemanticCache
from src.router import CostAwareRouter
from src.llm import LLMService
from src.evaluation import EvaluationMetrics
from src.database import ExperimentDB

class ExperimentRunner:
    """Orchestrates caching and routing benchmarks across datasets."""

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        embedding_service: Optional[EmbeddingService] = None,
        db: Optional[ExperimentDB] = None
    ):
        self.dataset_path = dataset_path
        self.embedding_service = embedding_service or EmbeddingService()
        self.db = db or ExperimentDB()
        self.router = CostAwareRouter()
        self.dataset = self._load_dataset()
        self._train_ml_router_if_possible()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load evaluation dataset from JSON."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _train_ml_router_if_possible(self) -> None:
        """Train router ML classifier on dataset queries with known expected tiers."""
        queries = []
        labels = []
        for item in self.dataset:
            expected_tier = item.get("expected_model_tier", "tier_1_low_cost")
            for q in item.get("queries", []):
                queries.append(q["text"])
                labels.append(expected_tier)
        if len(queries) >= 10:
            self.router.train_ml_router(queries, labels)

    def run_benchmark(
        self,
        cache_strategy: str = "semantic",
        routing_strategy: str = "rule_based",
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute an end-to-end evaluation run simulating a realistic cache lifecycle:
        1. Seed queries populate the cache.
        2. Test queries arrive sequentially and evaluate cache hit, precision, routing, cost, and latency.
        """
        exp_id = f"exp_{cache_strategy}_{routing_strategy}_{int(threshold*100)}_{uuid.uuid4().hex[:6]}"
        name = experiment_name or f"{cache_strategy.capitalize()} Cache ({routing_strategy}, θ={threshold:.2f})"

        # Initialize fresh cache and LLM service
        cache = SemanticCache(
            embedding_service=self.embedding_service,
            default_threshold=threshold,
            default_ttl_seconds=DEFAULT_CACHE_TTL_SECONDS
        )
        llm = LLMService()

        # Step 1: Pre-populate / Seed Cache with canonical entries
        seed_queries = []
        eval_queries = []

        for item in self.dataset:
            intent_id = item["intent_id"]
            expected_ans = item["expected_answer"]
            expected_tier = item.get("expected_model_tier", "tier_1_low_cost")

            for q in item.get("queries", []):
                q_dict = {
                    "query_id": q["id"],
                    "query_text": q["text"],
                    "intent_id": intent_id,
                    "expected_answer": expected_ans,
                    "expected_tier": expected_tier,
                    "paraphrase_type": q.get("type", "standard"),
                    "is_seed": q.get("is_seed", False),
                    "adversarial_intent": q.get("adversarial_intent")
                }
                if q.get("is_seed"):
                    seed_queries.append(q_dict)
                else:
                    eval_queries.append(q_dict)

        # Seed the cache
        for s in seed_queries:
            # Baseline A (none) does not seed
            if cache_strategy != "none":
                cache.put(
                    query=s["query_text"],
                    response=s["expected_answer"],
                    response_model=s["expected_tier"],
                    intent_id=s["intent_id"],
                    metadata={"seeded": True}
                )

        # Step 2: Compute Baseline Cost (Cost if every query was routed to Tier 2 without cache)
        baseline_cost_usd = 0.0
        for q in eval_queries:
            prompt_tokens = llm.estimate_tokens(q["query_text"]) + 25
            compl_tokens = llm.estimate_tokens(q["expected_answer"])
            baseline_cost_usd += llm.calculate_cost("tier_2_high_capability", prompt_tokens, compl_tokens)

        # Step 3: Sequential Evaluation
        query_results = []

        for q in eval_queries:
            t_start = time.time()
            query_text = q["query_text"]
            true_intent = q["intent_id"]

            # Embedding overhead cost for query (for semantic cache)
            emb_cost = (llm.estimate_tokens(query_text) / 1_000_000) * EMBEDDING_COST_PER_M_TOKENS

            # Cache lookup
            lookup_res = cache.lookup(query_text, strategy=cache_strategy, threshold=threshold)

            if lookup_res.is_hit and lookup_res.matched_entry:
                # CACHE HIT
                matched_entry = lookup_res.matched_entry
                is_correct_reuse = (
                    matched_entry.intent_id == true_intent
                    if q.get("paraphrase_type") != "adversarial_intent_shift"
                    else False
                )
                cost_usd = emb_cost  # Near zero, only embedding lookup
                latency_ms = lookup_res.lookup_latency_ms + 10.0  # In-memory retrieval overhead
                response_text = matched_entry.response
                model_tier = "cached_response"
                quality_score = 0.98 if is_correct_reuse else 0.15  # Heavy penalty for false positive reuse
                reasoning = lookup_res.reason
            else:
                # CACHE MISS -> Router & Inference
                is_correct_reuse = None
                routing_dec = self.router.route(query_text, strategy=routing_strategy)
                selected_tier = routing_dec.selected_tier

                gen_res = llm.generate(
                    query=query_text,
                    model_tier=selected_tier,
                    reference_answer=q["expected_answer"]
                )
                cost_usd = emb_cost + gen_res.cost_usd
                latency_ms = lookup_res.lookup_latency_ms + gen_res.latency_ms
                response_text = gen_res.answer
                model_tier = selected_tier
                quality_score = MODEL_PRICING[selected_tier]["quality_score_base"]
                reasoning = f"Cache miss ({lookup_res.reason}). Router selected {selected_tier}."

                # Store newly generated response into cache if caching is active
                if cache_strategy != "none":
                    cache.put(
                        query=query_text,
                        response=response_text,
                        response_model=selected_tier,
                        intent_id=true_intent,
                        metadata={"auto_cached": True}
                    )

            query_results.append({
                "query_id": q["query_id"],
                "query_text": query_text,
                "intent_id": true_intent,
                "is_seed": False,
                "paraphrase_type": q["paraphrase_type"],
                "cache_hit": lookup_res.is_hit,
                "matched_query_id": lookup_res.matched_entry.query_id if lookup_res.matched_entry else None,
                "similarity_score": lookup_res.similarity_score,
                "is_correct_reuse": is_correct_reuse,
                "model_tier": model_tier,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "quality_score": quality_score,
                "response_text": response_text,
                "reasoning": reasoning
            })

        # Calculate metrics
        metrics = EvaluationMetrics.evaluate_experiment_results(query_results, baseline_cost_usd)

        experiment_record = {
            "experiment_id": exp_id,
            "name": name,
            "timestamp": time.time(),
            "dataset_version": "v1.0-faq",
            "embedding_model": self.embedding_service.model_name,
            "similarity_threshold": threshold,
            "cache_strategy": cache_strategy,
            "routing_strategy": routing_strategy,
            "total_queries": metrics["total_queries"],
            "cache_hits": metrics["cache_hits"],
            "cache_hit_rate": metrics["cache_hit_rate"],
            "cache_precision": metrics["cache_precision"],
            "incorrect_reuse_rate": metrics["incorrect_reuse_rate"],
            "avg_latency_ms": metrics["latency_overall"]["mean"],
            "p50_latency_ms": metrics["latency_overall"]["p50"],
            "p95_latency_ms": metrics["latency_overall"]["p95"],
            "baseline_cost_usd": metrics["baseline_cost_usd"],
            "actual_cost_usd": metrics["actual_cost_usd"],
            "cost_savings_pct": metrics["cost_savings_pct"],
            "avg_quality_score": metrics["avg_quality_score"],
            "config": {
                "threshold": threshold,
                "cache_strategy": cache_strategy,
                "routing_strategy": routing_strategy,
                "model_tier_distribution": metrics["model_tier_distribution"]
            }
        }

        # Persist to SQLite
        self.db.save_experiment(experiment_record, query_results)

        return {
            "summary": experiment_record,
            "metrics": metrics,
            "query_details": query_results
        }

    def run_strategy_comparison_suite(self) -> Dict[str, Any]:
        """
        Run the 3 primary caching strategies plus routing comparisons:
        1. No Caching (Baseline A)
        2. Exact Match Caching (Baseline B)
        3. Semantic Caching with Rule-Based Routing (Proposed)
        4. Semantic Caching with ML-Based Routing
        5. Semantic Caching with Always Low-Cost Routing
        6. Semantic Caching with Always High-Capability Routing
        """
        results = {}
        suites = [
            ("none", "always_high_capability", 0.88, "Baseline A: No Caching (All Tier 2)"),
            ("exact", "rule_based", 0.88, "Baseline B: Exact-Match Cache"),
            ("semantic", "always_low_cost", 0.88, "Semantic Cache (Always Tier 1)"),
            ("semantic", "always_high_capability", 0.88, "Semantic Cache (Always Tier 2)"),
            ("semantic", "rule_based", 0.88, "Proposed: Semantic Cache + Rule-Based Router"),
            ("semantic", "ml_based", 0.88, "Proposed: Semantic Cache + ML-Based Router"),
        ]

        for c_strat, r_strat, th, name in suites:
            res = self.run_benchmark(
                cache_strategy=c_strat,
                routing_strategy=r_strat,
                threshold=th,
                experiment_name=name
            )
            results[name] = res["summary"]

        return results

    def run_threshold_sensitivity_sweep(
        self,
        thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate semantic cache across multiple thresholds [0.70 to 0.98]
        to measure the Hit Rate vs. Cache Precision trade-off.
        """
        sweep_thresholds = thresholds or [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
        sweep_results = []

        for th in sweep_thresholds:
            res = self.run_benchmark(
                cache_strategy="semantic",
                routing_strategy="rule_based",
                threshold=th,
                experiment_name=f"Sensitivity θ={th:.2f}"
            )
            s = res["summary"]
            sweep_results.append({
                "threshold": th,
                "cache_hit_rate": s["cache_hit_rate"],
                "cache_precision": s["cache_precision"],
                "incorrect_reuse_rate": s["incorrect_reuse_rate"],
                "cost_savings_pct": s["cost_savings_pct"],
                "avg_latency_ms": s["avg_latency_ms"],
                "p95_latency_ms": s["p95_latency_ms"],
                "actual_cost_usd": s["actual_cost_usd"],
                "avg_quality_score": s["avg_quality_score"]
            })

        return sweep_results
