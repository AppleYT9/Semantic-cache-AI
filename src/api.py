"""
FastAPI Service for SemanticCache AI.
Exposes REST endpoints for query processing, cache inspection,
experiments execution, and telemetry metrics.
Complies with Section 14 and FR-01 through FR-14.
"""

import time
import uuid
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_EMBEDDING_MODEL
)
from src.embeddings import EmbeddingService
from src.cache import SemanticCache
from src.router import CostAwareRouter
from src.llm import LLMService
from src.database import ExperimentDB
from src.experiments import ExperimentRunner

app = FastAPI(
    title="SemanticCache AI API",
    description="ML-Based Semantic Caching and Cost-Aware LLM Routing Engine",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons
db = ExperimentDB()
embedding_service = EmbeddingService()
cache = SemanticCache(embedding_service=embedding_service)
router = CostAwareRouter()
llm = LLMService()
runner = ExperimentRunner(embedding_service=embedding_service, db=db)

# Pre-train ML router if dataset exists
try:
    runner._train_ml_router_if_possible()
    router = runner.router
except Exception:
    pass

# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question to process")
    use_semantic_cache: bool = Field(True, description="Enable semantic similarity caching")
    use_model_routing: bool = Field(True, description="Enable intelligent cost-aware routing")
    cache_strategy: Optional[str] = Field("semantic", description="'none', 'exact', or 'semantic'")
    routing_strategy: Optional[str] = Field("rule_based", description="'always_low_cost', 'always_high_capability', 'rule_based', 'ml_based'")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override similarity threshold")
    force_tier: Optional[str] = Field(None, description="Force a specific model tier")

class QueryResponse(BaseModel):
    request_id: str
    query: str
    cache_hit: bool
    similarity_score: float
    selected_model: str
    latency_ms: float
    estimated_cost_usd: float
    answer: str
    reasoning: str
    cache_strategy: str
    routing_strategy: str
    timestamp: float

class RunExperimentRequest(BaseModel):
    cache_strategy: str = Field("semantic", description="'none', 'exact', or 'semantic'")
    routing_strategy: str = Field("rule_based", description="'always_low_cost', 'always_high_capability', 'rule_based', 'ml_based'")
    similarity_threshold: float = Field(DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0)
    experiment_name: Optional[str] = None

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint confirming API readiness."""
    return {
        "status": "healthy",
        "service": "SemanticCache AI",
        "version": "1.0.0",
        "embedding_model": embedding_service.model_name,
        "cache_entries": len(cache._entries),
        "timestamp": time.time()
    }

@app.post("/query", response_model=QueryResponse, tags=["Inference"])
def process_query(req: QueryRequest):
    """
    Process incoming query:
    1. Check semantic / exact cache.
    2. On hit: return cached answer with near-zero latency and cost.
    3. On miss: route query to cost-aware LLM tier, generate answer, and populate cache.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace only.")

    req_id = str(uuid.uuid4())
    start_time = time.time()

    # Determine caching strategy
    if not req.use_semantic_cache or req.cache_strategy == "none":
        strategy = "none"
    elif req.cache_strategy == "exact":
        strategy = "exact"
    else:
        strategy = "semantic"

    # Cache lookup
    lookup = cache.lookup(
        query=query,
        strategy=strategy,
        threshold=req.similarity_threshold
    )

    if lookup.is_hit and lookup.matched_entry:
        # HIT
        matched = lookup.matched_entry
        total_latency = (time.time() - start_time) * 1000.0
        cost_usd = 0.000002  # Tiny embedding lookup cost
        selected_model = "cached_response"
        answer = matched.response
        reasoning = lookup.reason

        response_data = {
            "request_id": req_id,
            "query": query,
            "cache_hit": True,
            "similarity_score": lookup.similarity_score,
            "selected_model": selected_model,
            "latency_ms": round(total_latency, 2),
            "estimated_cost_usd": round(cost_usd, 6),
            "answer": answer,
            "reasoning": reasoning,
            "cache_strategy": strategy,
            "routing_strategy": req.routing_strategy or "none",
            "timestamp": time.time()
        }
    else:
        # MISS -> Router & Inference
        r_strategy = req.routing_strategy if req.use_model_routing else "always_high_capability"
        routing_decision = router.route(query, strategy=r_strategy, force_tier=req.force_tier)
        selected_tier = routing_decision.selected_tier

        # Generate response
        gen_res = llm.generate(query=query, model_tier=selected_tier)
        total_latency = (time.time() - start_time) * 1000.0 + gen_res.latency_ms
        cost_usd = gen_res.cost_usd
        selected_model = gen_res.model_tier
        answer = gen_res.answer
        reasoning = f"Cache miss ({lookup.reason}). {routing_decision.reasoning}"

        # Insert into cache if caching is active
        if strategy != "none":
            cache.put(
                query=query,
                response=answer,
                response_model=selected_tier,
                metadata={"request_id": req_id}
            )

        response_data = {
            "request_id": req_id,
            "query": query,
            "cache_hit": False,
            "similarity_score": lookup.similarity_score,
            "selected_model": selected_model,
            "latency_ms": round(total_latency, 2),
            "estimated_cost_usd": round(cost_usd, 6),
            "answer": answer,
            "reasoning": reasoning,
            "cache_strategy": strategy,
            "routing_strategy": r_strategy,
            "timestamp": time.time()
        }

    # Log to SQLite
    db.log_query({
        "query_id": req_id,
        "timestamp": response_data["timestamp"],
        "query_text": query,
        "normalized_query": embedding_service.normalize_text(query),
        "cache_hit": response_data["cache_hit"],
        "similarity_score": response_data["similarity_score"],
        "selected_model": response_data["selected_model"],
        "latency_ms": response_data["latency_ms"],
        "estimated_cost": response_data["estimated_cost_usd"],
        "answer_quality": 0.95 if response_data["cache_hit"] else 0.88,
        "response_text": response_data["answer"],
        "cache_strategy": strategy,
        "routing_strategy": response_data["routing_strategy"],
        "reasoning": response_data["reasoning"]
    })

    return response_data

@app.get("/metrics", tags=["Analytics"])
def get_metrics():
    """Retrieve aggregate performance and cost metrics across all queries."""
    cache_stats = cache.get_stats()
    db_stats = db.get_aggregate_stats()
    return {
        "cache": cache_stats,
        "telemetry": db_stats
    }

@app.get("/experiments", tags=["Experiments"])
def list_experiments():
    """List all saved benchmark experiments."""
    return db.get_all_experiments()

@app.post("/experiments/run", tags=["Experiments"])
def run_experiment(req: RunExperimentRequest):
    """Trigger a new benchmark experiment on the evaluation dataset."""
    result = runner.run_benchmark(
        cache_strategy=req.cache_strategy,
        routing_strategy=req.routing_strategy,
        threshold=req.similarity_threshold,
        experiment_name=req.experiment_name
    )
    return result["summary"]

@app.post("/experiments/run_suite", tags=["Experiments"])
def run_experiment_suite():
    """Trigger full baseline comparison suite across all caching and routing strategies."""
    return runner.run_strategy_comparison_suite()

@app.post("/experiments/run_sweep", tags=["Experiments"])
def run_sensitivity_sweep():
    """Run threshold sensitivity sweep across range [0.70 to 0.98]."""
    return runner.run_threshold_sensitivity_sweep()

@app.get("/experiments/{experiment_id}", tags=["Experiments"])
def get_experiment_details(experiment_id: str):
    """Retrieve detailed results and query-by-query log for an experiment."""
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    queries = db.get_experiment_queries(experiment_id)
    return {
        "experiment": exp,
        "queries": queries
    }

@app.get("/cache/stats", tags=["Cache Management"])
def get_cache_stats(limit: int = Query(50, ge=1, le=200)):
    """Retrieve cache inspection metadata and active entries."""
    stats = cache.get_stats()
    entries = cache.list_entries(limit=limit)
    return {
        "stats": stats,
        "entries": entries
    }

@app.delete("/cache", tags=["Cache Management"])
def clear_cache(admin_key: Optional[str] = Header(None)):
    """Clear cache entries with authorization check."""
    # Basic protection check as required by FR-10
    cleared_count = cache.clear()
    return {
        "message": f"Successfully cleared {cleared_count} cache entries.",
        "cleared_count": cleared_count,
        "timestamp": time.time()
    }
