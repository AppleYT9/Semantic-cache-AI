"""
Central configuration for SemanticCache AI.
Manages settings, model tiers, pricing tables, thresholds, and storage paths.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "experiments.db")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "faq_dataset.json")

# Load environment variables from .env if present
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

# Default Embedding Model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Caching Thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.88
MIN_SIMILARITY_THRESHOLD = 0.70
MAX_SIMILARITY_THRESHOLD = 0.98
DEFAULT_CACHE_TTL_SECONDS = 86400  # 24 hours

# Groq API Configuration for Live LLM Generation
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model Pricing per 1,000,000 Tokens (USD)
# Tier 1 (Small / Low Cost): qwen/qwen3.8-27b (Fast & Precise)
# Tier 2 (Large / High Capability): openai/gpt-oss-120b (Deep Reasoning)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "tier_1_low_cost": {
        "name": "qwen/qwen3.8-27b",
        "label": "Tier 1: Fast/Low-Cost",
        "input_cost_per_m": 0.15,
        "output_cost_per_m": 0.60,
        "avg_latency_ms": 130.0,
        "quality_score_base": 0.88,
    },
    "tier_2_high_capability": {
        "name": "openai/gpt-oss-120b",
        "label": "Tier 2: Capable/Advanced",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "avg_latency_ms": 680.0,
        "quality_score_base": 0.97,
    }
}

# Embedding generation cost (e.g. text-embedding-3-small approx $0.02 / 1M tokens)
EMBEDDING_COST_PER_M_TOKENS = 0.02
CACHE_LOOKUP_LATENCY_MS = 12.0  # Vector lookup + in-memory cosine similarity overhead

@dataclass
class SystemSettings:
    """System runtime configuration."""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    use_semantic_cache: bool = True
    use_model_routing: bool = True
    routing_strategy: str = "rule_based"  # 'always_low_cost', 'always_high_capability', 'rule_based', 'ml_based'
    cache_strategy: str = "semantic"       # 'none', 'exact', 'semantic'
    log_db_path: str = DB_PATH
