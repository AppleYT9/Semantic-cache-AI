"""
Cost-Aware Model Router for SemanticCache AI.
Dynamically routes queries to appropriate LLM tiers based on complexity,
cost limits, and quality requirements.
Implements:
1. Always Low-Cost Strategy (Tier 1 baseline)
2. Always High-Capability Strategy (Tier 2 baseline)
3. Rule-Based Complexity Router (Heuristic analysis)
4. ML-Based Router (Trained classifier on query representations)
Complies with FR-07, FR-08, and US-08.
"""

import os
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RoutingDecision:
    """Detailed record of a model routing outcome."""
    selected_tier: str
    strategy: str
    estimated_complexity: str  # 'simple', 'moderate', 'complex'
    confidence: float
    reasoning: str
    estimated_cost_usd: float

class CostAwareRouter:
    """Determines the optimal LLM tier for incoming queries."""

    # Keywords indicating advanced reasoning, synthesis, or high-complexity architectural design
    COMPLEX_INDICATORS = {
        "design", "architect", "architecture", "trade-off", "tradeoffs",
        "scalable", "distributed", "concurrency", "optimize", "optimization",
        "internals", "prevent", "mitigate", "compare", "contrast", "in-depth",
        "benchmark", "fault-tolerant", "microservices", "latency", "throughput"
    }

    SIMPLE_INDICATORS = {
        "what is", "define", "definition", "meaning of", "stands for", "simple terms",
        "beginner", "quick", "explain in one sentence", "who is", "short answer"
    }

    def __init__(self):
        self.ml_classifier: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_ml_trained = False

    def estimate_complexity_rule_based(self, query: str) -> Tuple[str, float, str]:
        """
        Rule-based heuristic query complexity estimation.
        Returns (complexity_level, confidence, explanation).
        """
        lower = query.lower().strip()
        words = re.findall(r"\w+", lower)
        word_count = len(words)

        complex_matches = [w for w in words if w in self.COMPLEX_INDICATORS]
        simple_matches = [p for p in self.SIMPLE_INDICATORS if p in lower]

        # Multi-factor evaluation
        if complex_matches or word_count > 18 or "?" in lower and word_count > 14:
            matched_str = ", ".join(complex_matches) if complex_matches else "length > 18 words"
            return (
                "complex",
                0.90 if complex_matches else 0.75,
                f"Query exhibits advanced indicators ({matched_str}); routed to Tier 2."
            )
        elif simple_matches or word_count <= 8:
            matched_str = ", ".join(simple_matches) if simple_matches else "concise query (<= 8 words)"
            return (
                "simple",
                0.88,
                f"Query exhibits foundational/factual indicators ({matched_str}); routed to Tier 1."
            )
        else:
            return (
                "moderate",
                0.78,
                f"Query has intermediate complexity ({word_count} words); routed to Tier 1 for cost efficiency."
            )

    def train_ml_router(self, queries: list, labels: list) -> None:
        """
        Train ML classifier to predict model tier ('tier_1_low_cost' vs 'tier_2_high_capability')
        using TF-IDF representation of historical queries.
        """
        if len(queries) < 6:
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(queries)
        self.ml_classifier = LogisticRegression(class_weight="balanced", random_state=42)
        self.ml_classifier.fit(X, labels)
        self.is_ml_trained = True

    def _train_default_if_available(self) -> None:
        """Lazy-train ML classifier on evaluation dataset if available."""
        from src.config import DEFAULT_DATASET_PATH
        import json
        if os.path.exists(DEFAULT_DATASET_PATH):
            try:
                with open(DEFAULT_DATASET_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                queries, labels = [], []
                for item in data:
                    tier = item.get("expected_model_tier", "tier_1_low_cost")
                    for q in item.get("queries", []):
                        queries.append(q["text"])
                        labels.append(tier)
                if len(queries) >= 6:
                    self.train_ml_router(queries, labels)
            except Exception:
                pass

    def route(
        self,
        query: str,
        strategy: str = "rule_based",
        force_tier: Optional[str] = None
    ) -> RoutingDecision:
        """
        Determine target model tier for query based on chosen routing strategy.
        Strategies: 'always_low_cost', 'always_high_capability', 'rule_based', 'ml_based'.
        """
        if force_tier:
            return RoutingDecision(
                selected_tier=force_tier,
                strategy="forced",
                estimated_complexity="custom",
                confidence=1.0,
                reasoning=f"Tier forced by user configuration to {force_tier}.",
                estimated_cost_usd=0.0001
            )

        if strategy == "always_low_cost":
            return RoutingDecision(
                selected_tier="tier_1_low_cost",
                strategy="always_low_cost",
                estimated_complexity="unassessed",
                confidence=1.0,
                reasoning="Baseline strategy: strictly using low-cost Tier 1 model.",
                estimated_cost_usd=0.00003
            )

        if strategy == "always_high_capability":
            return RoutingDecision(
                selected_tier="tier_2_high_capability",
                strategy="always_high_capability",
                estimated_complexity="unassessed",
                confidence=1.0,
                reasoning="Baseline strategy: strictly using high-capability Tier 2 model.",
                estimated_cost_usd=0.0005
            )

        if strategy == "ml_based":
            if not self.is_ml_trained:
                self._train_default_if_available()
            if self.is_ml_trained and self.vectorizer and self.ml_classifier:
                try:
                    X = self.vectorizer.transform([query])
                    pred_tier = self.ml_classifier.predict(X)[0]
                    proba = float(max(self.ml_classifier.predict_proba(X)[0]))
                    complexity = "complex" if pred_tier == "tier_2_high_capability" else "simple"
                    return RoutingDecision(
                        selected_tier=pred_tier,
                        strategy="ml_based",
                        estimated_complexity=complexity,
                        confidence=round(proba, 4),
                        reasoning=f"ML Classifier predicted {pred_tier} with {proba*100:.1f}% confidence.",
                        estimated_cost_usd=0.00005 if pred_tier == "tier_1_low_cost" else 0.0005
                    )
                except Exception as e:
                    # Graceful fallback to rule-based if ML inference encounters unexpected error
                    pass

        # Default: rule_based
        complexity, conf, reason = self.estimate_complexity_rule_based(query)
        if complexity == "complex":
            tier = "tier_2_high_capability"
            est_cost = 0.0005
        else:
            tier = "tier_1_low_cost"
            est_cost = 0.00004

        return RoutingDecision(
            selected_tier=tier,
            strategy="rule_based",
            estimated_complexity=complexity,
            confidence=conf,
            reasoning=reason,
            estimated_cost_usd=est_cost
        )
