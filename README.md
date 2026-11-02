# ⚡ SemanticCache AI: ML-Based Semantic Caching & Cost-Aware LLM Routing Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Groq Cloud](https://img.shields.io/badge/Groq%20Cloud-Accelerated%20Inference-f55036.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 18 Passed](https://img.shields.io/badge/Tests-18%20Passed-brightgreen.svg)](https://github.com/AppleYT9/Semantic-cache-AI)

**SemanticCache AI** is an enterprise-grade semantic caching and dynamic cost-aware model routing engine for Large Language Models (LLMs). It dramatically reduces operational inference costs and response latency while maintaining strict factual response quality.

By coupling dense vector embeddings (`all-MiniLM-L6-v2`) with cosine similarity thresholding and an adaptive multi-tier LLM router (Tier 1 Fast/Low-Cost vs. Tier 2 Deep Reasoning), the engine achieves:
- **⚡ Sub-15ms Latency** on cache hits (~98% latency reduction).
- **💰 81.3% Cost Reduction** compared to an un-cached high-capability baseline.
- **🎯 100% Cache Precision** with **0% False Cache Reuse** at the optimal empirical threshold ($\theta = 0.88$).
- **🤖 Live Groq API Inference** supporting high-throughput production models (`qwen/qwen3.8-27b` and `openai/gpt-oss-120b`).

---

## 📑 Table of Contents

1. [Why Semantic Caching?](#-why-semantic-caching)
2. [How It Works (System Workflow)](#-how-it-works-system-workflow)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Empirical Benchmark Results](#-empirical-benchmark-results)
6. [Threshold Sensitivity Analysis ($\theta \in [0.70, 0.98]$)](#-threshold-sensitivity-analysis)
7. [Installation & Setup](#-installation--setup)
8. [Running the Application](#-running-the-application)
9. [REST API Documentation](#-rest-api-documentation)
10. [Testing & Verification](#-testing--verification)
11. [License](#-license)

---

## 💡 Why Semantic Caching?

Traditional caching mechanisms (Redis, Memcached, HTTP caches) rely on **exact string matching** (MD5/SHA256 hashes of queries). However, in natural language LLM applications:

| User A Query | User B Query | User C Query |
| :--- | :--- | :--- |
| *"What is a semantic cache and how does it work?"* | *"Can you explain semantic caching in simple terms?"* | *"How does a semantic cache operate?"* |

All three queries possess **identical semantic intent**, but traditional caches miss 100% of the time because the raw byte strings differ. This causes redundant, expensive roundtrips to proprietary LLM APIs, inflating cloud bills and introducing 1000ms–3000ms latency bottlenecks.

**SemanticCache AI** solves this by storing queries as high-dimensional dense vector embeddings. Incoming queries are evaluated for semantic proximity using vector cosine similarity. When the similarity exceeds the verified boundary threshold ($\theta \ge 0.88$), the system serves the cached response instantaneously in **sub-15 milliseconds at \$0.00 cost**.

---

## 🔍 How It Works (System Workflow)

```
                            [ User Query Input ]
                                     │
                                     ▼
                     [ 1. Query Normalization ]
                     (Lowercase, whitespace trim,
                      punctuation normalization)
                                     │
                                     ▼
                     [ 2. Vector Embedding Engine ]
                     (384-dimensional dense vector,
                      L2-normalized hypersphere)
                                     │
                                     ▼
                     [ 3. Cosine Similarity Lookup ]
                   cos(u, v) = (u · v) / (||u|| · ||v||)
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
         (sim ≥ 0.88)                            (sim < 0.88)
                 │                                       │
                 ▼                                       ▼
        [ 🟢 CACHE HIT ]                        [ 🟠 CACHE MISS ]
     • Retrieve from memory                  • Route to Cost-Aware Router
     • Latency: < 15ms                       • Analyze query complexity:
     • Cost: $0.000000                         - Heuristic Rule-Based
     • Update LRU & hit counts                 - ML Logistic Classifier
                 │                                       │
                 │                           ┌───────────┴───────────┐
                 │                           │                       │
                 │                     (Tier 1: Simple)      (Tier 2: Complex)
                 │                     qwen/qwen3.8-27b      openai/gpt-oss-120b
                 │                     $0.15 / 1M tokens     $2.50 / 1M tokens
                 │                           │                       │
                 │                           └───────────┬───────────┘
                 │                                       │
                 │                                       ▼
                 │                          [ Live Groq LLM Inference ]
                 │                                       │
                 │                                       ▼
                 │                          [ Store in Semantic Cache ]
                 │                          (Save query, vector, response,
                 │                           timestamp, and TTL metadata)
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                        [ SQLite Telemetry Logger ]
                        (Log latency, cost, similarity,
                         model tier, and routing reasoning)
                                     │
                                     ▼
                           [ Response to User ]
```

### Mathematical Formulation
The semantic proximity between an incoming query vector $\mathbf{q}$ and a stored vector $\mathbf{k}$ in the cache is calculated using Cosine Similarity:

$$\text{Similarity}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \cdot \mathbf{k}}{\|\mathbf{q}\|_2 \|\mathbf{k}\|_2} = \sum_{i=1}^{d} q_i k_i \quad (\text{for unit vectors})$$

- If $\max_{\mathbf{k} \in \mathcal{C}} \text{Similarity}(\mathbf{q}, \mathbf{k}) \ge \theta$: Cache **HIT**, return cached answer.
- If $\max_{\mathbf{k} \in \mathcal{C}} \text{Similarity}(\mathbf{q}, \mathbf{k}) < \theta$: Cache **MISS**, route query to the optimal LLM tier, generate live response, and index $(\mathbf{q}, \text{Response})$ in cache.

---

## 🏛️ System Architecture

The engine is engineered around modular, loosely coupled components:

1. **Embedding Pipeline (`src/embeddings.py`)**:
   - Generates 384-dimensional dense vectors.
   - Dual-mode architecture: an ultra-fast sub-millisecond semantic projection engine (< 0.05ms) for instant cold startup, alongside native `SentenceTransformer` (`all-MiniLM-L6-v2`) support.
   - Vector caching to avoid re-embedding repeated queries.

2. **Semantic Cache Store (`src/cache.py`)**:
   - Thread-safe in-memory vector store protected with `threading.RLock`.
   - Built-in Time-To-Live (TTL) expiration, dynamic entry invalidation, and LRU eviction.
   - Detailed lookup metadata (similarity score, latency breakdown, match rationale).

3. **Cost-Aware Model Router (`src/router.py`)**:
   - **Rule-Based Heuristic Router**: Inspects grammatical indicators, question patterns, architectural keywords, and token length to classify complexity into *simple*, *moderate*, or *complex*.
   - **ML-Based Classifier**: Uses TF-IDF representation with balanced Logistic Regression to predict whether a query requires Tier 1 or Tier 2.

4. **Multi-Tier LLM Service (`src/llm.py`)**:
   - Direct integration with **Groq Cloud API** using standard library HTTP transports for maximum portability and zero SDK bloat.
   - **Tier 1 (Fast / Low-Cost)**: `qwen/qwen3.8-27b` (\$0.15/M input, \$0.60/M output).
   - **Tier 2 (High Capability / Deep Reasoning)**: `openai/gpt-oss-120b` (\$2.50/M input, \$10.00/M output).
   - Accurate token counting and USD cost calculation per query.

5. **Telemetry & Audit Database (`src/database.py`)**:
   - High-performance SQLite database logging every request, latency, cost, and routing reasoning.

---

## 📁 Project Structure

```
Semantic-cache-AI/
├── dashboard/
│   ├── app.py                     # Streamlit Interactive Dashboard & Visual Evaluation Studio
│   └── components.py              # Ice-Blue UI theme, KPI cards, and Plotly layout styling
├── data/
│   ├── experiments.db             # SQLite database storing query logs & benchmark runs
│   └── faq_dataset.json           # Evaluation benchmark dataset (canonical, paraphrases, shifts)
├── scripts/
│   ├── create_dataset.py          # Synthetic & curated evaluation dataset generator
│   └── run_benchmark.py           # CLI benchmark runner & sensitivity sweep runner
├── src/
│   ├── __init__.py                # Package initialization
│   ├── api.py                     # FastAPI REST API endpoints (/query, /metrics, /cache, etc.)
│   ├── cache.py                   # Thread-safe semantic cache with TTL & vector similarity
│   ├── config.py                  # Global configurations, model tiers, and pricing tables
│   ├── database.py                # SQLite database operations & query logging
│   ├── embeddings.py              # Sentence embeddings pipeline & fast semantic projection
│   ├── evaluation.py              # Evaluation metrics (hit rate, precision, latency percentiles)
│   ├── experiments.py             # Reproducible 6-strategy benchmark runner
│   ├── llm.py                     # Live Groq API client & cost accounting
│   └── router.py                  # Cost-aware complexity router (Rule-Based & ML-Based)
├── tests/
│   ├── test_api.py                # FastAPI endpoints integration tests
│   ├── test_cache.py              # Semantic cache unit tests (matching, TTL, invalidation)
│   ├── test_evaluation.py         # Evaluation metrics and percentile calculations tests
│   └── test_router.py             # Rule-based & ML router classification tests
├── .env.example                   # Example environment variables template
├── .gitignore                     # Git ignore rules (protecting .env and local databases)
├── requirements.txt               # Production Python dependencies
└── README.md                      # Project documentation
```

---

## 📊 Empirical Benchmark Results

Evaluated against a standardized enterprise benchmark dataset with semantic paraphrases, structural restructurings, and adversarial intent-shift queries:

| Architecture / Strategy | Cache Hit Rate | Cache Precision | Incorrect Reuse | Cost Savings (%) | Avg Latency (ms) | p95 Latency (ms) | Actual Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A: No Caching (All Tier 2)** | 0.0% | 100.0% | 0.0% | **0.0%** | 680.0 ms | 700.0 ms | \$0.07604 |
| **Baseline B: Exact-Match Cache** | 20.0% | 100.0% | 0.0% | **76.4%** | 148.0 ms | 680.0 ms | \$0.01795 |
| **Proposed: Semantic Cache + Rule Router** | **38.0%** | **100.0%** | **0.0%** | **81.3%** | **120.8 ms** | **680.0 ms** | **\$0.01425** |
| **Proposed: Semantic Cache + ML Router** | **38.0%** | **100.0%** | **0.0%** | **71.6%** | **172.9 ms** | **680.0 ms** | **\$0.02156** |
| **Semantic Cache (Always Tier 1)** | 38.0% | 100.0% | 0.0% | **96.2%** | 88.0 ms | 130.0 ms | \$0.00289 |
| **Semantic Cache (Always Tier 2)** | 38.0% | 100.0% | 0.0% | **37.3%** | 425.2 ms | 680.0 ms | \$0.04768 |

---

## 🎯 Threshold Sensitivity Analysis ($\theta \in [0.70, 0.98]$)

The similarity threshold $\theta$ governs the trade-off between recall (Hit Rate) and safety (Precision):

- **$\theta < 0.80$**: *Under-constrained zone*. Cache hit rate climbs to 58%–66%, but false cache reuse begins occurring when queries share keywords but have differing semantic intent.
- **$\theta \in [0.85, 0.88]$**: **Pareto Optimal Frontier**. Achieves **38.0% hit rate** and **81.3% cost reduction** with **100.0% precision** and **0.0% false cache reuse**.
- **$\theta > 0.92$**: *Over-constrained zone*. Hit rate degrades toward the exact-match baseline (20%), rejecting valid natural paraphrases.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AppleYT9/Semantic-cache-AI.git
cd Semantic-cache-AI
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and add your [Groq API Key](https://console.groq.com/keys):
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
USE_PYTORCH_EMBED=0
```

---

## 🚀 Running the Application

### Option 1: Interactive Streamlit Dashboard
Launch the visual inference playground and evaluation studio:
```bash
python -m streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser.

Features included in the dashboard:
1. **Live Semantic Inference Playground**: Real-time query testing, cache hit/miss badges, latency and cost meters, response viewing, and cache store invalidation.
2. **Baseline & Strategy Benchmarks**: Visual bar charts and metric comparisons for all 6 caching strategies.
3. **Threshold Sensitivity ($\theta$ Sweep)**: Dual-axis Pareto trade-off curve across $\theta \in [0.70, 0.98]$.
4. **Cost-Aware Router Diagnostics**: Query complexity classification inspector.
5. **Telemetry & Live Logs**: Real-time query history table and export.
6. **Architecture & Model Specs**: Technical specifications and economics breakdown.

### Option 2: FastAPI REST Microservice
Launch the high-throughput production API server:
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API docs available at `http://localhost:8000/docs`.

### Option 3: Run Benchmark CLI Script
Run reproducible baseline comparisons and threshold sweeps directly from terminal:
```bash
python scripts/run_benchmark.py
```

---

## 🔌 REST API Documentation

### `POST /query`
Process an incoming natural language query through the semantic caching and routing pipeline.

**Request:**
```json
{
  "query": "Can you explain semantic caching in simple terms?",
  "similarity_threshold": 0.88,
  "cache_strategy": "semantic",
  "routing_strategy": "rule_based"
}
```

**Response (Cache Hit):**
```json
{
  "query_id": "9b1deb4d3b7d48f8",
  "cache_hit": true,
  "similarity_score": 0.9412,
  "matched_query": "What is a semantic cache and how does it work?",
  "selected_model": "tier_1_low_cost",
  "latency_ms": 11.4,
  "estimated_cost": 0.0,
  "response": "A semantic cache stores LLM query embeddings and responses...",
  "routing_reasoning": "Cache hit: retrieved response from semantic cache."
}
```

### Key Endpoints:
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check, cache status, and active models |
| `POST` | `/query` | Execute query with semantic caching & routing |
| `GET` | `/metrics` | Retrieve aggregate latency, cost, and hit rate telemetry |
| `GET` | `/cache/stats` | Inspect in-memory cache entries and stats |
| `DELETE` | `/cache` | Flush active cache memory |
| `GET` | `/experiments` | Retrieve historical benchmark experiments |
| `POST` | `/experiments/run_suite` | Trigger full 6-strategy benchmark suite |
| `POST` | `/experiments/run_sweep` | Trigger similarity threshold sensitivity sweep |

---

## 🧪 Testing & Verification

The codebase includes an automated test suite verifying cache logic, routing heuristics, evaluation formulas, and API endpoints.

Run all tests:
```bash
python -m pytest tests/ -v
```

Output:
```
tests/test_api.py ......                 [ 33%]
tests/test_cache.py .....                [ 61%]
tests/test_evaluation.py ..              [ 72%]
tests/test_router.py .....               [100%]

======================= 18 passed in 10.05s =======================
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

**Developed with ❤️ for high-efficiency, cost-optimized Generative AI systems.**
