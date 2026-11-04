# CDM-1 — Cognitive Decision Model
### Independent Trainable Hybrid Neural Decision Architecture | Open-Source Weights + Commercial API

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-46%20Passed-brightgreen.svg)]()
[![Model Weights](https://img.shields.io/badge/Model%20Weights-SafeTensors%20633MB-success.svg)]()

---

## 1. Executive Summary

**CDM-1 (Cognitive Decision Model)** is an independent, trainable AI decision model purpose-built for operational reasoning, candidate action choice, priority scoring, risk estimation, and calibrated decision-making. 

CDM-1 is **not** a chatbot, **not** a conventional RAG application, and **not** an LLM wrapper. It is a dedicated 158M-parameter hybrid neural model combining contextual language representations, tabular structured feature embeddings, gated cognitive fusion, action-conditioned predictive outcome heads, and a multi-head decision network with hard policy constraint enforcement and selective abstention.

---

## 2. Parameter Accounting

| Component | Architecture Role | Parameters | Status |
|---|---|---|---|
| **Component A: Language Encoder** | `answerdotai/ModernBERT-base` (22 layers, hidden 768) | 149,014,272 | Contextual backbone (gradual unfreezing supported) |
| **Component B: Structured Encoder** | Tabular embeddings + missingness flags + MLP | 173,312 | Trainable |
| **Component C: Gated Fusion** | Dynamic Gated & Cross-Attention Latent Fusion | 394,496 | Trainable |
| **Component D: Predictive Network** | Multi-outcome regression, candidate-conditioned outcomes, & risk | 69,509 | Trainable |
| **Component E: Decision Network** | Binary, Scoring, Priority, Risk & Abstention heads | 134,663 | Trainable |
| **Candidate Scorer & Projector** | Candidate conditioning, bilinear interaction & hard constraint masking | 8,573,752 | Trainable ($\Delta W_{L2} = 2.929 > 0$) |
| **TOTAL CDM-1 MODEL** | **Hybrid Cognitive Decision Architecture** | **158,360,004** | **Target Met (150–200M)** |

---

## 3. High-Level Neural Architecture

```
         Input Text                             Structured Tabular Features
              │                                      │
     ┌────────▼────────┐                   ┌─────────▼─────────┐
     │ Pretrained      │                   │ Categorical Embs  │
     │ ModernBERT Base │                   │ Missingness Masks │
     │ (149M Params)   │                   │ Continuous Normal.│
     └────────┬────────┘                   └─────────┬─────────┘
              │ Text Projection                      │ MLP Projection
              ▼                                      ▼
     ┌─────────────────────────────────────────────────────────┐
     │         Component C: Cognitive Representation Fusion    │
     │     Gated Dynamic Weighting + LayerNorm [dim = 256]     │
     └────────────────────────────┬────────────────────────────┘
                                  │ Shared Cognitive State [B, D]
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│  Component D: Predictive Net    │       │   Component E: Decision Network │
│  - Candidate-Conditioned Phi    │       │   - Head 1: Binary (YES/NO)     │
│  - Action-Specific Binary Prob  │       │   - Head 2: Candidate Ranking   │
│  - Action-Specific Multiclass   │       │   - Head 3: Continuous Score    │
│  - Numeric Resolution Hours     │       │   - Head 4: Priority Levels     │
│  - Action-Specific Risk Score   │       │   - Head 5: Risk Tiering        │
│  - Monte Carlo Dropout Variance │       │   - Head 6: Calibrated Conf.    │
└────────────────┬────────────────┘       │   - Head 7: Selective Abstention│
                 │                        └────────────────┬────────────────┘
                 ▼                                         ▼
     ┌──────────────────────────────────────────────────────────────┐
     │           Candidate Scorer & Policy Masking Engine           │
     │  - Bilinear Context-Action Interaction                       │
     │  - Hard Policy Constraint Masking (-1e9 logits for forbidden)│
     │  - All-Prohibited Edge Handling (Explicit Abstention State)  │
     │  - Permutation Invariance Verified (Delta L2 = 0.0)          │
     └──────────────────────────────────────────────────────────────┘
```

---

## 4. Quickstart Guide

### Installation
```bash
git clone https://github.com/Ajimsha1080/CDM-1.git
cd CDM-1
pip install -r requirements.txt
```

### Local Inference via Python SDK
```python
from cdm import CognitiveDecisionModel

# Load strictly validated offline SafeTensors checkpoint
model = CognitiveDecisionModel.from_pretrained("experiments/release/cdm-1-general")

# Run decision inference
result = model.decide(
    text="Customer reports item arrived broken 12 days ago with attached damage photos.",
    structured_features={
        "sentiment_score": -0.85,
        "interaction_count": 1.0,
        "account_tenure_months": 18.0,
        "monetary_value_usd": 89.99,
        "sla_remaining_hours": 14.0,
        "domain": "customer_support",
        "urgency_level": "medium",
    },
    task="choice",
    candidates=["auto_resolve", "issue_refund", "escalate_to_tier2"],
    prohibited_actions=["auto_resolve"]  # Hard policy constraint
)

print("Decision:", result.decision)               # issue_refund
print("Candidate Probabilities:", result.candidate_scores)
print("Calibrated Confidence:", result.confidence) # 0.9995
print("Priority Tier:", result.priority)           # MEDIUM
print("Operational Score:", result.score)          # 92.45
print("Abstain:", result.abstain)                 # False
print("Predicted Risk:", result.risk)              # 0.0412
```

### Python SDK (`cdm_sdk`)
```python
from sdk.cdm_sdk import CDM

# Connects to hosted API or executes locally
client = CDM(api_key="sk_test_developer_key", base_url="http://localhost:8000")

response = client.decide(
    text="Unauthorized root login detected on production database cluster.",
    task="binary"
)

print(response.decision)   # YES
print(response.confidence) # 0.9998
print(response.priority)   # CRITICAL
```

### Production FastAPI REST Service
Start the REST API server:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Submit authenticated decision request:
```bash
curl -X POST http://localhost:8000/v1/decisions \
  -H "Authorization: Bearer sk_test_developer_key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Payment gateway latency spike +400% in last 15 minutes.",
    "structured_features": {
      "monetary_value_usd": 50000,
      "sla_remaining_hours": 0.5,
      "domain": "business_operations",
      "urgency_level": "critical"
    },
    "task": "binary"
  }'
```

---

## 5. Benchmark Performance & Empirical Validation

CDM-1 was benchmarked against classical tabular algorithms and fine-tuned transformers on identical holdout partitions (N=58) across seeds `[42, 123, 999]`:

| Evaluation Metric | Logistic Regression | Random Forest | XGBoost (v3.4.1) | **CDM-1 (Neural Architecture)** |
|---|---|---|---|---|
| **Binary Accuracy** | 0.9310 | 1.0000 | 1.0000 | **0.9483** |
| **Binary AUROC** | 0.9905 | 1.0000 | 1.0000 | **0.9821** |
| **Brier Score** | 0.0628 | 0.0002 | 0.0002 | **0.0449** |
| **Candidate Choice Top-1 Accuracy** | N/A | N/A | N/A | **0.8966** |
| **Candidate Choice NDCG** | N/A | N/A | N/A | **0.9618** |
| **Operational Priority (Macro F1)** | N/A | N/A | N/A | **0.8538** |
| **Risk Classification Accuracy** | N/A | N/A | N/A | **0.9483** |
| **Prohibited Selection Rate** | N/A | N/A | N/A | **0.0000%** (Zero Violations) |
| **Permutation Invariance ($\Delta_{L2}$)** | N/A | N/A | N/A | **0.0000** (Strictly Invariant) |
| **Candidate Scalability ($K=32$)** | N/A | N/A | N/A | **31.65 ms** CPU Latency |
| **Out-of-Distribution (OOD) Safety** | N/A | N/A | N/A | **100% Policy Adherence** |

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for full statistical tables, confusion matrices, and research disclosures.

---

## 6. Training Pipeline (10 Stages)

All training scripts are fully reproducible:
```bash
# Stage 1: Pretrained Encoder Probe
python training/train_encoder.py

# Stage 2: Structured Tabular Encoder
python training/train_structured_encoder.py

# Stage 3: Cognitive Fusion Network
python training/train_fusion.py

# Stages 4 & 5: Decision Heads & Candidate Ranking
python training/train_decision.py

# Stage 6: Predictive Network (Real Outcome Labels)
python training/train_predictor.py

# Stage 7: End-to-End Multi-Task Optimization
python training/train_multitask.py

# Stage 8: Probability Calibration & Abstention
python training/calibrate.py

# Stage 9: Comprehensive Benchmark Evaluation Suite
python benchmarks/benchmark_suite.py

# Stage 10: SafeTensors Release Packaging
python scripts/package_release.py --copy-downloads
```

---

## 7. Project Structure

```
cdm-1/
├── cdm/
│   ├── config/              # Model configurations & feature schemas
│   ├── data/                # Data contracts, dataset loaders & preprocessors
│   ├── models/              # PyTorch neural components (A, B, C, D, E, Scorer, Unified Model)
│   ├── memory/              # Episodic, Semantic, Procedural, & Persistent Storage modules
│   ├── learning/            # Feedback collector, discrepancy auditor, adaptation gatekeeper
│   ├── inference/           # Standalone local inference runtime pipeline
│   └── utils/               # Training utilities, seeds, early stopping
├── datasets/                # Standardized JSONL splits (train, val, test, ood_test) & manifest
├── training/                # 10 reproducible PyTorch training & evaluation scripts
├── benchmarks/              # Benchmark suite, multi-seed baselines & empirical results
├── api/                     # FastAPI commercial service with persistent auth, metering, & rate limits
├── sdk/                     # Installable Python SDK (cdm_sdk) with explicit mode validation
├── docs/                    # Model Card, Benchmark report, Dataset Provenance & API docs
├── scripts/                 # Packaging & deployment utilities
├── tests/                   # 46 comprehensive unit, gradient, constraint, and integration tests
├── Dockerfile               # Production container definition
├── docker-compose.yml       # API + Pgvector deployment configuration
├── requirements.txt         # Core dependencies
└── pyproject.toml           # Package build definition
```

---

## 8. License & Safety Disclaimer

- **License**: Apache-2.0 License.
- **Safety**: CDM-1 outputs decision recommendations, risk metrics, and rankings with hard constraint enforcement. It is **not** an autonomous execution agent. External systems must independently authorize financial transactions, security modifications, and operational mutations.
