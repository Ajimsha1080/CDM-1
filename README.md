# CDM-1 — Cognitive Decision Model
### Independent Trainable Hybrid Neural Decision Architecture | Open-Source Weights + Commercial API

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-19%20Passed-brightgreen.svg)]()

---

## 1. Executive Summary

**CDM-1 (Cognitive Decision Model)** is an independent, trainable AI decision model purpose-built for operational reasoning, choice selection, priority scoring, risk estimation, and calibrated decision-making. 

CDM-1 is **not** a chatbot, **not** a conventional RAG application, and **not** an LLM wrapper. It is a dedicated 158M-parameter hybrid neural model combining contextual language representations, tabular structured feature embeddings, gated cognitive fusion, forward-looking predictive heads, and a multi-head decision network with hard policy constraint enforcement and selective abstention.

---

## 2. Parameter Accounting

| Component | Architecture Role | Parameters | Status |
|---|---|---|---|
| **Component A: Language Encoder** | `answerdotai/ModernBERT-base` (22 layers, hidden 768) | 149,014,272 | Frozen backbone (gradual unfreezing supported) |
| **Component B: Structured Encoder** | Tabular embeddings + missingness flags + MLP | 173,312 | Trainable |
| **Component C: Gated Fusion** | Dynamic Gated & Cross-Attention Latent Fusion | 394,496 | Trainable |
| **Component D: Predictive Network** | Multi-outcome regression, classification, & risk heads | 69,509 | Trainable |
| **Component E: Decision Network** | Binary, Scoring, Priority, Risk & Abstention heads | 134,663 | Trainable |
| **Candidate Scorer & Projector** | Candidate conditioning, bilinear interaction & masking | 8,573,752 | Trainable |
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
                                  │ Shared Cognitive State
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│  Component D: Predictive Net    │       │   Component E: Decision Network │
│  - Binary Outcome Probability   │       │   - Head 1: Binary (YES/NO)     │
│  - Multi-Class Outcome Tiers    │       │   - Head 2: Candidate Choice    │
│  - Numeric Resolution Hours     │       │   - Head 3: Continuous Score    │
│  - Continuous Risk Estimation   │       │   - Head 4: Priority Levels     │
│  - Monte Carlo Dropout Variance │       │   - Head 5: Risk Tiering        │
└─────────────────────────────────┘       │   - Head 6: Calibrated Conf.    │
                                          │   - Head 7: Selective Abstention│
                                          └─────────────────────────────────┘
```

---

## 4. Quickstart Guide

### Installation
```bash
git clone https://github.com/organization/cdm-1.git
cd cdm-1
pip install -r requirements.txt
```

### Local Inference via Python SDK
```python
from cdm import CognitiveDecisionModel

# Load exported SafeTensors checkpoint
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
    prohibited_actions=[]
)

print("Decision:", result.decision)               # issue_refund
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

### FastAPI Commercial REST Service
Start the REST API server:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Submit decision request:
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

## 5. Benchmark Performance

CDM-1 was benchmarked on holdout temporal partitions against standard baselines:

| Metric | Logistic Regression | Random Forest | HistGradientBoosting | **CDM-1 (Neural)** |
|---|---|---|---|---|
| **Binary Decision Accuracy** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **Binary F1 Score** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **AUROC** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **Brier Score** | 0.0011 | 0.0000 | 0.0000 | **0.0020** |
| **Expected Calibration Error (ECE)** | 0.0610 | 0.0480 | 0.0350 | **0.0172** |
| **Priority Classification (Macro F1)** | N/A | N/A | N/A | **1.0000** |
| **Continuous Score MAE** | N/A | N/A | N/A | **0.7463** |
| **Predictive Outcome Brier Score** | N/A | N/A | N/A | **0.0014** |
| **Hard Policy Constraint Violation Rate** | N/A | N/A | N/A | **0.0000%** |
| **Selective Abstention Coverage** | 100% | 100% | 100% | **95.24%** |

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

# Stage 9: Comprehensive Benchmark Evaluation
python training/evaluate.py

# Stage 10: SafeTensors Release Packaging
python training/export_model.py
```

---

## 7. Project Structure

```
cdm-1/
├── cdm/
│   ├── config/              # Model configurations & feature schemas
│   ├── data/                # Data contracts, dataset loaders & preprocessors
│   ├── models/              # PyTorch neural components (A, B, C, D, E, Scorer, Unified Model)
│   ├── memory/              # Episodic, Semantic, and Procedural memory modules
│   ├── learning/            # Feedback collector, discrepancy auditor, adaptation gatekeeper
│   ├── inference/           # Standalone local inference runtime pipeline
│   └── utils/               # Training utilities, seeds, early stopping
├── datasets/                # Standardized JSONL splits (train, val, test) & data builder
├── training/                # 10 reproducible PyTorch training & evaluation scripts
├── benchmarks/              # Benchmark suite & baseline models comparison
├── api/                     # FastAPI commercial service with authentication & rate limiting
├── sdk/                     # Installable Python SDK (cdm_sdk)
├── docs/                    # Model card and benchmark limitations report
├── tests/                   # 19 comprehensive unit and integration tests
├── Dockerfile               # Production container definition
├── docker-compose.yml       # API + Pgvector deployment configuration
├── requirements.txt         # Core dependencies
└── pyproject.toml           # Package build definition
```

---

## 8. License & Safety Disclaimer

- **License**: Apache-2.0 License.
- **Safety**: CDM-1 outputs decision recommendations, risk metrics, and rankings. It is **not** an autonomous execution agent. External systems must independently authorize financial transactions, security modifications, and operational mutations.
