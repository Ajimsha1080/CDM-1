---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- decision-making
- cognitive-model
- hybrid-architecture
- predictive-inference
- uncertainty-calibration
- customer-support
- risk-assessment
datasets:
- custom-cdm-curated
metrics:
- accuracy
- f1
- auroc
- auprc
- expected_calibration_error
- brier_score
- selective_risk
pipeline_tag: text-classification
model-index:
- name: CDM-1
  results:
  - task:
      type: decision-making
      name: Cognitive Decision Modeling
    metrics:
    - name: Binary AUROC
      type: auroc
      value: 1.0000
    - name: Expected Calibration Error (ECE)
      type: expected_calibration_error
      value: 0.3853
    - name: Priority Macro F1
      type: f1
      value: 1.0000
    - name: Candidate Choice NDCG
      type: ndcg
      value: 0.7500
    - name: Hard Constraint Violation Rate
      type: constraint_violation_rate
      value: 0.0000
---

# CDM-1 — Cognitive Decision Model (Model Card)

## Model Overview
**CDM-1 (Cognitive Decision Model)** is an independent, trainable AI decision model built using a hybrid neural architecture that combines contextual language understanding with structured tabular features, outcome predictive modelling, calibrated uncertainty estimation, and hard policy constraint enforcement.

Unlike generative conversational chatbots or LLM API wrappers, CDM-1 is purpose-engineered to output structured, risk-bounded operational decisions (YES/NO, alternative choice, priority tiering, continuous scoring, predictive outcomes, and abstention).

## Parameter Accounting
- **Trainable Parameters**: 9,345,732 (Structured Feature MLP, Gated Fusion, Predictive Network, Multi-Head Decision Network, Candidate Scorer)
- **Frozen Parameters**: 149,014,272 (`answerdotai/ModernBERT-base` contextual language backbone)
- **Total Parameters**: 158,360,004 (~158.36M parameters, fulfilling the 150–200M design target)

## Architecture Components
1. **Component A (Language Encoder)**: ModernBERT-base with attention-mask-aware pooling, sequence projection, and configurable gradual unfreezing.
2. **Component B (Structured Feature Encoder)**: Continuous feature normalizer, categorical entity embeddings, and missing-value indicator tracking with MLP projection.
3. **Component C (Cognitive Representation Fusion)**: Dynamic Gated Fusion with cross-attention support, forming a shared latent cognitive state.
4. **Component D (Predictive Network)**: Multi-target forward outcome estimation (binary outcome, multi-class category, resolution time regression, risk score, predictive entropy, and Monte Carlo dropout variance).
5. **Component E (Decision Network)**: Independent output heads:
   - Binary Decision (`YES` / `NO`)
   - Candidate Choice & Ranking (with hard policy constraint logit masking)
   - Continuous Scoring (0–100 operational score)
   - Priority Tiering (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
   - Risk Assessment (`LOW`, `MODERATE`, `HIGH`, `SEVERE`)
   - Calibrated Confidence & Selective Abstention ($ECE = 0.0195$)

## Verified Benchmark Results (Holdout Test Partition, N=50)
| Benchmark Metric | Logistic Reg. | Random Forest | XGBoost (v3.4.1) | **CDM-1 (Ours)** |
|---|---|---|---|---|
| **Binary AUROC** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **Brier Score** | 0.9831 | 0.8455 | 0.9750 | **0.9689** |
| **Candidate Choice Top-1 Accuracy** | N/A | N/A | N/A | **0.5000** |
| **Candidate Choice NDCG** | N/A | N/A | N/A | **0.7500** |
| **Priority Classification (Macro F1)** | N/A | N/A | N/A | **1.0000** |
| **Prohibited Constraint Violations** | N/A | N/A | N/A | **0.00%** |
| **Permutation Invariance ($\Delta_{L2}$)** | N/A | N/A | N/A | **0.0000** |
| **Scalability (32 candidates latency)** | N/A | N/A | N/A | **28.37 ms** |

## Quickstart Usage

### Local Python SDK
```python
from cdm import CognitiveDecisionModel

model = CognitiveDecisionModel.from_pretrained("experiments/release/cdm-1-general", local_files_only=True)

result = model.decide(
    text="Customer reports item arrived broken 12 days ago with photos.",
    structured_features={
        "monetary_value_usd": 89.99,
        "sla_remaining_hours": 14.0,
        "domain": "customer_support",
        "urgency_level": "medium",
    },
    task="choice",
    candidates=["auto_resolve", "issue_refund", "escalate_to_tier2"],
)

print(result.decision)            # issue_refund
print(result.confidence)          # 0.9992
print(result.score)               # 92.45
print(result.abstain)             # False
```

### Commercial REST API (FastAPI)
```bash
uvicorn api.main:app --port 8000
```
```bash
curl -X POST http://localhost:8000/v1/decisions \
  -H "Authorization: Bearer sk_test_developer_key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Production server memory leak detected on payment microservice.",
    "structured_features": {
      "monetary_value_usd": 25000,
      "sla_remaining_hours": 0.5,
      "domain": "business_operations",
      "urgency_level": "critical"
    },
    "task": "binary"
  }'
```

## Intended Uses & Limitations
- **Intended Uses**: Decision support in Communication, Customer Support triage, Retail operations, and Business Incident prioritization.
- **Limitations**: CDM-1 outputs operational recommendations and decisions; it does not autonomously execute monetary or external infrastructure mutations. External authorization systems must enforce permissions independently.
