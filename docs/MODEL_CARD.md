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
    - name: Binary Accuracy
      type: accuracy
      value: 0.9483
    - name: Binary AUROC
      type: auroc
      value: 0.9821
    - name: Expected Calibration Error (ECE)
      type: expected_calibration_error
      value: 0.0549
    - name: Candidate Choice Top-1
      type: accuracy
      value: 0.8966
    - name: Candidate Choice NDCG
      type: ndcg
      value: 0.9618
    - name: Priority Macro F1
      type: f1
      value: 0.8538
    - name: Risk Classification Accuracy
      type: accuracy
      value: 0.9483
    - name: Hard Constraint Violation Rate
      type: constraint_violation_rate
      value: 0.0000
---

# CDM-1 — Cognitive Decision Model (Model Card)

## Model Overview
**CDM-1 (Cognitive Decision Model)** is an independent, trainable AI decision model built using a hybrid neural architecture that combines contextual language understanding with structured tabular features, outcome predictive modelling, calibrated uncertainty estimation, and hard policy constraint enforcement.

Unlike generative conversational chatbots or LLM API wrappers, CDM-1 is purpose-engineered to output structured, risk-bounded operational decisions (YES/NO, alternative choice, priority tiering, continuous scoring, predictive outcomes, and abstention).

## Parameter Accounting
- **Trainable Parameters**: 9,345,732 (Structured Feature MLP, Gated Fusion, Action-Conditioned Predictive Network, Multi-Head Decision Network, Candidate Scorer)
- **Frozen Parameters**: 149,014,272 (`answerdotai/ModernBERT-base` contextual language backbone)
- **Total Parameters**: 158,360,004 (~158.36M parameters, fulfilling the 150–200M design target)

## Architecture Components
1. **Component A (Language Encoder)**: ModernBERT-base with attention-mask-aware pooling, sequence projection, and configurable gradual unfreezing.
2. **Component B (Structured Feature Encoder)**: Continuous feature normalizer, categorical entity embeddings, and missing-value indicator tracking with MLP projection.
3. **Component C (Cognitive Representation Fusion)**: Dynamic Gated Fusion with cross-attention support, forming a shared latent cognitive state.
4. **Component D (Action-Conditioned Predictive Network)**: Multi-target forward outcome estimation conditioned on candidate actions (binary outcome, multi-class category, resolution time regression, risk score, predictive entropy, and Monte Carlo dropout variance).
5. **Component E (Decision Network)**: Independent output heads:
   - Binary Decision (`YES` / `NO`, Accuracy 94.83%, AUROC 0.9821)
   - Candidate Choice & Ranking (Top-1 89.66%, NDCG 0.9618, with hard policy constraint logit masking)
   - Continuous Scoring (0–100 operational score, MAE 12.98)
   - Priority Tiering (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, Accuracy 91.38%)
   - Risk Assessment (`LOW`, `MODERATE`, `HIGH`, `SEVERE`, Accuracy 94.83%)
   - Calibrated Confidence & Selective Abstention (Calibrated $ECE = 0.0549$, Coverage 63.79%, Selective Accuracy 100.0%)

## Verified Benchmark Results (Holdout Test Partition, N=58)
| Benchmark Metric | Logistic Reg. | Random Forest | XGBoost (v3.4.1) | **CDM-1 (Ours)** |
|---|---|---|---|---|
| **Binary Accuracy** | 0.9310 | 1.0000 | 1.0000 | **0.9483** |
| **Binary AUROC** | 0.9905 | 1.0000 | 1.0000 | **0.9821** |
| **Brier Score** | 0.0628 | 0.0002 | 0.0002 | **0.0449** |
| **Candidate Choice Top-1 Accuracy** | N/A | N/A | N/A | **0.8966** |
| **Candidate Choice NDCG** | N/A | N/A | N/A | **0.9618** |
| **Priority Classification (Macro F1)** | N/A | N/A | N/A | **0.8538** |
| **Risk Classification Accuracy** | N/A | N/A | N/A | **0.9483** |
| **Prohibited Constraint Violations** | N/A | N/A | N/A | **0.00%** |
| **Permutation Invariance ($\Delta_{L2}$)** | N/A | N/A | N/A | **0.0000** |
| **Scalability (32 candidates latency)** | N/A | N/A | N/A | **31.65 ms** |

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
