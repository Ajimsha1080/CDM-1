> [!NOTE]
> **Synthetic Data Notice**: All benchmark figures, experimental evaluation runs, and datasets referenced in this document are synthetic and self-generated for simulation, algorithmic validation, and methodology research only.

# CDM-1 — Empirical Benchmark Results & Baseline Comparisons

**Auditor**: Independent AI Research Auditor & Experimental Validation Engineer  
**Date**: November 6, 2026  
**Evaluation Partition**: Canonical Holdout Test Set ($N=58$), Multi-Seed Evaluations (`[42, 123, 999]`)  
**Data Reference**: [`benchmarks/benchmark_results.json`](../../../benchmarks/benchmark_results.json)  

---

## 1. Multi-Seed Comparative Benchmark (Holdout Partition, N=58)

| Model Architecture | Binary Accuracy (Mean ± Std) | 95% Confidence Interval | Binary Macro F1 | AUROC | Brier Score | ECE | Prediction Consistency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.9310 ± 0.0000 | [0.9310, 0.9310] | 0.9310 | 0.9905 | 0.0628 | 0.0812 | 100.0% |
| **Random Forest** | 1.0000 ± 0.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.0002 | 0.0010 | 100.0% |
| **XGBoost (v3.4.1)** | 1.0000 ± 0.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.0002 | 0.0010 | 100.0% |
| **ModernBERT Baseline (Text-Only)** | 1.0000 ± 0.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.0015 | 0.0150 | 100.0% |
| **CDM-1 (Hybrid Multi-Task)** | **0.9483** | **[0.8912, 1.0000]** | **0.9483** | **0.9857** | **0.0400** | **0.0710** | **100.0%** |

### Scientific Analysis:
1. Classical tabular tree algorithms (Random Forest, XGBoost) achieve 100% classification on numerical indicators alone when strong separable signals exist in tabular numbers, but they cannot process unstructured narratives, cannot score variable candidate sets, and cannot model counterfactual action outcomes.
2. The fine-tuned ModernBERT text probe achieves high text classification accuracy, but lacks structured feature awareness and cannot enforce hard zero-tolerance policy constraints.
3. CDM-1 simultaneously optimizes across 6 distinct multi-task objectives: binary decision, candidate choice ranking, priority classification, ordinal risk categorization, continuous score regression, and counterfactual outcome risk estimation.

---

## 2. Dedicated Neural Decision Head Performance (CDM-1)

### Candidate Action Choice & Ranking
- **Top-1 Choice Accuracy**: **94.83%** (55 / 58 test items correctly ranked optimal action top-1)
- **NDCG (Normalized Discounted Cumulative Gain)**: **0.9809**
- **Prohibited Action Selection Rate**: **0.0000%** (58 / 58 evaluations strictly obeyed policy constraints)
- **Candidate Scorer Parameter Learning Norm**: $\Delta W_{L2} = 2.7208$

### Operational Priority Classification
- **Accuracy**: **96.55%** (56 / 58)
- **Macro F1**: **0.9625**
- **Confusion Matrix**:
  ```
  Class 0 (Low):      [18,  0, 0,  0]
  Class 1 (Medium):   [ 0, 19, 2,  0]
  Class 2 (High):     [ 0,  0, 9,  0]
  Class 3 (Critical): [ 0,  0, 0, 10]
  ```
- **Class-2 (High) Performance**: Precision: **0.8182**, Recall: **1.0000** (9/9 captured), F1: **0.9000** (resolved prior under-recall through minority loss weighting).

### Ordinal Risk Level Assessment
- **Accuracy**: **94.83%** (55 / 58)
- **Macro F1**: **0.9273**
- **Confusion Matrix**:
  ```
  R0 (Low):      [21,  0, 0,  0]
  R1 (Moderate): [ 0, 15, 0,  0]
  R2 (High):     [ 0,  3, 6,  0]
  R3 (Severe):   [ 0,  0, 0, 13]
  ```

### Predictive Outcome & Continuous Scoring
- **Predictive Outcome Brier Score**: **0.0244**
- **Predictive Outcome MAE**: **0.0295** (normalized)
- **Continuous Score MAE**: **10.7904** (RMSE: 17.3830)

### Reliability, Coverage & Selective Abstention
- **Calibrated Temperature**: $T = 0.7957$
- **Expected Calibration Error (ECE)**: **0.0710**
- **Selective Coverage**: **72.41%**
- **Selective Accuracy**: **100.0%** (zero errors when decision core acts)
- **Selective Risk**: **0.0000**
- **All-Candidates-Prohibited Abstention Rate**: **100% adherence** (returns explicit sentinel `-1`, decision `ABSTAIN`)

---

## 3. Domain-Level Breakdown

| Domain | Binary Accuracy | Priority Accuracy | Sample Count |
|:---|:---:|:---:|:---:|
| **Customer Support** | 100.0% | 100.0% | 12 |
| **Communication** | 100.0% | 100.0% | 11 |
| **Retail Operations** | 100.0% | 100.0% | 12 |
| **Business Operations** | 100.0% | 100.0% | 11 |
| **Finance and Credit** | 75.00% | 83.33% | 12 |

---

## 4. Cross-Domain Generalization (Held-Out Domains, N=8)

Evaluated on unseen domains (`healthcare_triage`, `compliance_legal`) across 8 independent clinical and regulatory cases:
- **Binary Decision Accuracy**: **50.00%** (expected under zero-shot transfer without in-domain training examples)
- **Prohibited Candidate Selection Rate**: **0.0000%** (100% strict policy adherence maintained)
- **Zero-Shot Policy Compliance**: **Verified True**

---

## 5. Candidate Scalability & Permutation Invariance

- **Permutation Invariance**: Candidate order permuted arbitrarily via `[3, 1, 2, 4, 0]`. Vector L2 delta: **$0.0000$** (strictly invariant).
- **Scalability across Candidate Counts ($K$)**:
  - $K=2$: 19.48 ms
  - $K=5$: 18.41 ms
  - $K=10$: 21.33 ms
  - $K=20$: 22.31 ms
  - $K=32$: 26.15 ms

---

## 6. Runtime Latency & Memory Footprint

- **Average Inference Latency (Full Neural Pass)**: **1,160.25 ms / sample** (CPU execution)
- **Candidate-Only Scorer Latency**: **18.41 ms** to **26.15 ms**
- **Resident Set Size (RSS) Memory**: **639.19 MB**
- **Device Profile**: 12-core CPU, Windows 11, PyTorch 2.12.1+cpu
