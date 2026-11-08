> [!NOTE]
> **Synthetic Data Notice**: All benchmark figures, experimental evaluation runs, and datasets referenced in this document are synthetic and self-generated for simulation, algorithmic validation, and methodology research only.

# CDM-1 — Scientific Experimental Methodology

**Auditor**: Independent AI Research Auditor & Experimental Validation Engineer  
**Date**: November 6, 2026  
**Standards Adhered**: NeurIPS/ICLR Empirical Reproducibility Checklist & ACM ML Evaluation Standards  

---

## 1. Experimental Objectives

The purpose of this evaluation is to empirically establish:
1. The statistical accuracy, calibration, and ranking performance of CDM-1's neural decision heads across multi-seed holdout partitions.
2. The isolated empirical contribution of cognitive subsystems (Memory, Reasoning, Prediction, Planning) via controlled ablation studies.
3. The comparative performance of CDM-1 relative to competitive tabular algorithms (Logistic Regression, Random Forest, XGBoost) and a fine-tuned transformer text-only baseline (ModernBERT).
4. The robustness of policy constraint adherence, permutation invariance, and selective abstention under out-of-distribution (OOD) and contradictory conditions.

---

## 2. Benchmark Partitions & Dataset Splits

All evaluations are conducted on strictly partitioned JSONL datasets with deterministic SHA-256 integrity verification:

| Partition | File Path | Record Count | SHA-256 Digest | Purpose |
|:---|:---|:---:|:---|:---|
| **Training** | `datasets/train.jsonl` | 210 | `4b74b481599b63f4...` | Multi-task parameter optimization |
| **Validation** | `datasets/val.jsonl` | 40 | `edb19a210bc537d3...` | Checkpoint selection & early stopping |
| **Holdout Test** | `datasets/test.jsonl` | 58 | `68a963fc4f8b4fbc...` | Canonical benchmark evaluation |
| **Held-Out Domain** | `datasets/heldout_domain_test.jsonl` | 8 | `6c2cd78cf0a354e5...` | Cross-domain evaluation (`healthcare_triage`, `compliance_legal`) |
| **Out-of-Distribution** | `datasets/ood_test.jsonl` | 6 | `389f33ed927fb457...` | Zero-shot constraint stress testing |

### Partitioning Rules:
- **Zero Entity Overlap**: No customer ID, session ID, or transaction ID appears in more than one partition.
- **Stratified Distribution**: In-domain classes (`customer_support`, `business_operations`, `finance_and_credit`, `communication`, `retail`) are strictly balanced across training, validation, and test sets.
- **Partition Manifest**: Recorded in [`datasets/split_manifest.json`](../../../datasets/split_manifest.json).

---

## 3. Evaluation Hardware & Software Environment

- **CPU**: AMD Ryzen 5 7600X 6-Core Processor (12 logical CPUs, 4.70 GHz)
- **RAM**: 32 GB DDR5
- **Operating System**: Windows 11 Pro 64-bit
- **Python**: 3.14.0
- **PyTorch**: 2.12.1+cpu (TorchScript & autograd verified)
- **Transformers**: 5.17.0
- **Scikit-Learn**: 1.9.0
- **XGBoost**: 3.4.1
- **Random Seeds Evaluated**: `[42, 123, 999]` for tabular baselines; seed `42` for canonical neural evaluations.

---

## 4. Evaluated Baselines & Input Parity

To ensure fair comparison, all tabular and neural models receive equivalent input information:
1. **Logistic Regression**:
   - L2 regularization ($C=1.0$), L-BFGS solver, `max_iter=500`.
   - Input: 13 normalized numeric features + one-hot categorical features.
2. **Random Forest**:
   - 100 estimators, Gini impurity criterion, `max_depth=5`.
   - Input: Identical feature vector.
3. **XGBoost (`xgboost.XGBClassifier`)**:
   - 100 boosted trees, `max_depth=4`, learning rate $\eta = 0.1$, histogram tree method.
   - Input: Identical feature vector.
4. **ModernBERT Baseline (Text-Only Transformer)**:
   - ModernBERT-base pooled representations (dim=768) fed into a linear decision classification probe.
   - Input: Raw narrative text alone.
5. **CDM-1 (Ours - Hybrid Neural Architecture)**:
   - ModernBERT language backbone + Structured tabular MLP + Dynamic Gated Fusion + Multi-Task Decision & Predictive Heads.
   - Input: Multimodal narrative text + structured continuous/categorical features simultaneously.

---

## 5. Metrics & Statistical Validation

- **Classification**: Accuracy, Macro-averaged F1, AUROC, AUPRC.
- **Continuous Calibration**: Brier Score ($B = \frac{1}{N}\sum (p_i - y_i)^2$) and Expected Calibration Error (ECE across 10 bins).
- **Ranking**: Top-1 Choice Accuracy, Normalized Discounted Cumulative Gain (NDCG).
- **Regression**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
- **Constraint Compliance**: Prohibited Action Selection Rate ($\frac{\text{Prohibited Actions Chosen}}{\text{Total Decisions}}$).
- **Selective Prediction**: Selective Coverage and Selective Risk under calibrated confidence gating ($T=0.7957$, threshold $\tau = 0.70$).
- **Runtime Profiling**: Inference latency per sample (ms) and peak Resident Set Size (RSS) memory consumption via `psutil`.
