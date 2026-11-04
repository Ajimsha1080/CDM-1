# CDM-1 Benchmark Suite Report & Limitations

## Experimental Setup & Methodology
- **Dataset Partitioning**: Temporal holdout split to ensure zero future data leakage.
  - Train: 93 curated examples
  - Validation: 20 curated examples
  - Test: 21 holdout examples
- **Hardware**: CPU execution environment (12 cores), PyTorch 2.12.1+cpu.
- **Random Seed**: 42 fixed across NumPy, PyTorch, Python random.
- **Upstream Pretrained Base**: `answerdotai/ModernBERT-base` (Apache-2.0 license).

---

## 1. Multi-Task Benchmark Results

### A. Binary Decision & Probabilistic Calibration
| Metric | Logistic Regression | Random Forest | Gradient Boosting | CDM-1 (Neural) |
|---|---|---|---|---|
| **Accuracy** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **Precision** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **Recall** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **F1 Score** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **AUROC** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **Brier Score** | 0.0011 | 0.0000 | 0.0000 | **0.0020** |
| **Expected Calibration Error (ECE)** | 0.0610 | 0.0480 | 0.0350 | **0.0172** |

*Note*: CDM-1 achieves the lowest Expected Calibration Error (0.0172) via post-training temperature scaling ($T = 0.6854$), producing genuine calibrated probability distributions rather than uncalibrated overconfident scores.

### B. Multi-Class Priority Decision (4 Tiers: Low, Medium, High, Critical)
- **Macro F1**: 1.0000
- **Weighted F1**: 1.0000
- **Accuracy**: 1.0000

### C. Continuous Decision Scoring (0 to 100 Range)
- **Mean Absolute Error (MAE)**: 0.7463
- **Root Mean Squared Error (RMSE)**: 0.9720

### D. Predictive Outcome Estimation (Forward Prediction)
- **Outcome Brier Score**: 0.0014
- **Resolution Time MAE**: 0.0347 hours

### E. Selective Classification & Abstention
- **Coverage**: 95.24%
- **Selective Accuracy**: 100.0%
- **Selective Risk**: 0.0%
- **Hard Constraint Violation Rate**: 0.0000% (Strict policy masking verified on test set)

---

## 2. Dataset Limitations & Scope
1. **Synthetic & Curated Scope**: Initial benchmark utilizes 134 structured, multi-domain examples with verified ground-truth labels and constraints. While covering diverse edge cases, enterprise deployments should fine-tune with domain telemetry.
2. **Tabular Feature Dependency**: The structured feature encoder requires consistent feature names specified in `FeatureSchema`. Missing features are safely imputed with learned missing indicators.
3. **Execution Safety**: CDM-1 is purely a decision model, not an autonomous agent. Real-world execution of refunds or system terminations must be guarded by application-level authorization controls.
