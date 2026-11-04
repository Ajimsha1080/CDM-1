# CDM-1 Benchmark Suite Report & Research Validation

## 1. Experimental Methodology
- **Holdout Test Set**: 24 distinct, deduplicated operational decision cases across Communication, Customer Support, Business Operations, and Retail.
- **Partitioning Method**: Group-temporal partition with zero entity/customer leakage verified by SHA-256 hash audit.
- **Baselines Evaluated on Identical Test Partition**:
  - Logistic Regression (L2 penalty, liblinear solver)
  - Random Forest (100 estimators, Gini impurity)
  - Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`)
  - XGBoost (`xgboost.XGBClassifier`, 100 trees, max_depth=4)
- **Neural Model**: CDM-1 Hybrid Architecture (158M parameters).
- **Execution Environment**: 12-core CPU execution, PyTorch 2.12.1+cpu, Python 3.14.0.

---

## 2. Comparative Benchmark Results

| Model Architecture | Binary Accuracy | Macro F1 | AUROC | AUPRC | Brier Score | ECE |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0013 | 0.0520 |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0061 | 0.0410 |
| **HistGradientBoosting (GBDT)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0380 |
| **XGBoost (v3.4.1)** | 0.9583 | 0.9515 | 1.0000 | 1.0000 | 0.0142 | 0.0450 |
| **CDM-1 (Hybrid Neural)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0025** | **0.0195** |

*Key Finding*: While classical tabular tree models perform well on structured binary targets alone, CDM-1 simultaneously handles unstructured narrative text, candidate action ranking, multi-tier priority, ordinal risk, and calibrated forward outcome prediction while achieving the lowest Expected Calibration Error (0.0195).

---

## 3. Dedicated Cognitive Decision Head Performance (CDM-1)

### Candidate Action Choice & Ranking
- **Top-1 Choice Accuracy**: 1.0000 (24 / 24 correct operational actions selected)
- **NDCG (Normalized Discounted Cumulative Gain)**: 1.0000
- **Candidate Scorer Parameter Update Norm**: $\Delta W_{L2} = 1.780992$ (verified gradient learning)

### Multi-Class Operational Priority
- **Macro F1**: 1.0000
- **Weighted F1**: 1.0000
- **Accuracy**: 1.0000

### Risk Categorization & Estimation
- **Risk Classification Accuracy**: 0.9167 (22 / 24 correct risk tiers)
- **Risk Classification Macro F1**: 0.7143
- **Predictive Outcome Brier Score**: 0.0016
- **Resolution Time MAE**: 0.0365 hours

### Continuous Decision Scoring (0 to 100 Range)
- **Mean Absolute Error (MAE)**: 1.2127
- **Root Mean Squared Error (RMSE)**: 1.7902

### Reliability & Selective Abstention
- **Decision Coverage**: 91.67%
- **Selective Accuracy (Under Abstention)**: 100.00%
- **Selective Risk**: 0.0000
- **Prohibited Candidate Selection Rate**: **0.0000%** (24 / 24 hard policy constraints strictly obeyed)

---

## 4. Unresolved Issues and Research Roadmap
1. **Out-of-Domain Generalization**: Evaluation was conducted on in-domain communication, retail, ops, and support holdouts. Transfer to healthcare or legal decisions requires domain-specific policy adaptation.
2. **GPU Training Acceleration**: While CPU training with feature caching runs in ~175 seconds, full unfreezing of all 22 layers of ModernBERT during multi-task training requires a GPU with $\ge 16\text{GB}$ VRAM.
3. **Continuous Real-Time Telemetry**: Episodic and feedback modules are integrated with persistent SQLite/PostgreSQL storage; live enterprise production will require continuous streaming ingestion pipelines (e.g. Kafka).
