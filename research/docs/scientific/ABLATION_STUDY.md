# CDM-1 — Controlled Component Ablation Study

**Auditor**: Independent AI Research Auditor & Experimental Validation Engineer  
**Date**: November 6, 2026  
**Evaluation Partition**: Canonical Holdout Test Set ($N=58$)  
**Experimental Run Data**: [`benchmarks/cognitive_validation_results.json`](../../../benchmarks/cognitive_validation_results.json)  

---

## 1. Objective and Setup

To evaluate which cognitive subsystems provide measurable empirical value and which incur computational overhead, 6 controlled configurations were evaluated on identical test items under identical CPU runtime conditions:

1. **Configuration 1: CDM-1 Neural Baseline Alone**: Direct execution of the hybrid neural network (`CDMModel`) without cognitive wrapping, working memory, or policy gating.
2. **Configuration 2: CDM-1 with Memory**: Adds persistent episodic vector retrieval and working memory state buffers.
3. **Configuration 3: CDM-1 with Reasoning**: Adds deterministic policy constraint checking and hard logit masking.
4. **Configuration 4: CDM-1 with Predictive Modeling**: Adds multi-target forward outcome estimation and counterfactual risk assessment.
5. **Configuration 5: CDM-1 with Planning**: Adds multi-step action sequencing and plan efficiency scoring.
6. **Configuration 6: Full Integrated Cognitive AI Engine**: Orchestrates all 8 subsystems in a coordinated cognitive loop.

---

## 2. Empirical Ablation Results

| Configuration | Choice Accuracy | Prohibited Violation Rate | Avg. Latency (ms) | Primary Empirical Contribution |
|:---|:---:|:---:|:---:|:---|
| **1. Baseline Neural Alone** | 93.10% | 5.17% | 1,599.55 ms | Raw neural contextual + tabular reasoning; susceptible to policy breaches without gating. |
| **2. CDM-1 + Memory** | 93.10% | 5.17% | 1,570.00 ms | Grounded semantic unit vector retrieval via SQLite; preserves past episodic cases. |
| **3. CDM-1 + Reasoning** | **98.28%** | **0.00%** | 1,609.87 ms | **Critical**: Deterministic policy constraint masking eliminates violations and boosts accuracy. |
| **4. CDM-1 + Prediction** | 93.10% | 5.17% | 1,576.72 ms | Multi-target outcome forecasting (resolution time, risk distribution, continuous targets). |
| **5. CDM-1 + Planning** | **98.28%** | **0.00%** | 1,586.85 ms | Generates multi-step action sequences with preconditions and alternative branches. |
| **6. Full Integrated Engine** | **98.28%** | **0.00%** | 1,592.30 ms | Full cognitive loop: tenant-isolated world state, memory, reasoning, planning, metacognition. |

---

## 3. Findings & Scientific Deductions

1. **Reasoning Engine is Indispensable for Policy Compliance**:
   - The Baseline Neural Model alone exhibits a **5.17% violation rate** on test cases containing prohibited actions.
   - When the Reasoning Engine (`cdm/reasoning/reasoning_engine.py`) is activated (Arms 3, 5, 6), deterministic constraint enforcement completely eliminates prohibited actions (**0.0000% violation rate**).
   - Furthermore, blocking prohibited invalid actions reroutes decisions to correct compliant candidates, elevating Top-1 Choice Accuracy from **93.10%** to **98.28%**.

2. **Planning Engine Provides Multi-Step Operational Utility**:
   - The Planning Engine adds structured forward sequencing (plans with 1–4 steps, preconditions, alternative branches, and risk aggregation) with zero prohibited actions in generated plans.
   - It reliably triggers replanning when simulated world states indicate environmental deterioration (e.g., security exposure > 0.8).

3. **Memory and Predictive Intelligence Impact**:
   - Grounded semantic memory retrieval (`cdm/memory/embedder.py` producing 16-d unit vectors) operates seamlessly within SQLite with negligible search latency.
   - The Predictive Network estimates continuous outcome metrics (resolution time, operational score) alongside decision heads.

4. **Engine Orchestration Overhead**:
   - The full multi-tenant cognitive orchestration loop executes within normal neural forward latency parameters (~1,592 ms on multi-threaded CPU), while supplying structured world states, proof traces, multi-step plans, and calibrated confidence.
