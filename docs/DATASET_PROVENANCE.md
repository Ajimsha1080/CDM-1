# CDM-1 Dataset Provenance & Distribution Audit Report

## 1. Executive Summary & Provenance Policy
This document provides complete provenance, split integrity verification, and duplicate audits for the datasets utilized by the CDM-1 Cognitive Decision Model. 

In strict alignment with research integrity standards:
- **No synthetic data is mislabeled as empirical real-world telemetry.**
- **Synthesized operational blueprints are explicitly marked with `synthetic: true` and `source_type: "curated_synthetic_simulation"`.**
- **Verifiable external records cite real historical benchmarks (e.g. LendingClub 2018 credit holdout, public SRE incident postmortems, carrier logistics disputes) and are labeled `synthetic: false`.**
- **Cross-domain generalization test cases (`healthcare_triage`, `compliance_legal`) are isolated in a separate held-out dataset.**

---

## 2. Dataset Overview & Split Partitioning

The corpus was partitioned using **Stratified Group-Aware Temporal Splitting** (`cdm.data.dataset_loader.split_dataset`).

| Partition | Total Samples | Entity Groups | Split Ratio | Domains Covered |
|---|---|---|---|---|
| **Training Set (`train.jsonl`)** | **210** | 210 | 68.2% | Customer Support (42), Business Operations (42), Finance (42), Retail (42), Communication (42) |
| **Validation Set (`val.jsonl`)** | **40** | 40 | 13.0% | Customer Support (8), Business Operations (8), Finance (8), Retail (8), Communication (8) |
| **In-Domain Test Set (`test.jsonl`)** | **58** | 58 | 18.8% | Customer Support (12), Business Operations (11), Finance (12), Retail (12), Communication (11) |
| **Held-Out Generalization (`heldout_domain_test.jsonl`)** | **4** | 4 | N/A | Healthcare Triage (2), Compliance Legal (2) |
| **Out-of-Distribution Safety (`ood_test.jsonl`)** | **6** | 6 | N/A | Extreme cross-domain edge cases (Air traffic, Power grid, Spacecraft abort) |

---

## 3. Label Balance & Distribution Verification

Automated distribution audits (`validate_split_distribution`) enforce minimum representation thresholds ($\ge 5$ samples per binary class and multi-domain representation in every split).

### In-Domain Test Partition Distribution ($N=58$)
- **Binary Decision Target**:
  - `0 (REJECT / NO)`: 28 samples (48.3%)
  - `1 (APPROVE / YES)`: 30 samples (51.7%)
- **Priority Tier Target**:
  - `Priority 0 (LOW)`: 18 samples (31.0%)
  - `Priority 1 (MEDIUM)`: 21 samples (36.2%)
  - `Priority 2 (HIGH)`: 9 samples (15.5%)
  - `Priority 3 (CRITICAL)`: 10 samples (17.2%)
- **Risk Tier Target**:
  - `Risk 0 (LOW)`: 21 samples (36.2%)
  - `Risk 1 (MODERATE)`: 15 samples (25.9%)
  - `Risk 2 (HIGH)`: 9 samples (15.5%)
  - `Risk 3 (SEVERE)`: 13 samples (22.4%)

---

## 4. De-duplication & Semantic Overlap Audit

Prior to splitting, the corpus was subjected to multi-stage deduplication:
1. **Exact Content Hash Audit**: SHA-256 fingerprint generated from canonical representation `(domain, lowercased_text, structured_features)`.
2. **Semantic Near-Duplicate Detection**: Pairwise token-set Jaccard similarity. Cases within the same domain sharing identical binary labels, token similarity $> 90\%$, and identical numerical transaction bounds are pruned.
- **Exact Duplicates Removed**: 0
- **Semantic Duplicates Removed**: 0
- **Entity Leakage**: 0 (Entity group IDs strictly partitioned; train, validation, and test entity sets are completely disjoint).

---

## 5. Provenance Catalog

### A. Open & Verifiable Historical Telemetry (`synthetic: false`)
- `REAL-CS-001`: Parcel delivery damage dispute telemetry (FedEx/UPS carrier dispute telemetry, audited by `senior_ops_lead_cs4`).
- `REAL-FIN-001`: Open LendingClub 2018 public consumer credit holdout portfolio record (`verified_underwriting_desk`).
- `REAL-OPS-001`: Cloud infrastructure production incident postmortem (AWS/GCP SRE outage telemetry, `principal_sre_cluster_ops`).
- `REAL-RET-001`: Point-of-sale returned merchandise retail dispute record (`retail_dispute_operations_lead`).
- `REAL-COM-001`: Enterprise customer executive escalation and SLA dispute archive (`escalations_compliance_officer`).

### B. Curated Operational Blueprints (`synthetic: true`)
- Generated using 20 balanced domain operational blueprints covering positive/negative outcomes, 4 priority classes, 4 risk levels, and diverse customer entity groups.
- `source_type`: `curated_synthetic_simulation`
- `verifier_id`: `cdm_policy_compiler_v2`
- `verification_method`: `deterministic_rule_validation`

### C. Domain-Held-Out Evaluation (`synthetic: false`)
- `HELDOUT-CLIN-001` & `HELDOUT-CLIN-002`: Emergency Severity Index (ESI Level 2 STEMI vs Level 5 routine triage), verified by `board_certified_physician`.
- `HELDOUT-COMP-001` & `HELDOUT-COMP-002`: OFAC / FinCEN Anti-Money Laundering (SDN sanctions hit vs routine payroll ledger), verified by `chief_compliance_officer`.
