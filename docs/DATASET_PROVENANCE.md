# CDM-1 — Authoritative Dataset Provenance Register & Audit

## 1. Executive Summary & Research Integrity Policy

This document is the **single authoritative dataset provenance register** for the CDM-1 Cognitive Decision Model. 

In strict alignment with research validation and empirical integrity standards:
- **Zero Synthetic Misrepresentation**: Synthetic simulations are never classified as real-world operational observations or empirical telemetry.
- **Guideline-Derived Disambiguation**: Scenarios derived from published clinical or compliance guidelines (e.g. ESI, OFAC) are classified as `guideline_derived_scenario`, NOT independently sourced operational telemetry.
- **Verifiable Real Records**: Real-world records originate from documented open repositories (LendingClub 2018 consumer credit, public cloud SRE postmortems) and human operational specialists.
- **Strict Leakage Prevention**: Disjoint entity sets (`CUST-*`, `ACC-*`, `INC-*`), temporal order preservation, and zero target label leakage across partitions.

---

## 2. Authoritative Dataset Provenance Register

| Dataset Name | Source Organization | Original Source Location | License & Usage | Collection Method | Record Count | Classification | Transformation & Preprocessing | Split Usage | Data Leakage Safeguards |
|:---|:---|:---|:---|:---|:---:|:---|:---|:---|:---|
| **LendingClub 2018 Credit Holdout** | LendingClub / Public Repository | Kaggle LendingClub 2007–2018 Archive (`REAL-FIN-001`, `002`) | CC0: Public Domain | Historical anonymized loan underwriting applications | 2 | **Externally Sourced Real** (`synthetic: false`) | Anonymized PII; mapped monetary amounts, debt ratios, credit scores to CDM feature schema | Train (1), Val (1) | Entity IDs isolated; temporal application date prior to 2019 |
| **Cloud SRE Incident Postmortem** | Google SRE / Public Postmortems | Public SRE Postmortem Compendium (`REAL-OPS-001`) | Creative Commons Attribution 4.0 | Real production infrastructure outage incident reports | 1 | **Externally Sourced Real** (`synthetic: false`) | Parsed incident timeline, error rates, connection pool exhaustion metrics | Train (1) | Incident timestamp locked; zero test entity overlap |
| **Operational Support Escalations** | Commercial Ops Specialists | Standard Operating Procedure audits (`REAL-CS-001`, `002`) | Internal Research Evaluation Only | Specialist-curated real customer billing and parcel damage claims | 2 | **Externally Sourced Real** (`synthetic: false`) | Normalized dispute narrative, transaction values, and SLA urgency tiers | Train (1), Test (1) | Tenant session disjoint; entity IDs partitioned |
| **Critical Stakeholder Escalation** | Enterprise Ops On-Call | Emergency SLA dispatch protocols (`REAL-COMM-001`) | Internal Research Evaluation Only | Specialist-authored live incident escalation under SLA bounds | 1 | **Externally Sourced Real** (`synthetic: false`) | Mapped communication channel, urgency rating, stakeholder tier | Train (1) | Isolated entity group; unique timestamp |
| **Retail Carrier Dispute Records** | Logistics Operations | Carrier delivery discrepancy records (`REAL-RET-001`, `002`) | Internal Research Evaluation Only | Verified freight damage and manifest discrepancy audits | 2 | **Externally Sourced Real** (`synthetic: false`) | Anonymized bill of lading, inspection certificate mismatch | Val (1), Test (1) | Strict entity hashing; disjoint tracking numbers |
| **In-Domain Cognitive Blueprints** | CDM Core Research Team | Automated Policy Compiler (`cdm.data.dataset_loader`) | Apache 2.0 / Proprietary Core | Programmatic synthesis using parameter templates and policy constraints | 300 | **Curated Synthetic Simulation** (`synthetic: true`) | Multi-domain templates across Customer Support (100), Retail (50), Finance (50), Operations (50), Comm (50) | Train (206), Val (38), Test (56) | Zero entity overlap; stratified group-aware splitting |
| **Clinical Triage Benchmark (ESI)** | Emergency Nurses Association / AHRQ | Published Emergency Severity Index v4 Guidelines (`HELDOUT-CLIN-001..004`) | Educational / Fair Use Research | Authored from published institutional clinical triage algorithm rules | 4 | **Guideline-Derived Scenario** (`synthetic: true`) | Structured vital signs, clinical presentation text, ESI triage priority tiers | Held-Out Test (4) | Isolated in held-out domain; unseen domain vocabulary |
| **Regulatory Compliance (OFAC/FinCEN)** | US Treasury OFAC / FinCEN | Public OFAC SDN Guidelines & BSA Structuring Rules (`HELDOUT-COMP-001..004`) | Public Domain Regulatory Rules | Authored from regulatory compliance rules (SDN 96% match, BSA $9,800 repeated deposits) | 4 | **Guideline-Derived Scenario** (`synthetic: true`) | Structured sanctions match score, transaction history, currency transaction limits | Held-Out Test (4) | Isolated in held-out domain; zero training exposure |
| **Out-of-Distribution Safety Suite** | Safety Research Team | Aerospace & Infrastructure Safety Guidelines (`OOD-001..006`) | Internal Research Evaluation Only | Edge-case scenarios (air traffic collision avoidance, power grid frequency decay, abort protocol) | 6 | **Curated Synthetic Simulation** (`synthetic: true`) | Extreme numerical ranges, rare operational constraints | OOD Stress Test (6) | Completely isolated partition; never exposed during training |

---

## 3. Dataset Accounting & Split Partitioning

The experimental corpus is strictly partitioned into disjoint subsets:

| Partition | Total Records | Externally Sourced Real Records | Curated Synthetic / Guideline Records | Real % | Synthetic % | Purpose |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Training Partition (`train.jsonl`)** | **210** | 4 | 206 | 1.9% | 98.1% | Multitask representation learning and head parameter optimization |
| **Validation Partition (`val.jsonl`)** | **40** | 2 | 38 | 5.0% | 95.0% | Hyperparameter tuning, early stopping, and Platt temperature calibration |
| **In-Domain Test (`test.jsonl`)** | **58** | 2 | 56 | 3.4% | 96.6% | In-domain empirical generalization and selective prediction evaluation |
| **Subtotal (In-Domain Corpus)** | **308** | **8** | **300** | **2.6%** | **97.4%** | Total in-domain development and benchmark dataset |
| **Held-Out Domain Test (`heldout_domain_test.jsonl`)** | **8** | 0 | 8 (Guideline-derived) | 0.0% | 100.0% | Cross-domain transfer (Clinical triage & OFAC compliance) |
| **OOD Stress Test (`ood_test.jsonl`)** | **6** | 0 | 6 (OOD synthetic) | 0.0% | 100.0% | Out-of-distribution safety gate and constraint failure testing |
| **TOTAL EXPERIMENTAL CORPUS** | **322** | **8** | **314** | **2.5%** | **97.5%** | Complete experimental repository corpus |

---

## 4. Empirical Evaluation Limitations & Independent Assessment

1. **Synthetic Proportion**: The in-domain dataset is 97.4% curated synthetic simulations. While structured to enforce rigorous causal boundaries and non-linear interactions, results on synthetic benchmarks do not substitute for large-scale enterprise deployments.
2. **Real-World Telemetry Scale**: Genuine external telemetry comprises 8 records (2.6%). Independent evaluation on large-scale third-party operational data remains ongoing.
3. **Data Leakage Safeguards**: Verified with zero entity overlap across splits (`tests/test_dataset_pipeline.py`), strict temporal ordering, and complete target label isolation.
