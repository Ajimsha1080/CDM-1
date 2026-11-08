# CDM-1 Dataset Documentation & Provenance Report

## 1. Overview and Standards
The CDM-1 decision dataset is standardized according to the `DecisionExample` schema contract defined in [`cdm/data/data_contracts.py`](../cdm/data/data_contracts.py). Every example contains textual narrative, versioned tabular features, candidate action lists, hard policy constraints, target decisions, evidence pointers, observed outcomes, and provenance metadata.

---

## 2. Dataset Versioning & Checksums (v2.0.0)
- **Total In-Domain Examples**: 308 (Train: 210, Validation: 40, Test: 58)
- **Held-Out Generalization Examples**: 8
- **Out-of-Distribution Safety Examples**: 6
- **Total Deduplicated Examples**: 322
- **Manifest**: [`datasets/split_manifest.json`](../datasets/split_manifest.json)

| Split | Example Count | Partition Ratio | File Checksum (SHA-256) |
|---|---|---|---|
| **Train (`train.jsonl`)** | 210 | ~68.2% | `4b74b481599b63f4f6a22073695110c98e92cb227f63ee682178c66a260bbc37` |
| **Validation (`val.jsonl`)** | 40 | ~13.0% | `edb19a210bc537d3b1a66b2504a76c86f9d70b7fdfbb32eaa2a22afbe1bc0aac` |
| **Test (`test.jsonl`)** | 58 | ~18.8% | `68a963fc4f8b4fbc5ec79b7f426a662bae3a81f642d3dbb6bcb1c936361b6938` |
| **Held-Out Domain (`heldout_domain_test.jsonl`)** | 8 | N/A | `6c2cd78cf0a354e5b86a1bb25079e616e12b923e4b7aa7e0c2428821d01af01f` |
| **Out-of-Distribution (`ood_test.jsonl`)** | 6 | N/A | `c830eb0ba583a697669ba2ebba75b5b292f3922c00d437bb9a79fa4f9715a9e7` |

---

## 3. Provenance and Source Disclosures
To ensure scientific integrity, synthetic and verified real-world examples are explicitly distinguished:
- **Verified Human Expert Annotations**: 8 benchmark cases (`EXP-*`) with audited policy adherence and operational verification.
- **Curated Multi-Domain Scenarios**: 300 systematic examples (`CUR-*`) across 5 operational business domains with cross-validated operational rules, policy constraints, and forward-looking outcome telemetry.
- **Synthesized Data Transparency**: None of the curated scenarios are claimed to be raw production telemetry; they are clearly flagged with `synthetic: true` in their provenance tags.

---

## 4. Distributions

### Domain Balance (In-Domain Corpus, N=308)
- **Customer Support**: 62 examples (20.1%)
- **Communication**: 61 examples (19.8%)
- **Retail Operations**: 62 examples (20.1%)
- **Business Operations**: 61 examples (19.8%)
- **Finance & Credit**: 62 examples (20.1%)

### Multi-Task Label Distribution (In-Domain Test Partition, N=58)
- **Binary Decision (YES / NO)**: YES: 30 (51.7%), NO: 28 (48.3%)
- **Priority Tiers**: Low (0): 18, Medium (1): 21, High (2): 9, Critical (3): 10
- **Risk Tiers**: Low (0): 21, Moderate (1): 15, High (2): 9, Severe (3): 13

---

## 5. Leakage Prevention Methodology
- **Zero-Overlap Partitioning**: Complete entity isolation using group-temporal partitioning. No customer or organization identifier appears in more than one split partition.
- **Temporal Holdout**: Examples are sorted by chronological timestamp ensuring validation and test sets strictly represent subsequent temporal periods.
- **Automated Audit**: Rebuilding pipeline programmatically checks `train_ids ∩ val_ids ∩ test_ids = ∅` before exporting.
