# CDM-1 Dataset Documentation & Provenance Report

## 1. Overview and Standards
The CDM-1 decision dataset is standardized according to the `DecisionExample` schema contract defined in [`cdm/data/data_contracts.py`](../cdm/data/data_contracts.py). Every example contains textual narrative, versioned tabular features, candidate action lists, hard policy constraints, target decisions, evidence pointers, observed outcomes, and provenance metadata.

---

## 2. Dataset Versioning & Checksums (v1.0.0)
- **Total Deduplicated Examples**: 134
- **Manifest**: [`datasets/split_manifest.json`](../datasets/split_manifest.json)

| Split | Example Count | Partition Ratio | File Checksum (SHA-256) |
|---|---|---|---|
| **Train (`train.jsonl`)** | 90 | ~67.2% | `6f0d9c5c78be9828e985f412eb025d9f1c726967056041c78a209b344962ecd4` |
| **Validation (`val.jsonl`)** | 20 | ~14.9% | `2d591c303f11071afdbed0250eaebc94a0c3170286bdb42753991cc5234b89f5` |
| **Test (`test.jsonl`)** | 24 | ~17.9% | `8cfef5669893c74ac7eb90eb318cb379516c7cca2900e4dc8216fdc6d3784564` |

---

## 3. Provenance and Source Disclosures
To ensure scientific integrity, synthetic and verified real-world examples are explicitly distinguished:
- **Verified Human Expert Annotations**: 6 benchmark cases (`EXP-CS-001`, `EXP-CS-002`, `EXP-COMM-001`, `EXP-COMM-002`, `EXP-RET-001`, `EXP-OPS-001`) with audited policy adherence and operational verification.
- **Curated Multi-Domain Scenarios**: 128 systematic examples (`CUR-*`) with cross-validated operational business rules, policy constraints, and forward-looking outcome telemetry.
- **Synthesized Data Transparency**: None of the curated scenarios are claimed to be raw production telemetry; they are clearly flagged with `synthetic: true` in their provenance tags.

---

## 4. Distributions

### Domain Balance
- **Customer Support**: 34 examples (25.4%)
- **Communication**: 34 examples (25.4%)
- **Retail Operations**: 33 examples (24.6%)
- **Business Operations**: 33 examples (24.6%)

### Multi-Task Label Distribution
- **Binary Decision (YES / NO)**: YES: 92 (68.7%), NO: 42 (31.3%)
- **Priority Tiers**: Low (0): 50, Medium (1): 25, High (2): 25, Critical (3): 34
- **Risk Tiers**: Low (0): 50, Moderate (1): 42, Severe (3): 34, High (2): 8

---

## 5. Leakage Prevention Methodology
- **Zero-Overlap Partitioning**: Complete entity isolation using group-temporal partitioning. No customer or organization identifier appears in more than one split partition.
- **Temporal Holdout**: Examples are sorted by chronological timestamp ensuring validation and test sets strictly represent subsequent temporal periods.
- **Automated Audit**: Rebuilding pipeline programmatically checks `train_ids ∩ val_ids ∩ test_ids = ∅` before exporting.
