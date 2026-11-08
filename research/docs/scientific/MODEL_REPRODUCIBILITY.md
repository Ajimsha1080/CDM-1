# CDM-1 — Model Reproducibility & Artifact Verification

**Auditor**: Independent AI Research Auditor & Experimental Validation Engineer  
**Date**: November 6, 2026  
**Artifact Evaluated**: `experiments/release/cdm-1-general/model.safetensors`  
**Manifest Reference**: [`dist/release_manifest.json`](../../../dist/release_manifest.json)  

---

## 1. Checkpoint Verification & Cryptographic Hashes

The distributed model checkpoint is saved in SafeTensors format and strictly verified:

| Property | Measured Value | Verification Result |
|:---|:---|:---:|
| **File Path** | `experiments/release/cdm-1-general/model.safetensors` | Verified Present |
| **File Size** | **633,465,536 bytes** (~633.47 MB) | Verified Match |
| **SHA-256 Digest** | `9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a` | Strict Match |
| **Model Configuration** | `experiments/release/cdm-1-general/config.json` | Verified Match |
| **Feature Schema** | `experiments/release/cdm-1-general/feature_schema.json` | Verified Match |
| **Packaged Tokenizer** | `experiments/release/cdm-1-general/tokenizer.json` | Offline Usable |

---

## 2. Parameter Accounting

Audited via `model.count_parameters()`:
- **Trainable Parameters**: **9,345,732**
  - Language Encoder live sequence projection: 196,864
  - Structured Feature Encoder: 173,312
  - Cognitive Gated Fusion: 394,496
  - Action-Conditioned Predictive Network: 69,509
  - Multi-Head Decision Network: 134,663
  - Candidate Scorer & Projector: 8,376,888
- **Frozen Parameters**: **149,014,272** (`answerdotai/ModernBERT-base` contextual language backbone)
- **Total Parameters**: **158,360,004** (~158.36M, fulfilling the 150–200M design target)

---

## 3. Strict Checkpoint Loading & Clean Offline Execution

The checkpoint was tested via automated extraction into a temporary clean directory:
1. `dist/CDM-1-Complete-With-Weights.zip` extracted to an isolated directory.
2. Initialized via `CognitiveDecisionModel.from_pretrained(extracted_dir, local_files_only=True)`.
3. Offline decision inference executed:
   - Output: `Decision=YES`, `Confidence=0.9498`.
   - Result: **Strict verification passed without remote network requests.**

---

## 4. End-to-End Reproduction Commands

Every reported empirical result can be independently reproduced via the following commands:

```bash
# 1. Environment Installation
pip install -r requirements.txt

# 2. Execute Full Multi-Task Training Pipeline (15 Epochs)
python training/train_multitask.py

# 3. Post-Hoc Probability Calibration
python training/calibrate.py

# 4. Run Canonical Benchmark Suite (Baselines + CDM-1 across Seeds 42, 123, 999)
python benchmarks/benchmark_suite.py

# 5. Run Scientific Validation & Ablation Harness
python benchmarks/cognitive_validation_harness.py

# 6. Run Complete Test Suite (60 Automated Tests)
pytest tests/

# 7. Package and Verify Release Archives
python scripts/package_release.py --copy-downloads
```
