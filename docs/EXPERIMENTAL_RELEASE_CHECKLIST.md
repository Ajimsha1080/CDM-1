# CDM-1 Experimental Software Release Checklist

This document provides a pre-flight verification checklist for releasing and deploying the **CDM-1 (Cognitive Decision Model)** experimental release.

---

## 1. Environment & Runtime Integrity

- [x] **Supported Python Runtime**: Verified on Python 3.14.0 (supports Python 3.10 to 3.14).
- [x] **Dependency Tree Verification**: `python -m pip check` reports `No broken requirements found.`
- [x] **Byte-Compilation**: `python -m compileall -q .` completes cleanly with exit code 0.
- [x] **Pinned Reproducibility**: `requirements-lock.txt` pins exact versions of PyTorch, Transformers, Safetensors, Pydantic, FastAPI, Click, and Cryptography.

---

## 2. Automated Test Suite & Zero-Skips Policy

- [x] **Pytest Collection**: `pytest --collect-only -q` collects 278 tests across 26 test modules with **0 collection errors**.
- [x] **Test Execution**: Full test suite passes: **265 passed, 0 failures, 0 errors, 0 skips**.
- [x] **Zero-Skips Enforcement**: `python scripts/check_no_skips.py test-results.xml` confirms 0 skipped tests out of 265 executed.
- [x] **Live Neural Inference Suite**: `pytest -m requires_weights` passes 7/7 tests without fallbacks.

---

## 3. Cryptographic Security & Release Manifests

- [x] **Public Key Consistency**: Verified `TRUSTED_RELEASE_PUBLIC_KEY` in `cdm/security/manifest_verification.py` matches `docs/RELEASE_SECURITY.md` and `release_manifest.json` exactly:
  `4784afcf98a872ec65cc951f8803c6619e777923b559196f4cd9297bbadf56a5`
- [x] **SafeTensors Model Checkpoint**:
  - Path: `experiments/release/cdm-1-general/model.safetensors`
  - Exact Size: `633,465,536 bytes`
  - SHA-256 Digest: `9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a`
  - Tensor Count: Exactly 215 tensors (ModernBERT-base + multi-task heads).
- [x] **Dual Verification Modes**:
  - Mode A (`--mode software_only`): Verified succeeds for weightless distribution.
  - Mode B (`--mode full_model`): Verified strictly fails if weights are missing, truncated, or tampered.
  - Regression Suite (`tests/test_manifest_integrity.py`): 20/20 tests passed.

---

## 4. Dataset Provenance & Empirical Benchmarks

- [x] **Authoritative Dataset Register**: Documented in `docs/DATASET_PROVENANCE.md`:
  - 308 In-Domain Records (300 synthetic simulations / 97.4%, 8 real production records / 2.6%).
  - 8 Held-Out Test Records (Emergency Severity Index triage, OFAC/FinCEN compliance guidelines).
  - 6 Out-of-Distribution Stress Records (extreme volatility and cascading failure scenarios).
- [x] **Reproducible Benchmark Claims**: `docs/MODEL_CARD.md` aligned with empirical measurements from `research/benchmarks/benchmark_results.json` (N=58 holdout cases, 94.83% Top-1 Choice, 96.55% Priority, 94.83% Risk Level).
- [x] **Synthetic Disclosures**: Explicit notices in `docs/BENCHMARKS.md` and `docs/MODEL_CARD.md` disclosing synthetic simulations.

---

## 5. Runtime Modes & Fail-Safe Operation

- [x] **Software-Only Mode**: Weightless execution runs deterministically with rule-based scoring (`prediction_source: "rule_based_fallback"`, `used_neural_inference: False`).
- [x] **Production Mode Strictness**: Production startup (`CDM_ENV=production` or `CDM_PRODUCTION_MODE=1`) strictly refuses to start or fall back when weights are missing (`FileNotFoundError` / `RuntimeError`).
- [x] **Candidate Pre-Ranking Validation**: Action candidate schemas and dynamic world preconditions validated before ranking.
- [x] **Fail-Closed Abstention**: Unauthorized actions or failed candidates safely trigger abstention.
- [x] **Decision Provenance**: Complete structured dictionary persisted on every step.

---

## 6. API, Security & Production Readiness

- [x] **Authentication**: `PersistentApiKeyManager` with Argon2/SHA-256 hashed bearer tokens.
- [x] **Tenant Isolation**: Strict `tenant_id` partitioning across all databases and sessions.
- [x] **Payload Limits**: 1 MB global body size limit (HTTP 413).
- [x] **Rate Limiting**: `DurableUsageMeter` enforces per-tenant requests-per-minute limits.
- [x] **Error Handling**: Internal stack traces masked from clients in production.
- [x] **Operational Probes**:
  - `/v1/health` (liveness).
  - `/v1/ready` (readiness with model, SHA-256, and Ed25519 signature checks).
- [x] **Storage Crash Consistency**: SQLite WAL mode enabled across all stores.

---

## 7. Packaging & Distribution Verification

- [x] **Clean Extraction**: `scripts/package_release.py` extracts archives to temporary directory and executes offline test inference.
- [x] **Distribution Archives**:
  - `CDM-1-Complete-With-Weights.zip` (1,122 MB, includes full neural weights).
  - `CDM-1-Project.zip` (2.00 MB, lightweight software-only package).
- [x] **Git Repository**: Branch `main` clean and fully synchronized with origin.
