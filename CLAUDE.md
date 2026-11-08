# CDM-1 — Agent & Engineering Guidelines

This document establishes non-negotiable operational rules for all AI coding agents, maintainers, and contributors working on the CDM-1 (Cognitive Decision Model) codebase.

---

## 1. Core Agent Rules

1. **Fix Source, Not Tests**:
   - Always fix underlying source code and business logic.
   - Never alter, weaken, comment out, or delete tests to make a failing suite pass. If a test specifies an invariant, that invariant is law.

2. **No Skipping or Weakening Tests**:
   - `pytest.skip` is strictly forbidden in normal test suites.
   - Tests requiring optional external environments must be explicitly tagged with dedicated markers (`@pytest.mark.requires_docker`, `@pytest.mark.requires_weights`) and filtered cleanly in CI runner configurations.
   - The zero-skips gate (`scripts/check_no_skips.py`) must pass on every run.

3. **No Hand-Written Results or Mock Evidence**:
   - Never write "PASS", "verified", or synthetic metrics into reports or documentation by hand.
   - All reports (`docs/VERIFICATION.md`, CI badges, manifests) must be produced directly from live, verifiable command execution.

4. **Never Touch Keys or Signatures**:
   - Never hardcode, embed, forge, or tamper with private signing keys or cryptographic signatures.
   - Digital signature verification (Ed25519) and HMAC provenance records are active security mechanisms protecting artifact and execution integrity. Private keys reside strictly in secure environment variables or external KMS.

---

## 2. Architecture Overview

CDM-1 operates as an **LLM + CDM Control Loop**:
- **Layer 1: Foundation Model Layer** — External LLM (Anthropic API / Claude 3.5 Sonnet) or local SLM. Generates candidate tool calls and action proposals.
- **Layer 2: CDM Cognitive Engine** — Dual-memory (Episodic + Semantic), perception, world state tracking, context distillation.
- **Layer 3: Decision & Planning Engine** — Goal management, invariant policy gating, risk scoring, sequential planner, candidate ranking.
- **Layer 4: Execution & Feedback** — Container-sandboxed shell tool, hardened HTTP client, signed HMAC provenance audit trail, verifiable feedback back into Layer 2.

---

## 3. Standard Verification Commands

```bash
# Run standard test suite with zero skips:
pytest -rs --strict-markers -m "not requires_docker and not requires_weights" --junitxml=test-results.xml
python scripts/check_no_skips.py test-results.xml

# Verify release manifest integrity and signature:
python scripts/check_manifest_integrity.py

# Run end-to-end smoke test:
python scripts/smoke_test.py
```
