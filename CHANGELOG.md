# Changelog

All notable changes to the CDM-1 project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-09 (Alpha Release)

### Added
- **4-Layer Agent Control Loop** (`cdm/agent/loop.py`):
  - Layer 1 Foundation Model Provider Interface (`cdm/llm/`) with Anthropic Claude, OpenAI-compatible local models, and offline mock fallbacks.
  - Layer 2 Cognitive Engine integration providing episodic memory search, semantic procedural rules, and relational world state transitions.
  - Layer 3 Deterministic Policy Gate (`cdm/gate/policy_gate.py`) evaluating hard prohibited actions, preconditions, financial limits, and neural risk scores.
  - Layer 4 Sandboxed Tools (`cdm/execution/tools/`) including timeout-bounded shell, jail-enforced file read/write/edit, and allowlisted HTTP calls.
  - Independent Ground-Truth Outcome Verifier (`cdm/execution/verifier.py`) inspecting real disk and metric states rather than trusting LLM self-reporting.
  - Tamper-evident HMAC-SHA256 signed provenance audit trail (`cdm/execution/audit.py`) with key ID rotation.
- **Agent REST API Endpoints**:
  - `POST /v1/agent/runs` - Start or execute an agent control loop run.
  - `GET /v1/agent/runs/{id}` - Retrieve run history and pending approval state.
  - `POST /v1/agent/runs/{id}/approve` - Operator approval gate to execute high-risk operations.
  - `POST /v1/agent/runs/{id}/reject` - Operator rejection gate with structured reason.
- **API Key Management CLI** (`scripts/create_api_key.py`):
  - CLI commands to generate (`--create`), display (`--list`), and revoke (`--revoke`) persistent cryptographic API keys.
- **Empirical Evaluation Benchmark** (`research/agent_eval/`):
  - 30 evaluation tasks in customer support and refund operations comparing LLM Alone, LLM + Policy Gate, and LLM + Gate + Cognitive Memory.
- **Adversarial Test Suite** (`tests/test_adversarial.py`):
  - Tests covering prompt injection in tool outputs, malformed JSON repair, prohibited tool interception, cross-tenant isolation, and budget exhaustion.
- **Automated CI Workflow** (`.github/workflows/ci.yml`):
  - GitHub Actions pipeline covering clean dependency installation, 0-skip pytest execution, secret scanning, and containerized `/v1/ready` readiness probe.

### Changed
- Converted repository from a standalone classifier model into an open-source autonomous agent control loop.
- Relocated research code, datasets, and scientific documentation under `research/`.
- Made neural weights strictly optional: if `model.safetensors` is absent, the system falls back to a deterministic rule-based scorer and passes `/v1/ready` with HTTP 200.
- Pinned ModernBERT backbone revision (`8949b909ec900327062f0ebf497f51aef5e6f0c8`) with offline `local_files_only` support.
- Hardened Docker container to multi-stage build, non-root user (`cdmuser`), and python-native healthcheck without external build tools.
- Set production mode as default (`CDM_ENV=production`) and eliminated hardcoded demo credentials.
