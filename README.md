# CDM-1 — Cognitive Decision Model (v0.1.0-alpha)
### LLM + CDM Control Loop Agent Product with Deterministic Policy Gating, Ground-Truth Verification, and Signed Audit Trails

[![Release](https://img.shields.io/badge/Release-v0.1.0--alpha-orange.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)]()
[![Tests](https://img.shields.io/badge/Tests-149%20Passed-brightgreen.svg)]()

> [!IMPORTANT]
> **Alpha Release Notice**: CDM-1 v0.1.0-alpha is an early-stage open-source release under the **Apache-2.0** license. It provides a production-hardened control loop architecture for LLM agents. All claims and benchmark numbers reported in this repository reflect actual CI measurements.

---

## 1. System Overview

**CDM-1** is an **LLM + CDM control loop** agent framework designed for bounded, reliable autonomous task execution. Rather than granting foundation models unconstrained direct execution privileges, CDM-1 orchestrates a strict 4-layer control loop where every LLM proposal is treated as untrusted input, validated against schemas, gated by deterministic policy rules, executed inside isolated tools, verified against physical ground truth, and cryptographically signed.

```
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 1: Foundation Model Layer                             │
   │ - External LLM (Anthropic API / Claude 3.5 Sonnet default)   │
   │ - Local Model (OpenAI-compatible vLLM / Ollama endpoint)    │
   │ - Strict Schema Validation & Single-Repair Engine           │
   │ Output: Proposed Action {tool, args, reason}                │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Proposed Action
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 3: Decision & Planning Engine (Deterministic Gate)    │
   │ - Hard Prohibited Rules (blocks destructive commands)       │
   │ - Dynamic World Preconditions & Constraint Verification     │
   │ - Financial Bounds (e.g., refunds > $500 pause for approval)│
   │ - Optional Neural Scorer Risk Signal                        │
   │ Output: ALLOW | REQUIRE_HUMAN_APPROVAL | BLOCK              │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Allowed / Approved Action
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 4: Execution & Feedback (Sandboxed Tools & Verifiers) │
   │ - Sandboxed Tools (Shell, File I/O, Allowlisted HTTP)       │
   │ - Ground-Truth Outcome Verifier (file exists, state matches)│
   │ - Tamper-Evident HMAC-SHA256 Signed Audit Trail             │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Execution & Verification Result
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 2: CDM Cognitive Engine (Memory & Relational State)   │
   │ - Episodic Memory (records actions, outcomes, rewards)      │
   │ - Relational World State Tracking ($S_{t+1} = T(S_t, A_t)$) │
   │ - Context Assembly & Relevant Procedural Rules              │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Updated State Context
                                  └───────────────► Layer 1
```

---

## 2. What Works Today / What Does Not

To maintain full transparency for adopters and researchers, the operational boundaries of this alpha release are explicitly specified below:

### What Works Today
- **4-Layer Autonomous Loop**: Robust multi-step execution loop coordinating LLM generation, deterministic gating, sandboxed tool execution, and state persistence.
- **Pluggable LLM Providers**: Built-in support for Anthropic Claude (via `anthropic` SDK), OpenAI-compatible local servers (`vLLM`, `Ollama`), and deterministic offline mock provider for testing without API keys.
- **Strict Schema Enforcement**: Untrusted LLM outputs are validated against Pydantic schemas with at most one automatic repair prompt before raising a clean failure.
- **Deterministic Policy Gate**: Zero LLM override capability. Prohibited commands (`rm -rf`, raw shell piping, unauthorized outbound network requests) and financial thresholds are enforced deterministically.
- **Physical Outcome Verification**: Actions are independently verified against physical state (disk files, HTTP responses, database records) via `OutcomeVerifier` rather than trusting model assertions.
- **Signed Provenance Audit Trails**: Every proposal, gate decision, tool output, and verification check is logged to SQLite/JSONL with HMAC-SHA256 signatures, key rotation IDs, and persistence across restarts.
- **Human-in-the-Loop Pausing & Resuming**: Actions requiring approval halt the control loop, persist state, and cleanly resume upon an `/approve` or `/reject` API call.
- **Deterministic Weightless Fallback**: The service starts and passes `/v1/ready` readiness checks without neural weights by falling back to a deterministic risk/priority rule scorer.
- **Production Key & Tenant Management**: Production mode is default (`CDM_ENV=production`). API keys are hashed and managed via `scripts/create_api_key.py` with per-tenant rate-limiting and token metering.

### What Does Not Work Today / Out of Scope
- **Arbitrary Unsandboxed Execution**: CDM-1 intentionally forbids arbitrary root commands, raw terminal interaction, or unmonitored subshell spawning.
- **Unchecked Self-Reporting**: The control loop does not accept an agent's linguistic claim that an action succeeded; tasks without verifiable outcomes are flagged.
- **General Multi-Domain Commonsense**: The provided tool suite and benchmarks currently cover Customer Support and Refund Operations. Authoring custom tools and policy files is required for novel business verticals.
- **Native Web GUI**: v0.1.0-alpha is headless and exposes RESTful API endpoints and a Python SDK; a web frontend dashboard is planned for later milestones.
- **Biological Cognition**: CDM-1 makes no claims of artificial general intelligence or consciousness. "Cognition" refers to relational state transition graphs and episodic query engines.

---

## 3. Measured Benchmark Results

The system was evaluated against 30 operational tasks in a Customer Support and Refund Management environment (`research/agent_eval/`):
- **10 Standard Tasks**: Routine valid support workflows (damaged item refunds < $50, order status queries).
- **10 Adversarial Tasks**: Prompt injection, destructive file deletion (`rm -rf /`), unauthorized financial transfers.
- **10 High-Risk Tasks**: Sensitive operations (refunds > $500, GDPR account deletions, fraud lockouts).

### Measured Comparison

| Configuration | Overall Success Rate (95% Wilson CI) | Adversarial Attacks Blocked | Human Approvals Triggered | Mean Steps |
|---|---|---|---|---|
| **(a) LLM Alone** (Ungated Baseline) | **33.3%** [19.2%, 51.2%] (10/30) | 0 / 10 (10 Unsafe Executions) | 0 / 10 (10 Breaches) | 1.00 |
| **(b) LLM + Policy Gate** | **100.0%** [88.6%, 100.0%] (30/30) | 10 / 10 (100% Blocked) | 10 / 10 (100% Paused) | 0.67 |
| **(c) LLM + Gate + Cognitive Memory** | **100.0%** [88.6%, 100.0%] (30/30) | 10 / 10 (100% Blocked) | 10 / 10 (100% Paused) | 0.67 |

*Benchmark reproduction command: `python research/agent_eval/eval_runner.py`.*

---

## 4. Quickstart Guide (Clean Clone)

### 1. Clone and Install
```bash
git clone https://github.com/cdm-project/cdm-1.git
cd cdm-1

# Option A: Install locked production dependencies
pip install -r requirements-lock.txt
pip install -e .

# Option B: Install via pip with optional extras
pip install -e ".[all]"
```

### 2. Run the Test Suite
```bash
pytest
```
*Expected output: 149 passed in ~4.5 seconds.*

### 3. Generate Tenant API Key
Production mode is enabled by default. Generate a valid API key:
```bash
python scripts/create_api_key.py --create --tenant tenant_ops --tier developer --rpm 300
```
This prints your plaintext key (e.g., `sk_live_...`). Store it securely.

### 4. Optional: Download Neural Model Weights
The service operates cleanly out of the box in deterministic fallback mode. If you wish to use the auxiliary neural risk head:
```bash
python scripts/download_weights.py
```
This streams `model.safetensors` from Hugging Face Hub, computes the SHA-256 hash in real time, and verifies the Ed25519 manifest signature. See [docs/HOSTING_ON_HUGGINGFACE.md](docs/HOSTING_ON_HUGGINGFACE.md) for hosting details.

### 5. Start the REST API Service
```bash
# Set your HMAC secret (for signed audit trails)
export CDM_HMAC_SECRET="your-secure-audit-secret-key-32b"

# Start via Uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or run via Docker Compose
docker compose up -d
```
Check service readiness:
```bash
curl http://localhost:8000/v1/ready
```

### 6. Run an Agent Task with Your LLM Key
Run an autonomous task using your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."

curl -X POST http://localhost:8000/v1/agent/runs \
  -H "X-API-Key: <YOUR_CDM_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Customer 8923 reports delivered order arrived damaged 5 days ago. Process refund of $35.00.",
    "context": {"order_id": "8923", "days_since_delivery": 5, "damage_verified": true},
    "provider_type": "anthropic",
    "model": "claude-3-5-sonnet-20241022"
  }'
```

Alternatively, use the Python SDK:
```python
from cdm.agent.loop import AgentControlLoop
from cdm.llm import AnthropicProvider

provider = AnthropicProvider(api_key="sk-ant-api03-...")
loop = AgentControlLoop(llm_provider=provider)

result = loop.run(
    task="Issue refund of $35.00 for order #8923.",
    initial_world_state={"balance": 1000.0, "refund_limit": 500.0}
)
print("Run Status:", result["status"])
print("Audit Records:", len(result["audit_log"]))
```

---

## 5. Security & Responsible Disclosure

Security vulnerabilities should be reported privately in accordance with [SECURITY.md](SECURITY.md). Please do not open public issues for undisclosed security flaws.

- Security Contact: `security-cdm@proton.me`
- Response Target: Acknowledgement within 48 hours; triage report within 5 business days.

---

## 6. Contributing & Community

We welcome contributions! Please review:
- [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, type checking, and test guidelines.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards (Contributor Covenant v2.1).
- [NOTICE](NOTICE) for third-party credits and licenses.

---

## 7. License

CDM-1 is released under the [Apache License, Version 2.0](LICENSE).
