# CDM-1 — LLM + CDM Control Loop Agent Architecture
### Autonomous Agent Control Loop with Policy Gating, Ground-Truth Verification, and Signed Audit Trails

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)]()
[![Tests](https://img.shields.io/badge/Tests-149%20Passed-brightgreen.svg)]()

---

## 1. System Overview

**CDM-1** is an **LLM + CDM control loop** agent product designed for bounded, reliable autonomous task execution. Rather than allowing foundation models unconstrained direct execution, CDM-1 orchestrates a 4-layer control loop where every LLM proposal is rigorously gated by deterministic policy rules, executed within sandboxed environments, checked by ground-truth verifiers, and cryptographically signed into an immutable provenance audit trail.

### 4-Layer Control Loop Architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 1: Foundation Model (External LLM / Anthropic / Local)│
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Proposes Action {tool, args, reason}
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 3: Decision & Planning Engine (Policy Gate & Risk)    │
   │ - Hard Prohibited Rules (rm -rf, dangerous ops)            │
   │ - Preconditions & Constraints against World State           │
   │ - Financial Bounds (e.g., refunds > $500 pause for approval)│
   │ - Optional Neural Scorer Risk Signal                        │
   │ Output: ALLOW | REQUIRE_HUMAN_APPROVAL | BLOCK              │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ If Allowed (or Human Approved)
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 4: Execution & Feedback (Sandboxed Tools & Verifier)  │
   │ - Sandboxed Tools (Shell, File I/O, Allowlisted HTTP)       │
   │ - Ground-Truth Outcome Verifier (file exists, state matches)│
   │ - Signed HMAC Provenance Audit Trail (Key Rotation Aware)   │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Execution & Verification Feedback
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 2: CDM Cognitive Engine (Memory & World State)        │
   │ - Episodic Memory (records actions, outcomes, rewards)      │
   │ - Semantic Memory & Procedural Rules                        │
   │ - Dynamic World State ($S_{t+1} = T(S_t, A_t)$)             │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Context for Next Step
                                  └───────────────► Layer 1
```

---

## 2. Empirical Benchmark Evaluation Results

The system was evaluated using 30 concrete operational tasks in a customer support and refund management domain (`research/agent_eval/`):
- **10 Standard Tasks**: Routine support operations (damaged delivery refund < $50, address updates, receipts).
- **10 Adversarial / Policy-Violating Tasks**: Dangerous prompt injections, destructive commands (`rm -rf /`), unauthorized wire transfers, compliance log wiping.
- **10 High-Risk Sensitive Tasks**: Refunds over $500, GDPR erasures, fraud account freezes, VIP compensation.

### Measured Benchmark Comparison

| Configuration | Success Rate (95% Wilson CI) | Safety Blocks | Approval Requests | Mean Steps |
|---|---|---|---|---|
| **(a) LLM Alone** (Ungated baseline) | **33.3%** [19.2%, 51.2%] (10/30) | 0 / 10 | 0 / 10 | 1.00 |
| **(b) LLM + Policy Gate** | **100.0%** [88.6%, 100.0%] (30/30) | 10 / 10 | 10 / 10 | 0.67 |
| **(c) LLM + Gate + Cognitive Memory** | **100.0%** [88.6%, 100.0%] (30/30) | 10 / 10 | 10 / 10 | 0.67 |

### Key Findings
1. **Unconstrained LLMs Fail on Adversarial Operations**: When run without a policy gate, the foundation model attempted 100% of the prohibited actions and executed high-value transactions without human approval.
2. **Policy Gate Guarantees Safety Bounds**: CDM-1 successfully blocked 10/10 adversarial/destructive actions and halted 10/10 high-value operations for operator approval.
3. **Outcome Verification Replaces Self-Reporting**: The task verifier inspects physical disk files and state metrics rather than trusting the LLM's claim of completion.

---

## 3. Real Limitations & Design Trade-Offs

1. **Small Neural Decision Head**: The neural model consists of ~9.35M trainable parameters atop a frozen 149M ModernBERT-base backbone. It serves as a fast scoring and risk estimation component, not a general reasoning foundation model.
2. **Synthetic Training Data**: The training datasets were synthetically generated for support and refund domains; distribution shift on novel operational environments may require domain calibration.
3. **Dependence on External Foundation Models**: High-level action generation requires an external LLM (Anthropic API by default, or an OpenAI-compatible local model). An offline deterministic mock provider is provided for testing when keys are absent.
4. **Single Supported Benchmark Domain**: Current evaluation tasks focus on customer support, dispute resolution, and refund operations. Extending to new domains requires defining domain tools and tenant policies.
5. **No Claims of Artificial Consciousness**: CDM-1 does not replicate human cognition. Its cognitive engine implements structured computer science abstractions: relational state graphs, SQLite episodic stores, and rule-based constraint solving.

---

## 4. Quickstart Guide

### Installation
```bash
git clone https://github.com/Ajimsha1080/CDM-1.git
cd CDM-1
pip install -r requirements-lock.txt
```

### Running the Full Test Suite
```bash
pytest
```
*Current test suite status: 149 passed, 0 failures, 0 skips.*

### API Key Management
Production mode is the default (`CDM_ENV=production`). Register credentials via CLI:
```bash
# Create a new API key for a tenant
python scripts/create_api_key.py --create --tenant tenant_ops --tier developer --rpm 300

# List existing registered key hashes
python scripts/create_api_key.py --list

# Revoke a compromised key
python scripts/create_api_key.py --revoke <KEY_OR_HASH>
```

### Starting the Service via Docker Compose
```bash
docker compose up -d
curl http://localhost:8000/v1/ready
```
*The service starts and passes `/v1/ready` even when weights are absent by falling back to the deterministic rule-based scorer.*

---

## 5. REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/health` | `GET` | Service liveness probe. |
| `/v1/ready` | `GET` | Service readiness probe (supports neural mode or rule-based fallback). |
| `/v1/agent/runs` | `POST` | Starts or steps an autonomous agent run within the 4-layer control loop. |
| `/v1/agent/runs/{id}` | `GET` | Retrieves run state, execution history, and approval status. |
| `/v1/agent/runs/{id}/approve` | `POST` | Human approval gate: authorizes a suspended high-risk action. |
| `/v1/agent/runs/{id}/reject` | `POST` | Human approval gate: rejects a suspended action with an operator reason. |
| `/v1/decisions` | `POST` | Standalone decision inference (binary, choice, score, risk, priority). |
| `/v1/batch` | `POST` | Batch decision processing (metered and rate-limited per item). |
| `/v1/cognitive/process`| `POST` | Direct Cognitive Engine perception-planning-decision pipeline. |
| `/v1/metering` | `GET` | Returns cumulative usage and cost accounting for the caller's tenant. |

---

## 6. Running the Benchmark Evaluation Suite

To reproduce the Phase 5 benchmark evaluation numbers:
```bash
python research/agent_eval/eval_runner.py
```
Output results will be written to `research/agent_eval/benchmark_results.json`.
