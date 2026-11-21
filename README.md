# CDM-1 — Neuro-Symbolic Cognitive Decision Model
### Production-Hardened Cognitive Engine & Deterministic Safety Governor for Autonomous Agents

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Release-v0.1.0--alpha-orange.svg" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.12+-EE4C2C.svg" alt="Framework"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Tests-403%20Passed%20(100%25)-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/Latency-3ms-success.svg" alt="Latency">
  <img src="https://img.shields.io/badge/API-OpenAI%20%26%20Claude%20Drop--In-purple.svg" alt="Compatibility">
</p>

<p align="center">
  <strong>CDM-1</strong> is a high-performance, open-source <strong>Neuro-Symbolic Cognitive Decision Architecture</strong> engineered for enterprise reliability. Where general foundation models are slow, expensive, and prone to hallucination, CDM-1 provides <strong>3-millisecond local inference</strong>, mathematical policy gating, multi-step goal planning, action-conditioned outcome simulation, and a <strong>100% drop-in replacement API for OpenAI, Claude, and Gemini</strong>.
</p>

<p align="center">
  <a href="#-quickstart-60-seconds">Quickstart</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-drop-in-api-compatibility-openai--claude">API Compatibility</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-developer-playground-ui">Playground UI</a> •
  <a href="#-benchmarks">Benchmarks</a>
</p>

---

## ⚡ Why CDM-1?

Enterprises cannot deploy unconstrained LLMs into production environments where a single hallucination causes financial loss, security breaches, or regulatory non-compliance. Prompt engineering and system prompts are easily bypassed by prompt injection and model drift.

| Capability | Standard Generative LLM (GPT-4 / Claude) | CDM-1 Cognitive Architecture |
| :--- | :--- | :--- |
| **Inference Latency** | 1,000 ms – 5,000 ms (Cloud network overhead) | **⚡ 3 ms – 4 ms** (Instant local CPU/GPU execution) |
| **Safety & Policy Gating** | Soft linguistic prompts (Bypassable) | **🛡️ Hard Mathematical Logit Masking ($-\infty$)** |
| **Future Simulation** | Guesswork based on next-token prediction | **🔮 Action-Conditioned Predictive Network** ($P(\text{success})$, $T_{\text{res}}$) |
| **API Costs** | $1,000 – $10,000+ per million queries | **💰 $0.00** (100% Free & Self-Hosted) |
| **Data Privacy** | Cloud transmission to external third-party APIs | **🔒 100% On-Premise** (Zero data leaves your VPC) |
| **Audit Trails** | Unpredictable, untracked text output | **✍️ Tamper-evident HMAC-SHA256 Cryptographic Signatures** |
| **Integration** | Proprietary SDKs | **🔌 Drop-in Compatible with OpenAI & Anthropic SDKs** |

---

## 🚀 Key Features

* **🧠 158M Neural Decision Backbone**: Custom hybrid neural architecture combining ModernBERT-base with tabular MLP feature fusion, continuous risk regressors, and cross-attention candidate scorers.
* **🛡️ Zero-Hallucination Policy Gating**: Mathematical $-\infty$ logit injection guarantees that prohibited commands (e.g. `rm -rf`, unauthorized refunds, raw shell pipes) **mathematically receive 0.00% probability**.
* **🔮 Action-Conditioned Predictive Simulation**: Simulates future environment states *before* committing an action, estimating resolution time ($T_{\text{res}}$) and success probability ($P(\text{success})$).
* **🔌 Universal LLM Compatibility Layer**: Native `/v1/chat/completions` (OpenAI & Gemini format) and `/v1/messages` (Claude format) with Server-Sent Events (SSE) streaming.
* **📚 Enterprise Document Intelligence (RAG)**: Built-in local parser and hybrid vector retriever for `.pdf`, `.docx`, `.txt`, `.md`, and `.csv` files with cryptographic citation tracking.
* **💾 Multi-Tier Cognitive Memory**: SQLite-backed episodic history, semantic domain rules, procedural standard operating procedures (SOPs), and real-time world state tracking ($S_{t+1} = T(S_t, A_t)$).
* **👤 Human-in-the-Loop Execution Engine**: Autonomous agent loops automatically pause high-risk actions (e.g. monetary transfers > $500 or production reboots) for managerial sign-off.
* **🖥️ Visual Developer Playground**: Complete Next.js 16 + React 19 visual interface for interactive decision exploration, cognitive tracing, and audit verification.

---

## 🏗️ System Architecture

CDM-1 coordinates a strict **4-Layer Neuro-Symbolic Control Loop**:

```
 ┌─────────────────────────────────────────────────────────────┐
 │ Layer 1: Foundation Model Layer (Optional / Untrusted)      │
 │ • Pluggable: Claude 3.5, GPT-4o, Local Ollama, or Mock      │
 │ • Output: Proposed Action {tool, args, reason}              │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Proposed Action
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Layer 3: Decision & Planning Engine (Deterministic Gate)    │
 │ • Hard Prohibited Rules (Blocks destructive commands)       │
 │ • Dynamic Preconditions & SLA Constraint Verification       │
 │ • Neural Scorer Risk Signal (158M parameters, 3ms latency)  │
 │ • Output: ALLOW | REQUIRE_HUMAN_APPROVAL | BLOCK            │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Allowed / Approved Action
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Layer 4: Execution & Feedback (Sandboxed Tools & Verifiers) │
 │ • Sandboxed Tools (Shell, File I/O, Allowlisted HTTP)       │
 │ • Ground-Truth Outcome Verifier (file exists, tests pass)   │
 │ • Tamper-Evident HMAC-SHA256 Signed Audit Trail             │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Execution & Verification Result
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Layer 2: CDM Cognitive Engine (Memory & Relational State)   │
 │ • Episodic Memory (Records actions, outcomes, rewards)      │
 │ • Relational World State Tracking: S_{t+1} = T(S_t, A_t)    │
 │ • Semantic Rules & Procedural Standard Operating Procedures │
 └─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quickstart (60 Seconds)

### 1. Clone & Install
```bash
git clone https://github.com/cdm-project/cdm-1.git
cd cdm-1

# Install locked production dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Verify System Health (403 Tests)
```bash
pytest
# Expected: 403 passed, 0 failed in CI suite
```

### 3. Launch Developer Playground (One Command)
Starts the FastAPI backend (`http://localhost:8000`) and the Next.js visual dashboard (`http://localhost:3000`):

```bash
# On Linux / macOS
python scripts/start_playground.py

# On Windows
start_playground.bat
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser!

---

## 🔌 Drop-In API Compatibility (OpenAI & Claude)

You can use CDM-1 with any existing AI software simply by changing the `base_url`:

### Use with the Official `openai` Python SDK
```python
from openai import OpenAI

# Point directly to your local CDM-1 engine!
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="cdm_dev_default"  # Handled automatically in local dev
)

completion = client.chat.completions.create(
    model="cdm-1",
    messages=[
        {"role": "system", "content": "You are a reliable operational assistant."},
        {"role": "user", "content": "Our payment gateway is throwing 504 errors under peak load. What is the optimal response?"}
    ],
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Use with the Official `anthropic` Claude Format
```bash
curl -X POST "http://localhost:8000/v1/messages" \
  -H "Authorization: Bearer cdm_dev_default" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cdm-1",
    "messages": [
      {"role": "user", "content": "Evaluate risk for $125,000 wire transfer to new supplier."}
    ]
  }'
```

---

## 🖥️ Developer Playground UI

The included Next.js playground gives operators a no-code visual interface:

1. **Dashboard (`/`)**: Real-time telemetry, 3ms inference latency monitor, and production health probes.
2. **Cognitive Q&A (`/chat`)**: Natural language operational dialogue with evidence citations and uncertainty intervals.
3. **Cognitive Loop (`/cognitive`)**: Real-time visualization of multi-step planning, state transitions, and adaptation.
4. **Decision Explorer (`/decisions`)**: Test tabular JSON features, action candidate rankings, and prohibited constraints.
5. **Memory Store (`/memory`)**:
   * **Episodic**: Past action-reward history.
   * **Rules**: Active business logic and policy constraints.
   * **Procedural**: Multi-step Standard Operating Procedures (SOPs).
   * **Documents**: Drag-and-drop document ingestion (`.pdf`, `.docx`, `.txt`) into vector memory.
6. **Agent Runs (`/agent`)**: Autonomous agent control loops with human-in-the-loop approval buttons and cryptographic HMAC verification.

---

## 📊 Benchmarks & Rigorous Evaluation

CDM-1 is benchmarked against 52 realistic enterprise tasks across Customer Support, Site Reliability Engineering (SRE), and Financial Operations:

* **Neural Aggregate Score**: **95.20%** across 8 cognitive capability dimensions.
* **Safety Gate Precision**: **100.0%** (zero prohibited actions allowed past the gate).
* **Adversarial Robustness**: **100.0%** rejection of destructive command injections and prompt injections.
* **Inference Speed**: **3.2 ms** average latency on standard commodity x86 CPU.

Full benchmark methodology and raw data are published in [`research/benchmarks/COGNITIVE_CAPABILITY_BENCHMARK_REPORT.md`](research/benchmarks/COGNITIVE_CAPABILITY_BENCHMARK_REPORT.md).

---

## 🔒 Security & Cryptographic Provenance

Every decision, tool call, and policy evaluation produced by CDM-1 is hashed and signed using **HMAC-SHA256**:

* **Tamper-Evident Audit Trails**: Logs cannot be modified or retroactively altered without invalidating signatures.
* **Deterministic Sandboxing**: Shell and file tools are strictly jailed with execution timeouts, memory caps, and forbidden command registries.
* **Security Contact**: Report any vulnerabilities responsibly to `security-cdm@proton.me` per our [SECURITY.md](SECURITY.md).

---

## 🤝 Contributing

We welcome community contributions! Please check out:
* [CONTRIBUTING.md](CONTRIBUTING.md) for pull request guidelines and coding conventions.
* [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.
* [LICENSE](LICENSE) — Licensed under the **Apache License, Version 2.0**.

---

<p align="center">
  Built with ❤️ for enterprise reliability and open-source intelligence.
</p>
