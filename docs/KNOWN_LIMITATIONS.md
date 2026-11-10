# CDM-1 Known Operational Limitations

## 1. Scope and Architecture Boundaries

1. **Non-AGI / Specialized Domain Decision Engine**:
   - CDM-1 is a hybrid cognitive decision and control loop system combining neural representation ranking, causal outcome prediction, and deterministic policy gating.
   - It is not an Artificial General Intelligence (AGI) and does not possess biological consciousness, intentionality, or unbounded autonomous agency.
   - Specialized models and decision heads are optimized for operational agent workflows (e.g., triage, customer dispute resolution, infrastructure operations, financial risk bounds).

2. **Dataset Composition Disclosure**:
   - The in-domain training, validation, and test datasets currently consist of **308 records**, of which **300 records (97.4%)** are curated high-fidelity synthetic simulations and **8 records (2.6%)** are externally sourced real production traces (LendingClub 2018, SRE incident postmortems, dispute records).
   - The **8 held-out test scenarios** are derived from institutional guidelines (Emergency Severity Index triage, OFAC/FinCEN compliance) and represent guideline-derived benchmarks rather than live production telemetry.
   - Performance claims on held-out benchmarks reflect adherence to institutional decision trees and synthetic validation distributions.

3. **External LLM Dependency for General Language Reasoning**:
   - CDM-1 uses specialized neural networks (`choice_head`, `outcome_head`, `risk_head` on ModernBERT-base) for decision ranking, outcome prediction, and risk calibration.
   - For unstructured open-domain dialogue or freeform text synthesis, CDM-1 integrates external LLM providers (e.g., Claude 3.5 Sonnet, OpenAI-compatible local vLLM/Ollama endpoints).
   - In offline environments where no external LLM API or local LLM server is accessible, autonomous plan proposal generation relies on pre-compiled tool schemas and structured cognitive templates.

---

## 2. Runtime and System Constraints

1. **Maximum Payload and Context Windows**:
   - The REST API enforces a global request body limit of **1 MB** (HTTP 413 Payload Too Large).
   - Context narrative text is bounded to **8,192 characters**.
   - Tabular structured features are bounded to **100 key-value pairs**.
   - Candidate actions per decision cycle are bounded to **32 options**.

2. **Concurrency and Storage Model**:
   - Storage backends utilize SQLite with Write-Ahead Logging (WAL) enabled (`api_keys.sqlite`, `api_usage_meter.sqlite`, `cognitive_memory.db`, `agent_runs.sqlite`).
   - SQLite provides excellent single-node crash recovery, ACID transactions, and multi-process read concurrency.
   - In distributed multi-region deployments with thousands of concurrent write-heavy nodes, migration to distributed relational backends (e.g., PostgreSQL / CockroachDB) is recommended.

3. **Sandboxed Tool Execution Isolation**:
   - Sandboxed tools (shell, file I/O, network HTTP) run under strict process isolation, path restriction, and timeout limits.
   - Raw terminal interactive commands (e.g., curses, ncurses, interactive shells like `bash -i` or `vim`) are disallowed by design to prevent hanging execution loops.

4. **Offline Model Execution Footprint**:
   - Loading `model.safetensors` (158.36M parameters) into RAM consumes ~1.8 GB memory on CPU. Systems with less than 4 GB free RAM should deploy in Mode A (Software-Only / Weightless) using deterministic rule-based evaluation.

---

## 3. Security and Deployment Recommendations

1. **Default Credentials and Secret Keys**:
   - In production mode (`CDM_ENV=production`), the application strictly refuses to start if default development keys (`dev_secret_key_...`) or unseeded HMAC signing keys are detected.
   - Operators must generate cryptographically strong keys using `scripts/create_api_key.py`.

2. **Network Perimeter**:
   - While CDM-1 provides authenticated bearer token validation, tenant isolation, and durable rate limiting, deploying behind an enterprise reverse proxy or API Gateway (e.g., NGINX, Cloudflare, AWS ALB) with TLS termination is strongly advised for public-facing deployments.
