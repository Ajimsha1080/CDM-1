# Contributing to CDM-1

Thank you for your interest in contributing to CDM-1! This document provides guidelines for local development, code style, running tests, and submitting pull requests.

---

## 1. Development Setup

### Prerequisites
- Python 3.11+
- Git
- Docker (optional, for containerized integration tests)

### Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ajimsha1080/CDM-1.git
cd CDM-1

# Create and activate a clean virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install locked dependencies and development tools
pip install -r requirements-lock.txt
pip install -e .
```

---

## 2. Running Tests

All changes must pass the full test suite with **0 skips and 0 failures** prior to submitting a pull request:

```bash
# Run the complete test suite
pytest -v

# Run targeted test suites
pytest tests/test_agent_loop_and_api.py
pytest tests/test_gate_and_execution.py
pytest tests/test_llm_provider.py
pytest tests/test_adversarial.py
pytest tests/test_manifest_integrity.py

# Run the 30-task empirical agent evaluation benchmark
python research/agent_eval/eval_runner.py
```

---

## 3. Code Style & Standards

- **Formatting & Linting**: Code should adhere to PEP 8 standards. Use `ruff` or `flake8` for linting.
- **Type Annotations**: All new functions, methods, and classes should include type annotations (`typing` and `pydantic`).
- **No Hallucinated Claims**: Never write arbitrary pass/fail claims or synthetic benchmark numbers into reports or documentation. All reported metrics must originate from reproducible script executions.
- **Fail-Safe Sandboxing**: Tools must catch exceptions internally and return structured `ToolResult` envelopes rather than raising uncaught exceptions into the agent loop.
- **Deterministic Gates**: Policy gating must always execute deterministic rules first before consulting neural models.

---

## 4. Pull Request Process

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make Small, Focused Commits**: Write clear, imperative commit messages (`feat: add tool X`, `fix: handle edge case Y in policy gate`).
3. **Verify Integrity**: Ensure all 149+ unit tests pass locally.
4. **Open Pull Request**: Fill out the provided PR template completely. PRs will automatically run GitHub Actions CI (tests, secret scanning, license check, and Docker build).
