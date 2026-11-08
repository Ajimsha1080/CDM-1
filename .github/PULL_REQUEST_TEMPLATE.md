## Description
Provide a brief summary of the changes in this pull request and the problem they resolve.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Architectural change / refactor
- [ ] Documentation update
- [ ] Security fix / hardening

## Affected Components
- [ ] Layer 1: LLM Provider (`cdm/llm/`)
- [ ] Layer 2: Cognitive Engine (`cdm/context/`, `cdm/memory/`)
- [ ] Layer 3: Policy Gate & Planning (`cdm/gate/`, `cdm/decision/`)
- [ ] Layer 4: Sandboxed Tools & Verifiers (`cdm/execution/`)
- [ ] Agent Control Loop & API (`cdm/agent/`, `api/`)
- [ ] Research & Evaluation (`research/`)

## Checklist
- [ ] I have read the [CONTRIBUTING.md](CONTRIBUTING.md) guide.
- [ ] My code adheres to PEP 8 standards with types and docstrings.
- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] All existing and new tests pass locally (`pytest`).
- [ ] No hardcoded secrets, private paths, or synthetic claims were added.
- [ ] No weight binaries (`.safetensors`, `.bin`, `.pt`) are tracked or committed.
- [ ] Any new dependency is Apache-2.0 compatible and documented in `NOTICE`.
