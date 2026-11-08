---
name: Feature Request
about: Suggest an idea or architectural improvement for CDM-1
title: '[FEATURE] '
labels: ['enhancement', 'discussion']
assignees: ''
---

## Feature Summary
A clear and concise description of the proposed feature or improvement.

## Problem Statement
What problem does this solve? (e.g., "When running multi-step tool calls, the verifier cannot...")

## Proposed Architecture / Solution
Which layer does this affect?
- [ ] Layer 1: Foundation Model & LLM Provider (`cdm/llm/`)
- [ ] Layer 2: Cognitive Engine & World State (`cdm/context/`, `cdm/memory/`)
- [ ] Layer 3: Decision, Planning & Policy Gate (`cdm/gate/`, `cdm/decision/`)
- [ ] Layer 4: Execution Tools & Outcome Verification (`cdm/execution/`)
- [ ] Agent Control Loop & API (`cdm/agent/`, `api/`)

Describe the proposed mechanism, API, or data schema.

## Alternatives Considered
A clear and concise description of any alternative solutions or workarounds you've considered.

## Security & Provenance Considerations
How does this change affect deterministic policy enforcement, tamper-evident audit logs, and tool sandboxing?
