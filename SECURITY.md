# Security Policy

## Supported Versions

Only the latest release on the `main` branch receives active security updates and vulnerability patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

We take the security of CDM-1 and its control loop architecture seriously. If you discover or suspect a security vulnerability, **please do not open a public issue or discuss it in public pull requests**.

### Private Reporting Process

Please report all security vulnerabilities privately by emailing:
**`security-cdm@proton.me`** (or use GitHub Private Vulnerability Reporting via the Security tab of the repository).

In your report, please include:
1. **Description**: Clear description of the vulnerability and potential impact.
2. **Reproduction Steps**: Step-by-step instructions or minimal reproducible script.
3. **Affected Components**: Specific files, endpoints (e.g. `/v1/agent/runs`, `/v1/ready`), or tools.
4. **Environment**: Python version, OS platform, and whether running standalone or containerized.
5. **Mitigation**: Any proposed fix or workaround if known.

### Response Timelines

Our maintainer team adheres to responsible disclosure practices and the following response timelines:

| Stage | Response Target |
|---|---|
| **Initial Acknowledgment** | Within **48 hours** of report receipt |
| **Vulnerability Triage & Severity Rating** | Within **5 business days** |
| **Status Updates** | Every **7 days** until resolved |
| **Patch Release & Security Advisory** | Within **30 calendar days** of confirmation |

### Scope & Policy Enforcement

CDM-1 is designed as an agent control loop with policy gating:
- Circumvention of the **Policy Gate** (`cdm/gate/policy_gate.py`) or sandboxing bounds is treated as a high-severity vulnerability.
- Unauthorized access across tenant boundaries (tenant isolation leakage) is treated as a critical severity vulnerability.
- Forgery of signed HMAC provenance records without the secret key is treated as a critical security issue.
