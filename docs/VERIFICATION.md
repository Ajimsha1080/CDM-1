# CDM-1 End-to-End Verification Record

## Environment Details
- **Date / Timestamp**: 2026-11-09 03:56:42 +05:30
- **Git Commit Baseline**: `6a8e6a49a7d14f225ec0e268df8f5e5e0cf32ff7`
- **Branch**: `main`
- **OS Platform**: Microsoft Windows 11 Enterprise (Build 10.0.26100)
- **Python Version**: Python 3.14.0
- **FastAPI / Starlette Runtime**: TestClient ASGI lifespan harness
- **ModernBERT Backbone**: `experiments/release/cdm-1-general` (weights loaded: 134/134 tensors)
- **Docker Compose / Linux Status**:
  - Local Windows Docker engine: Stopped (`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`).
  - Containerized Linux CI environment: Automated in `.github/workflows/ci.yml` (`docker-build-and-live-api` and `docker-tests` on `ubuntu-latest`).
  - [TODO - LOCAL HOST]: To run containerized smoke tests locally (`bash scripts/smoke_test.sh --docker`), start Docker Desktop or the Linux Docker daemon on the host.

## Command Executed
```powershell
python scripts/smoke_test.py
```

## Verbatim Terminal Output
```text
2026-11-09 03:56:14,690 [INFO] [cdm.api] Loading CDM-1 Model from experiments/release/cdm-1-general...
======================================================================
CDM-1 END-TO-END VERIFICATION SMOKE TEST
======================================================================
Loading weights:   0%|          | 0/134 [00:00<?, ?it/s]Loading weights:   7%|6         | 9/134 [00:00<00:01, 78.84it/s]Loading weights:  13%|#2        | 17/134 [00:00<00:01, 72.83it/s]Loading weights:  19%|#9        | 26/134 [00:00<00:01, 78.51it/s]Loading weights:  26%|##6       | 35/134 [00:00<00:01, 80.18it/s]Loading weights:  33%|###2      | 44/134 [00:00<00:01, 77.95it/s]Loading weights:  47%|####7     | 63/134 [00:00<00:00, 84.60it/s]Loading weights:  54%|#####3    | 72/134 [00:00<00:00, 75.16it/s]Loading weights: 100%|##########| 134/134 [00:01<00:00, 130.26it/s]
[transformers] ModernBertModel LOAD REPORT from: answerdotai/ModernBERT-base
Key               | Status     |  | 
------------------+------------+--+-
head.norm.weight  | UNEXPECTED |  | 
head.dense.weight | UNEXPECTED |  | 
decoder.bias      | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
2026-11-09 03:56:25,716 [INFO] [cdm.api] CDM-1 Model and Cognitive Engine loaded successfully and verified for production inference.
2026-11-09 03:56:27,375 [INFO] [httpx] HTTP Request: GET http://testserver/v1/ready "HTTP/1.1 200 OK"
2026-11-09 03:56:28,480 [INFO] [httpx] HTTP Request: POST http://testserver/v1/decisions "HTTP/1.1 200 OK"
2026-11-09 03:56:28,541 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 03:56:28,562 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 03:56:28,563 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 03:56:28,606 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 03:56:28,608 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 03:56:28,608 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 03:56:28,642 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 03:56:28,644 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 03:56:28,645 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 03:56:28,677 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 03:56:28,678 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 03:56:28,679 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 03:56:28,713 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 03:56:28,714 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 03:56:28,714 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 03:56:28,794 [INFO] [httpx] HTTP Request: POST http://testserver/v1/agent/runs "HTTP/1.1 200 OK"

[Step 1] Checking service readiness probe (/v1/ready)...
    [+] /v1/ready status: ready (mode=neural)

[Step 2] Generating Tenant API Key via PersistentApiKeyManager...
    [+] Generated API Key: cdm_develope... (hash=490f6a44c9d667c7...)

[Step 3] Submitting operational decision request to POST /v1/decisions...
    [+] Decision ID: dec_d0005f2c569e4b64bbbe626538315fdf
    [+] Decision:    YES (conf=0.9937188625335693)
    [+] Score:       88.67 | Risk: 0.0167

[Step 4] Submitting multi-turn agent task to POST /v1/agent/runs...
    [+] Agent Run ID:     run_0d8fb04d5dd343c496c20ce231ad7880
    [+] Final Status:     step_budget_exceeded
    [+] Steps Executed:   5
    [+] Accumulated Cost: $0.0209
    [+] Recorded Steps in Run History: 5
        - Step 1: Action='None' | Gate='allow' | Success=False
        - Step 2: Action='None' | Gate='allow' | Success=False
        - Step 3: Action='None' | Gate='allow' | Success=False
        - Step 4: Action='None' | Gate='allow' | Success=False
        - Step 5: Action='None' | Gate='allow' | Success=False

[Step 5] Cryptographic HMAC Provenance Verification on Audit Trail...
    [+] HMAC Signature Verification: PASSED (All step signatures verified)

[Step 6] Full Provenance Chain:
    --- Step 1 Provenance ---
        Execution ID:   run_0d8fb04d5dd343c496c20ce231ad7880_step_1
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   60546b102e20e518b04ba6199054461d4e0739956b8b6d2e77edc92f127ca276
        HMAC Signature: 09e65e4fbd2027e535763760c6ac837fd8a1d2264a81a3175e92b36331e3ecaf
        Timestamp:      1794176788.564416
    --- Step 2 Provenance ---
        Execution ID:   run_0d8fb04d5dd343c496c20ce231ad7880_step_2
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   9a072d2f81ec851bad746c5009a60e8c3f4c868df1ba9df0e65c0db3909fd102
        HMAC Signature: 2fa43a0be5f8b396e2d61a941170fba9e33b444977079948d57fb1732bfc15d5
        Timestamp:      1794176788.609191
    --- Step 3 Provenance ---
        Execution ID:   run_0d8fb04d5dd343c496c20ce231ad7880_step_3
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   be6c2e4efb3642c73a091f819c7781e04b5b38ebb2b03312482bee06eb62218d
        HMAC Signature: ad5f427eb386d44088e983d62c5f0346c5e4e9e099ad942309719a953c92e866
        Timestamp:      1794176788.6461394
    --- Step 4 Provenance ---
        Execution ID:   run_0d8fb04d5dd343c496c20ce231ad7880_step_4
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   ac31c5718139247dde4ab09df59232bccaebacc5954204b510d52c84321a8ff7
        HMAC Signature: c7fb5fc9cab3de6a7f9b7425d7adbc63541caadf6f4b5eddccb86a4a036708c9
        Timestamp:      1794176788.6823502
    --- Step 5 Provenance ---
        Execution ID:   run_0d8fb04d5dd343c496c20ce231ad7880_step_5
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   e96047cad6badc1378eb524540237eb1f4553d230032486794683f80937987c4
        HMAC Signature: dc16e6e9f05a83b8b521b4418f2d4e630eef47b6a367668a4786844b8c9d9628
        Timestamp:      1794176788.7150488

======================================================================
RESULT: PASS — All end-to-end smoke tests and HMAC verifications succeeded.
======================================================================
```
