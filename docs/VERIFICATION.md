# CDM-1 End-to-End Verification Record

## Environment Details
- **Date / Timestamp**: 2026-11-09 03:00:11 +05:30
- **Git Commit Baseline**: `f8b9c9709ee2e1847afccf5802b1d1aa55ba30e8`
- **Branch**: `main`
- **OS Platform**: Microsoft Windows 11 Enterprise (10.0.26100)
- **Python Version**: Python 3.14.0
- **FastAPI / Starlette Runtime**: TestClient ASGI lifespan harness
- **ModernBERT Backbone**: `experiments/release/cdm-1-general` (weights loaded: 134/134 tensors)

## Command Executed
```powershell
python scripts/smoke_test.py
```

## Terminal Output (Verbatim)
```text
2026-11-09 02:59:48,233 [INFO] [cdm.api] Loading CDM-1 Model from experiments/release/cdm-1-general...
======================================================================
CDM-1 END-TO-END VERIFICATION SMOKE TEST
======================================================================
Loading weights:   0%|          | 0/134 [00:00<?, ?it/s]Loading weights:   7%|7         | 10/134 [00:00<00:01, 98.10it/s]Loading weights:  15%|#4        | 20/134 [00:00<00:01, 91.67it/s]Loading weights:  25%|##5       | 34/134 [00:00<00:00, 107.80it/s]Loading weights:  40%|####      | 54/134 [00:00<00:00, 141.98it/s]Loading weights:  51%|#####1    | 69/134 [00:00<00:00, 126.86it/s]Loading weights:  62%|######1   | 83/134 [00:00<00:00, 106.02it/s]Loading weights: 100%|##########| 134/134 [00:00<00:00, 175.05it/s]
[transformers] ModernBertModel LOAD REPORT from: answerdotai/ModernBERT-base
Key               | Status     |  | 
------------------+------------+--+-
head.dense.weight | UNEXPECTED |  | 
decoder.bias      | UNEXPECTED |  | 
head.norm.weight  | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
2026-11-09 02:59:54,329 [INFO] [cdm.api] CDM-1 Model and Cognitive Engine loaded successfully and verified for production inference.
2026-11-09 02:59:55,133 [INFO] [httpx] HTTP Request: GET http://testserver/v1/ready "HTTP/1.1 200 OK"
2026-11-09 02:59:55,868 [INFO] [httpx] HTTP Request: POST http://testserver/v1/decisions "HTTP/1.1 200 OK"
2026-11-09 02:59:55,896 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 02:59:55,904 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 02:59:55,905 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 02:59:55,920 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 02:59:55,922 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 02:59:55,922 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 02:59:55,937 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 02:59:55,938 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 02:59:55,938 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 02:59:55,953 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 02:59:55,954 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 02:59:55,955 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 02:59:55,969 [WARNING] [cdm.llm.validator] Initial LLM output failed validation (Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'check_status'...lth of payment system.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing). Attempting single JSON repair...
2026-11-09 02:59:55,970 [ERROR] [cdm.llm.validator] JSON repair failed second check: Schema validation error: 2 validation errors for AgentActionProposal
tool
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
reason
  Field required [type=missing, input_value={'action': 'view_ticket',...ng validation failure.'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
2026-11-09 02:59:55,970 [ERROR] [cdm.execution.tools] Error executing tool 'shell_exec': Shell execution tool is disabled by default. Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true).
2026-11-09 02:59:56,030 [INFO] [httpx] HTTP Request: POST http://testserver/v1/agent/runs "HTTP/1.1 200 OK"

[Step 1] Checking service readiness probe (/v1/ready)...
    [+] /v1/ready status: ready (mode=neural)

[Step 2] Generating Tenant API Key via PersistentApiKeyManager...
    [+] Generated API Key: cdm_develope... (hash=2646f71e92716829...)

[Step 3] Submitting operational decision request to POST /v1/decisions...
    [+] Decision ID: dec_da77a7fcd92444acaad92bd08a8c85b9
    [+] Decision:    YES (conf=0.9937188625335693)
    [+] Score:       88.67 | Risk: 0.0167

[Step 4] Submitting multi-turn agent task to POST /v1/agent/runs...
    [+] Agent Run ID:     run_ecb919bb0d334fb18afa3f44a238b44e
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
        Execution ID:   run_ecb919bb0d334fb18afa3f44a238b44e_step_1
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   b0ca5c9feab7bb3f9c947c9a982ba181e2ae90f09610f13983275d288b963ecb
        HMAC Signature: 45af12d4f530ec9615bc89a827dde2af0e9bae3832a2234814ded8374558b608
        Timestamp:      1794173395.9058695
    --- Step 2 Provenance ---
        Execution ID:   run_ecb919bb0d334fb18afa3f44a238b44e_step_2
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   5b2e0c7c63711f9341535b4a5dcce46b4acdcddef5b18fa8cc371a7f8084177f
        HMAC Signature: 2d8842c3dc665d9233312b2e511765c79fcb0566e7f23e60a7e2fd1e47cdd191
        Timestamp:      1794173395.922576
    --- Step 3 Provenance ---
        Execution ID:   run_ecb919bb0d334fb18afa3f44a238b44e_step_3
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   a17adb8075367e2b182869e9f3167a34af4a89682897d6da75c73267c87bb178
        HMAC Signature: e30918b74e87516881e3108d28189ad078fefb11819d53544ec8be3bf01066e0
        Timestamp:      1794173395.9391668
    --- Step 4 Provenance ---
        Execution ID:   run_ecb919bb0d334fb18afa3f44a238b44e_step_4
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   8ed56855fec728025370e920896a51eb4c477a3de6a60b132f962ccaca5e07b2
        HMAC Signature: d3a9ae4d32318198f266a2ddf799d9cadb9f17aa9d28c825e29d22b11ef3a9d7
        Timestamp:      1794173395.9547722
    --- Step 5 Provenance ---
        Execution ID:   run_ecb919bb0d334fb18afa3f44a238b44e_step_5
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   defd8102128f373060bdd5b8eb897c55b6e364b34835be6fa7098d0528b0075e
        HMAC Signature: 7509c4a5a1563582251562bd2b089a84e354ff69398b11b6a5b8c50fbc679068
        Timestamp:      1794173395.970374

======================================================================
RESULT: PASS — All end-to-end smoke tests and HMAC verifications succeeded.
======================================================================
```
