# CDM-1 End-to-End Verification Record

## Environment Details
- **Date / Timestamp**: 2026-11-09 15:57:28 +05:30
- **Git Commit Baseline**: `8b1b57b`
- **Branch**: `main`
- **OS Platform**: Microsoft Windows 11 Enterprise (Build 10.0.26100)
- **Python Version**: Python 3.14.0
- **FastAPI / Starlette Runtime**: TestClient ASGI lifespan harness
- **ModernBERT Backbone**: `experiments/release/cdm-1-general` (weights loaded: 134/134 tensors)
- **Docker Verification Status (`scripts/smoke_test.sh --docker`)**:
  - **Status**: **NOT YET RUN** on local host.
  - **Reason**: Local Windows host environment Docker daemon is stopped / offline (`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.`).
  - **Policy**: Per `CLAUDE.md`, results are never simulated or fabricated. Docker verification runs in containerized CI (`.github/workflows/ci.yml`).

---

## 1. Release Manifest Cryptographic Verification

### Command Executed
```powershell
python scripts/check_manifest_integrity.py
```

### Verbatim Output
```text
[*] Verifying Ed25519 signature of release_manifest.json...
    [+] Ed25519 signature is VALID.
    [+] Architecture fields match config.json exactly.
    [+] config.json verified (1518 bytes, SHA-256: 4dcf8cb656a594c5...)
    [+] feature_schema.json verified (2839 bytes, SHA-256: 3e5e8f4e7c58786c...)
    [+] model.safetensors verified (633465536 bytes, SHA-256: 9fab839b607312d4...)
    [+] tokenizer.json verified (3583228 bytes, SHA-256: 6c8aaa9a542084f2...)
    [+] tokenizer_config.json verified (397 bytes, SHA-256: 8d895c33fed683bb...)

[+] Manifest integrity check PASSED (5 files verified, signature verified).
```

---

## 2. In-Process End-to-End Verification Smoke Test

### Command Executed
```powershell
python scripts/smoke_test.py
```

### Verbatim Output
```text
======================================================================
CDM-1 END-TO-END VERIFICATION SMOKE TEST
======================================================================
[transformers] ModernBertModel LOAD REPORT from: answerdotai/ModernBERT-base
Key               | Status     |  | 
------------------+------------+--+-
head.norm.weight  | UNEXPECTED |  | 
decoder.bias      | UNEXPECTED |  | 
head.dense.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED: can be ignored when loading from different task/architecture; not ok if you expect identical arch.
[cdm.api] CDM-1 Model and Cognitive Engine loaded successfully and verified for production inference.

[Step 1] Checking service readiness probe (/v1/ready)...
    [+] /v1/ready status: ready (mode=neural)

[Step 2] Generating Tenant API Key via PersistentApiKeyManager...
    [+] Generated API Key: cdm_develope... (hash=f3b4326dc5278729...)

[Step 3] Submitting operational decision request to POST /v1/decisions...
    [+] Decision ID: dec_fdc240a8031c43e3bd7524dc4ebcac14
    [+] Decision:    YES (conf=0.9937188625335693)
    [+] Score:       88.67 | Risk: 0.0167

[Step 4] Submitting multi-turn agent task to POST /v1/agent/runs...
    [+] Agent Run ID:     run_5b951474df5942baa674f72a341c8722
    [+] Final Status:     loop_detected
    [+] Steps Executed:   4
    [+] Accumulated Cost: $0.0726
    [+] Recorded Steps in Run History: 4
        - Step 1: Action='None' | Gate='allow' | Success=False
        - Step 2: Action='None' | Gate='allow' | Success=False
        - Step 3: Action='None' | Gate='blocked_loop' | Success=False
        - Step 4: Action='None' | Gate='blocked_loop' | Success=False

[Step 5] Cryptographic HMAC Provenance Verification on Audit Trail...
    [+] HMAC Signature Verification: PASSED (All step signatures verified)

[Step 6] Full Provenance Chain:
    --- Step 1 Provenance ---
        Execution ID:   run_5b951474df5942baa674f72a341c8722_step_1
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   ddccc9199a48c07761218e49770cbaa173b270254b58e02c8261266ee3056a24
        HMAC Signature: 4c4261d033c8b1b4e7eb6f4c195c874dd10dd8ca40e54dbfd49c3877311bebc2
        Timestamp:      1794220044.8245058
    --- Step 2 Provenance ---
        Execution ID:   run_5b951474df5942baa674f72a341c8722_step_2
        Tenant ID:      tenant_smoke_test
        Key ID:         cdm-prov-v1
        Model:          CDM-1-Control-Loop (v1.0.0)
        Payload Hash:   2ad3ed719bfad4199d0d777c86207391467c2df55c139a573ebdc149cf64cbbb
        HMAC Signature: 97471efdd4776d9676d3e339b4d36774ddc19612c8d1ab03731cdc9b6fccfab7
        Timestamp:      1794220044.856227

======================================================================
RESULT: PASS — All end-to-end smoke tests and HMAC verifications succeeded.
======================================================================
```
