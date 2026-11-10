# CDM-1 Model Release Security and Verification Specification

This document defines the cryptographic security architecture, release signing process, key management lifecycle, and verification enforcement for the CDM-1 Cognitive AI Engine.

---

## 1. Cryptographic Architecture

CDM-1 utilizes asymmetric digital signatures based on the **Ed25519 (PureEdDSA)** standard (RFC 8032) to ensure artifact authenticity and integrity:
- **Algorithm**: Ed25519 (Curve25519, SHA-512)
- **Manifest Canonicalization**: RFC 8785 JSON Canonicalization Scheme (JCS) / strict key-sorted UTF-8 canonical encoding with comma/colon separators.
- **Signed Payload**: Canonical digest of release version, model configuration, tensor counts, parameter counts, file sizes, and SHA-256 digests of all distributed artifacts.
- **Verification Boundary**: The trusted public verification key is embedded within `cdm/security/manifest_verification.py`. The private signing key is strictly isolated and never distributed within application bundles or repositories.

---

## 2. Private Key Isolation & Storage

The Ed25519 private key is the root of trust for all distributed weights and manifests.

1. **Storage Requirements**:
   - Private keys MUST be stored in Hardware Security Modules (HSMs) or secure cloud KMS (e.g., AWS KMS, GCP Cloud KMS, HashiCorp Vault) configured with export restrictions.
   - For air-gapped release pipelines, private keys must reside on FIPS 140-2 Level 3 certified hardware tokens.
   - Private keys are NEVER committed to version control, embedded in source files, included in container images, or exposed via environment variables in production runtimes.

2. **Signing Process**:
   - The release packager (`scripts/package_release.py`) signs `release_manifest.json` during the release build step.
   - In automated CI/CD pipelines, the signing action requires dual-control authorization and access to the dedicated signing secret (`CDM_RELEASE_SIGNING_KEY`).
   - The private key signs the canonical JSON representation of the manifest without the `signature` field, and the resulting signature is appended as `signature.ed25519_hex`.

---

## 3. Public Key Distribution & Embedding

1. **Embedded Public Key**:
   - The verified public key (`TRUSTED_RELEASE_PUBLIC_KEY`) is hardcoded in `cdm/security/manifest_verification.py`:
     ```python
     TRUSTED_RELEASE_PUBLIC_KEY = "4784afcf98a872ec65cc951f8803c6619e777923b559196f4cd9297bbadf56a5"
     ```
2. **Distribution Channels**:
   - The trusted public key is published across out-of-band communication channels, including official release notes and GitHub release tags.
   - Applications verify artifacts against this public key prior to checkpoint loading and initialization.

---

## 4. Key Rotation & Revocation Policy

1. **Scheduled Rotation**:
   - Release signing keys are scheduled for rotation annually (every 12 months).
   - A multi-key grace period allows `cdm/security/manifest_verification.py` to maintain a set of trusted active keys (`ACTIVE_PUBLIC_KEYS`) alongside the primary release key.

2. **Emergency Key Revocation**:
   - If a private key compromise is suspected or detected:
     1. The compromised key is immediately added to the hardcoded `REVOKED_KEYS_SET` in `cdm/security/manifest_verification.py`.
     2. A patched release is issued with an incremented security version and a newly generated keypair.
     3. Existing deployments reject manifests signed by revoked keys with a `SecurityVerificationError("Signing key has been revoked.")`.
     4. Cloud artifact repositories revoke download tokens for manifests signed with the compromised key.

---

## 5. Runtime Verification Enforcement

Every CDM-1 node enforces strict validation at multiple stages:

1. **Startup Check (`/v1/ready`)**:
   - Verifies that `experiments/release/cdm-1-general/model.safetensors` exists.
   - Computes the live SHA-256 byte digest and matches it against the trusted manifest.
   - Cryptographically verifies the Ed25519 signature of `release_manifest.json` against `TRUSTED_RELEASE_PUBLIC_KEY`.
   - Returns HTTP 503 if signature verification fails, files are missing, or hashes diverge.

2. **Offline Inference Verification**:
   - SafeTensors format prevents arbitrary pickle execution during deserialization.
   - The exact SHA-256 hash of the loaded weights (`9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a`) is bound into runtime inference provenance tokens.

3. **Prediction Provenance Binding**:
   - Inferences produce cryptographically signed provenance tokens binding model name, version, artifact hash, execution ID, tenant ID, task, and output digest.
   - Caller-modified outcomes or untrusted inputs are rejected by the planning engine.
