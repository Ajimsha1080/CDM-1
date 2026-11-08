# Hosting CDM-1 Model Weights on Hugging Face Hub

This guide explains how to host genuine CDM-1 neural model weights on Hugging Face Hub and configure clients to automatically download and verify them.

---

## 1. Architecture & Repository Separation

In accordance with open-source best practices:
- **GitHub Repository**: Contains code, documentation, benchmarks, tests, and configuration files under the **Apache-2.0** license. It contains **no binary model weights**.
- **Hugging Face Hub Repository**: Hosts the neural model weights (`model.safetensors`), configuration, and model card.

---

## 2. Model Repository Structure

A standard CDM-1 Hugging Face model repository should contain:

```text
cdm-1-general/
├── README.md                 # Hugging Face Model Card (copy of docs/MODEL_CARD.md)
├── model.safetensors         # Genuine neural weights (SHA-256 verified)
├── config.json               # Model configuration & dimensions
├── feature_schema.json       # Structured tabular feature schema
├── release_manifest.json     # Signed cryptographic release manifest
├── tokenizer.json            # Pinned ModernBERT tokenizer
└── tokenizer_config.json     # ModernBERT tokenizer config
```

---

## 3. Step-by-Step Publishing Guide

### Step 1: Install and Authenticate Hugging Face CLI
```bash
pip install huggingface_hub
huggingface-cli login
```
Provide your Hugging Face write token when prompted.

### Step 2: Create the Model Repository
```bash
huggingface-cli repo create cdm-project/cdm-1-general --type model
```

### Step 3: Prepare the Release Directory
Ensure your local `experiments/release/cdm-1-general` directory contains all artifacts:
```bash
# Verify the weights file checksum before upload:
python scripts/download_weights.py --verify-only
```
Expected output:
```text
[*] Verifying checksum for: experiments\release\cdm-1-general\model.safetensors
[+] Checksum verified: 9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a
```

### Step 4: Copy Model Card
Copy `docs/MODEL_CARD.md` to `experiments/release/cdm-1-general/README.md`:
```bash
cp docs/MODEL_CARD.md experiments/release/cdm-1-general/README.md
```

### Step 5: Upload Files to Hugging Face
Upload using the Hugging Face Python API or CLI:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="experiments/release/cdm-1-general",
    repo_id="cdm-project/cdm-1-general",
    repo_type="model",
)
```
Or via CLI:
```bash
huggingface-cli upload cdm-project/cdm-1-general experiments/release/cdm-1-general .
```

---

## 4. Downloading and Verifying Weights

Once uploaded, end users and deployment scripts can download and cryptographically verify the weights with a single command:

```bash
python scripts/download_weights.py --hub-repo cdm-project/cdm-1-general
```

The script:
1. Streams the weight file from Hugging Face.
2. Computes the SHA-256 hash in real time.
3. Compares the hash with the expected checksum (`9fab839b...`).
4. Atomically places the verified file into `experiments/release/cdm-1-general/model.safetensors`.
5. Rejects and deletes the download if any byte has been corrupted or tampered with.

If the user does not download weights, the CDM-1 server automatically operates in **deterministic fallback mode** and passes `/v1/ready` readiness checks.
