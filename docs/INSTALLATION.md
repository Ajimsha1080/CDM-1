# CDM-1 Dependency and Installation Guide

## 1. System Requirements

### Hardware Requirements
- **CPU**: x86_64 or ARM64 architecture with AVX2 instruction support. Minimum 4 physical cores recommended.
- **RAM**: Minimum 8 GB system RAM (16 GB recommended for full neural inference with ModernBERT-base backbone).
- **Disk Space**:
  - Software-only installation: ~500 MB.
  - Full model distribution (including `model.safetensors` and cache): ~5 GB free space.
- **GPU (Optional)**: NVIDIA GPU with CUDA 11.8+ or 12.x for accelerated neural inference; full CPU fallback is natively supported.

### Software Environment
- **Operating System**: Linux (Ubuntu 22.04+ / Debian 12+), macOS (13+ Ventura / 14+ Sonoma), or Windows 10/11 (PowerShell 7+ / WSL2).
- **Python Version**: Python 3.10 to Python 3.14 (Verified on Python 3.14.0).
- **C Compiler**: Standard C build essentials (optional, only for building specific C extensions).

---

## 2. Installation Procedures

### Step 1: Clone Repository & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Ajimsha1080/CDM-1.git
cd CDM-1

# Create an isolated Python virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
```

### Step 2: Install Core Dependencies

#### Option A: Exact Pinned Dependencies (Recommended for Production)
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements-lock.txt
```

#### Option B: Standard Dependency Ranges (Recommended for Development)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Environment Consistency
Confirm all package requirements are satisfied without conflicts:
```bash
python -m pip check
# Expected output: No broken requirements found.

python -m compileall -q .
# Expected output: (silent, exit code 0)
```

---

## 3. Neural Checkpoint Setup

CDM-1 operates in two verifiable deployment modes:

### Mode A: Software-Only Mode (Weightless)
In this mode, CDM-1 runs using deterministic rule-based heuristic scoring and risk assessment. No weights download is required.
Verify integrity:
```bash
python scripts/check_manifest_integrity.py --mode software_only
```

### Mode B: Full Neural Model Mode
For deep cognitive candidate ranking, causal outcome prediction, and calibrated risk estimation:
1. Ensure `model.safetensors` (633,465,536 bytes) is placed in:
   ```
   experiments/release/cdm-1-general/model.safetensors
   ```
2. Verify release manifest integrity, SHA-256 digests, and Ed25519 digital signatures:
   ```bash
   python scripts/check_manifest_integrity.py --mode full_model
   ```

---

## 4. Verification and Automated Testing

Execute the test suite to verify full functionality:

```bash
# Verify test collection
pytest --collect-only -q

# Execute standard test suite (zero skips required)
pytest -rs --strict-markers -m "not requires_docker and not requires_weights" --junitxml=test-results.xml

# Enforce zero-skips gate
python scripts/check_no_skips.py test-results.xml

# Run live neural inference tests (requires model.safetensors)
pytest -m requires_weights
```

---

## 5. Starting the Service

### Development Mode
```bash
export CDM_ENV=development
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode
```bash
export CDM_ENV=production
export CDM_MODEL_PATH=experiments/release/cdm-1-general
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Verify operational health:
```bash
# Liveness probe
curl -s http://localhost:8000/v1/health

# Readiness probe
curl -s http://localhost:8000/v1/ready
```
