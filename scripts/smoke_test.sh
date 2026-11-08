#!/usr/bin/env bash
# End-to-End Smoke Test Script for CDM-1
# Supports direct ASGI in-process execution or --docker compose mode.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [ "$1" == "--docker" ]; then
    echo "[*] Starting CDM-1 service via Docker Compose..."
    docker compose up -d --build
    trap "docker compose down" EXIT
    echo "[*] Waiting for container readiness (/v1/ready)..."
    python -c '
import urllib.request, time, sys
for i in range(30):
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/ready", timeout=2) as resp:
            if resp.status == 200:
                print("[+] Containerized service is READY.")
                sys.exit(0)
    except Exception:
        time.sleep(1)
print("[-] Timeout waiting for container readiness", file=sys.stderr)
sys.exit(1)
'
    CDM_API_URL="http://localhost:8000" python scripts/smoke_test.py
else
    python scripts/smoke_test.py "$@"
fi
