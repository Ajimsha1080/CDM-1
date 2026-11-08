#!/usr/bin/env bash
# End-to-End Smoke Test Script for CDM-1 (Item 7)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"
python scripts/smoke_test.py "$@"
