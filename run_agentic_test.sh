#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo " Agentic Graph Query System — Batch Test"
echo "============================================"
echo

cd "$(dirname "$0")"

echo "[1] Checking Python..."
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

echo "[2] Running batch tests..."
echo
python3 test_agentic_batch.py
echo
echo "Done."
