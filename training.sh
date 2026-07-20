#!/usr/bin/env bash

# Portable launcher for one training run. Activate the repository environment
# before calling this script, then pass any training.py options through it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" training.py "$@"
