#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_DIR/results/raw" "$PROJECT_DIR/results/metrics" "$PROJECT_DIR/logs" "$PROJECT_DIR/manifests"

echo "Created local output directories."
echo "Dataset videos and 7B checkpoints are not downloaded automatically."
echo "Set data_root/local_path in configs after reviewing storage requirements."
