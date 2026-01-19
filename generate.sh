#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Create timestamped directory BEFORE sbatch
TIMESTAMP=$(date +%Y-%m-%d/%H-%M-%S)
OUTPUT_DIR="${REPO_ROOT}/outputs/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
cp "${REPO_ROOT}/configs/config.yaml" "${OUTPUT_DIR}/config.yaml"

echo "Output directory: $OUTPUT_DIR"

sbatch \
    --output="${OUTPUT_DIR}/slurm.out" \
    --error="${OUTPUT_DIR}/slurm.err" \
    --export=ALL,OUTPUT_DIR="${OUTPUT_DIR}" \
    "${REPO_ROOT}/script.sh"

echo "Job submitted. Logs: $OUTPUT_DIR"
