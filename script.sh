#!/bin/bash
#SBATCH --job-name=generate_ineq_problems
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=40:00:00
# (we keep default --export=ALL so your outer conda stays active)

set -euo pipefail

###############################################################################
# Validate OUTPUT_DIR is set (should be passed from generate.sh)
###############################################################################
if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "ERROR: OUTPUT_DIR not set. Run via generate.sh"
    exit 1
fi

###############################################################################
# Determine repo root robustly in sbatch:
# SLURM runs a *copied* script from /var/... so BASH_SOURCE points there.
# Use SLURM_SUBMIT_DIR (where you ran sbatch from) as the anchor.
###############################################################################
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# If submit dir is inside a git repo, normalize to the git top-level
if git -C "$REPO_ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$REPO_ROOT" rev-parse --show-toplevel)"
fi

cd "$REPO_ROOT"

###############################################################################
# Activate conda environment
###############################################################################
source /mnt/lustre/work/luxburg/luj210/miniconda3/etc/profile.d/conda.sh
conda activate ineq

###############################################################################
# --- Diagnostics ---
###############################################################################
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "REPO_ROOT=$REPO_ROOT"
pwd
nvidia-smi || true

###############################################################################
# --- Ensure writable TMP for this job ---
# Use a *short* path to avoid libzmq IPC length limits (<107 chars)
###############################################################################
SHORT_TMP_ROOT="/tmp/pth-${USER:-unknown}"
SHORT_TMPDIR="${SHORT_TMP_ROOT}/datagen-${SLURM_JOB_ID:-local}"

mkdir -p "$SHORT_TMPDIR"
export TMPDIR="$SHORT_TMPDIR"
export XDG_RUNTIME_DIR="$TMPDIR"
export CUDA_CACHE_PATH="$TMPDIR/cuda_cache"
mkdir -p "$CUDA_CACHE_PATH"

###############################################################################
# --- Force caches to our TMP (avoid stale /scratch_local paths) ---
###############################################################################
unset TORCHINDUCTOR_CACHE_DIR TORCHINDUCTOR_REMOTE_CACHE PYTORCH_TUNING_CACHE_DIR \
      TRITON_CACHE_DIR XDG_CACHE_HOME

export XDG_CACHE_HOME="$TMPDIR/.cache"; mkdir -p "$XDG_CACHE_HOME"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torch/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_TORCH_COMPILE_CACHE_DIR="$XDG_CACHE_HOME/vllm/torch_compile_cache"
export TORCHINDUCTOR_USE_REMOTE_CACHE=0

###############################################################################
# --- Run data generation ---
###############################################################################
python generate.py "hydra.run.dir=${OUTPUT_DIR}"

echo "Data generation complete."