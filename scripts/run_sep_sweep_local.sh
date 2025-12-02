#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located (e.g., .../conformal-probes/scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set REPO_ROOT to the parent directory (e.g., .../conformal-probes)
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Define the config path relative to the root
CFG_YAML="${REPO_ROOT}/config/sep_sweep_local.yaml"

# Move to the root directory
cd "${REPO_ROOT}"

echo "Working directory set to: $(pwd)"
echo "Config file located at: ${CFG_YAML}"

# Load .env file if it exists
if [ -f "${REPO_ROOT}/.env" ]; then
  # 'set -a' causes variables defined from now on to be automatically exported
  set -a
  source "${REPO_ROOT}/.env"
  set +a
  echo "Loaded variables from .env"
fi

# Activate the uv venv
source .venv/bin/activate
# Ensure repo root is in PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Hugging Face caches (local directories, no HPC paths)
export HF_HOME="${REPO_ROOT}/hf_home"
export HF_HUB_CACHE="${REPO_ROOT}/hf_home"
export HF_DATASETS_CACHE="${REPO_ROOT}/hf_datasets_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Don’t talk to wandb servers
export WANDB_MODE=offline

# Local runs root
RUNS_ROOT="${REPO_ROOT}/runs_local"
mkdir -p "${RUNS_ROOT}"

echo "Python: $(which python)"; python -V
echo "RUNS_ROOT: ${RUNS_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

# 1) Generate answers (don’t compute uncertainties in generate_answers)
python scripts/run_sep_sweep.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}" \
  --no-compute-uncertainties

# 2) Compute semantic entropy (DeBERTa, etc.)
python scripts/compute_semantic_entropy_local.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}"

# 3) Train probes
python scripts/train_probes.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}" 

echo "Done. Inspect runs in ${RUNS_ROOT}"
