#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=gpu_long
#SBATCH --time=60:00:00
#SBATCH --job-name=sep_sweep
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/well/clifton/users/gar957/conformal-probes/slurm_logs/%x-%j.out
#SBATCH --error=/well/clifton/users/gar957/conformal-probes/slurm_logs/%x-%j.err
#SBATCH --signal=B:TERM@600
#SBATCH --nodelist=compg[028-042]

set -euo pipefail

REPO_ROOT="/well/clifton/users/gar957/conformal-probes"
CFG_YAML="${REPO_ROOT}/config/sep_sweep.yaml"

module load Anaconda3
eval "$(conda shell.bash hook)"

# Temporarily disable 'unbound variable' check because conda scripts are messy
set +u
conda activate "${REPO_ROOT}/conda_envs/se_probes"
set -u

# Add the repository root to PYTHONPATH so scripts can import 'semantic_uncertainty'
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Offline HF
export HF_HOME="${REPO_ROOT}/hf_home"
export HF_HUB_CACHE="${REPO_ROOT}/hf_home"
export HF_DATASETS_CACHE="${REPO_ROOT}/hf_datasets_cache"
export HF_HUB_OFFLINE=1

# We don't want wandb network calls
export WANDB_MODE=offline

# Job scratch root
JOB_ROOT="${TMPDIR}/sep_job_${SLURM_JOB_ID}"
RUNS_ROOT="${JOB_ROOT}/runs"
mkdir -p "${RUNS_ROOT}"

cd "${REPO_ROOT}"
echo "Python: $(which python)"; python -V
nvidia-smi || true
echo "Job scratch root: ${JOB_ROOT}"
echo "RUNS_ROOT: ${RUNS_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

# --- Build Arguments ---
# Base arguments for the generation script
GEN_ARGS=(
  --config "${CFG_YAML}"
  --runs-root "${RUNS_ROOT}"
  --no-compute-uncertainties
)

# Optional: If TARGET_MODEL is set (via export), append it.
# The ':-' ensures it doesn't crash on 'set -u' if unset.
if [[ -n "${TARGET_MODEL:-}" ]]; then
    echo ">>> Target Set: Processing Single Model: ${TARGET_MODEL}"
    GEN_ARGS+=(--model "${TARGET_MODEL}")
else
    echo ">>> Target Unset: Processing ALL Models from Config"
fi

# 1) Generate answers
# If TARGET_MODEL is set, this only generates for that model.
python scripts/run_sep_sweep.py "${GEN_ARGS[@]}"

# 2) Compute semantic entropy
# This script iterates the config list, but skips directories that don't exist.
# So it naturally handles single-model runs without needing extra flags.
python scripts/compute_semantic_entropy_local.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}"

# 3) Train probes
# Same here: it processes whatever valid data it finds in RUNS_ROOT.
python scripts/train_probes.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}" 

# Sync results
OUT_ROOT="${REPO_ROOT}/outputs/runs/${SLURM_JOB_ID}"
mkdir -p "${OUT_ROOT}"
rsync -avh --no-g --no-p "${RUNS_ROOT}/" "${OUT_ROOT}/"

echo "Done."