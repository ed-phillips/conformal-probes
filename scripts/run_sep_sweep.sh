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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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

# 1) Generate answers (no compute_uncertainties inside generate_answers)
python scripts/run_sep_sweep.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}" \
  --no-compute-uncertainties

# 2) Compute semantic entropy locally (offline)
python scripts/compute_semantic_entropy_local.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}"

# 3) Train probes (replaces notebook)
python scripts/train_probes.py \
  --config "${CFG_YAML}" \
  --runs-root "${RUNS_ROOT}" \
  --out "${RUNS_ROOT}/probes.pkl"

# Sync back to NFS
OUT_ROOT="${REPO_ROOT}/outputs/runs/${SLURM_JOB_ID}"
mkdir -p "${OUT_ROOT}"
rsync -avh --no-g --no-p "${RUNS_ROOT}/" "${OUT_ROOT}/"

echo "All done. Results synced to: ${OUT_ROOT}"
