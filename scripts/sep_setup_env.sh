#!/usr/bin/env bash
# Run ONCE on a LOGIN node with internet access.
# Sets up conda env and pre-downloads models/datasets into NFS caches.

set -euo pipefail

# ---------- Config ----------
PROJECT_NAME="conformal-probes"
PYTHON_MAJOR_MINOR="3.11"
CONDA_ENV_NAME="sep_env"

DATA_ROOT="${DATA:-/well/clifton/users/gar957}"
PROJECT_DIR="${DATA_ROOT}/${PROJECT_NAME}"

CONDA_ENVS_DIR="${PROJECT_DIR}/conda_envs"
TARGET_CONDA_ENV="${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}"
CONDA_PKGS_CACHE="${PROJECT_DIR}/conda_pkgs_cache"

HF_HOME_CACHE="${PROJECT_DIR}/hf_home"
HF_DATASETS_CACHE="${PROJECT_DIR}/hf_datasets_cache"

REPO_ROOT="${PROJECT_DIR}"
export REPO_ROOT

SWEEP_YAML="${REPO_ROOT}/config/sep_sweep.yaml"

# Optional: .env with HUGGING_FACE_HUB_TOKEN, OPENAI_API_KEY etc.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  . "${REPO_ROOT}/.env"
  set +a
  echo ".env loaded from ${REPO_ROOT}/.env"
else
  echo "No .env found at ${REPO_ROOT}/.env (skipping)"
fi

# Export HF caches
export HF_HOME="${HF_HOME_CACHE}"
export HF_HUB_CACHE="${HF_HOME_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"

mkdir -p "${CONDA_ENVS_DIR}" "${CONDA_PKGS_CACHE}" \
         "${HF_HOME_CACHE}" "${HF_DATASETS_CACHE}" \
         "${REPO_ROOT}/outputs" "${REPO_ROOT}/slurm_logs" "${REPO_ROOT}/scripts"

module load Anaconda3
eval "$(conda shell.bash hook)"

export CONDA_PKGS_DIRS="${CONDA_PKGS_CACHE}"

if [[ ! -d "${TARGET_CONDA_ENV}" ]]; then
  echo "Creating env ${CONDA_ENV_NAME} @ ${TARGET_CONDA_ENV}..."
  conda create --prefix "${TARGET_CONDA_ENV}" "python=${PYTHON_MAJOR_MINOR}" -y
else
  echo "Env exists: ${TARGET_CONDA_ENV}"
fi

echo "Activating env..."
conda activate "${TARGET_CONDA_ENV}"

# Install project (editable)
python -m pip install --upgrade pip
pip install -e "${REPO_ROOT}"

# Optional HF login
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential || true
fi

# ---------- Pre-download models and datasets from config ----------
python - <<'PY'
import os, yaml
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

repo_root = Path(os.environ["REPO_ROOT"])
cfg_path = repo_root / "config" / "sep_sweep.yaml"

if not cfg_path.exists():
    raise FileNotFoundError(f"Config not found at {cfg_path}")

cfg = yaml.safe_load(cfg_path.read_text())

# generator models
models = list(cfg.get("models", []))

# optional HF judge model (string or list)
jm = cfg.get("judge_model")
if isinstance(jm, str):
    models.append(jm)
elif isinstance(jm, list):
    models.extend(jm)

# entailment model for semantic entropy
sem = cfg.get("semantic_entropy", {})
ent_name = sem.get("entailment_model")
# map config key -> actual HF repo id
ent_map = {
    "deberta": "microsoft/deberta-v2-xlarge-mnli",
}
if ent_name in ent_map:
    models.append(ent_map[ent_name])

datasets = cfg.get("datasets", [])
token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

print("Pre-downloading models into HF cache:")
for m in models:
    print(" -", m)
    try:
        snapshot_download(repo_id=m, token=token)
    except Exception as e:
        print(f"ERROR snapshotting {m}: {e}")

print("Pre-downloading datasets into HF datasets cache:")
for ds in datasets:
    try:
        print(" -", ds)
        # Match names used in data_utils.load_ds
        if ds == "trivia_qa":
            _ = load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')
        elif ds == "med_qa":
            _ = load_dataset("bigbio/med_qa")
        elif ds == "squad":
            _ = load_dataset("squad_v2")
        elif ds == "nq":
            _ = load_dataset("nq_open")
        elif ds == "svamp":
            _ = load_dataset("ChilleD/SVAMP")
        elif ds == "bioasq":
            print("   (bioasq expects local JSON; skipping download)")
        else:
            _ = load_dataset(ds)
    except Exception as e:
        print(f"ERROR loading dataset {ds}: {e}")
PY

echo "Setup complete."
echo "Conda env: ${TARGET_CONDA_ENV}"
