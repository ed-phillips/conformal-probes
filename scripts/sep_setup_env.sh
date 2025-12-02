#!/usr/bin/env bash
# Run ONCE on a LOGIN node with internet access.
# Sets up conda env from sep_environment.yaml and pre-downloads models/datasets.

set -euo pipefail

# ---------- Config ----------
PROJECT_NAME="conformal-probes"
CONDA_ENV_NAME="se_probes"             # must match your YAML "name", but we use a prefix path
ENV_FILE_NAME="sep_env_312.yaml"   # your env file

DATA_ROOT="${DATA:-/well/clifton/users/gar957}"
PROJECT_DIR="${DATA_ROOT}/${PROJECT_NAME}"
REPO_ROOT="${PROJECT_DIR}"
export REPO_ROOT

ENV_YAML="${REPO_ROOT}/${ENV_FILE_NAME}"
SWEEP_YAML="${REPO_ROOT}/config/sep_sweep.yaml"

# HF caches on NFS
HF_HOME_CACHE="${PROJECT_DIR}/hf_home"
HF_DATASETS_CACHE="${PROJECT_DIR}/hf_datasets_cache"

# Where we want envs/pkgs to live (NOT in home)
CONDA_ENVS_DIR="${PROJECT_DIR}/conda_envs"
CONDA_PKGS_CACHE="${PROJECT_DIR}/conda_pkgs_cache"
TARGET_CONDA_ENV="${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}"

# Optional: .env with HUGGING_FACE_HUB_TOKEN etc.
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

# ---------- Conda setup ----------
module load Anaconda3
eval "$(conda shell.bash hook)"




# Make sure conda uses project storage for pkgs/envs
export CONDA_ENVS_DIR="${CONDA_ENVS_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_CACHE}"

if [[ ! -f "${ENV_YAML}" ]]; then
  echo "FATAL: env file not found at ${ENV_YAML}"
  exit 1
fi

echo "Using env YAML: ${ENV_YAML}"
echo "Target env prefix: ${TARGET_CONDA_ENV}"

# Create or update env from YAML using PREFIX (avoids ~/.conda)
if [[ -d "${TARGET_CONDA_ENV}" ]]; then
  echo "Environment already exists at ${TARGET_CONDA_ENV}. Updating from YAML..."
  conda env update -p "${TARGET_CONDA_ENV}" -f "${ENV_YAML}"
else
  echo "Creating environment at ${TARGET_CONDA_ENV} from YAML..."
  conda env create -p "${TARGET_CONDA_ENV}" -f "${ENV_YAML}"
fi

echo "Activating env..."
# Temporarily disable 'unbound variable' check because conda scripts are messy
set +u
conda activate "${TARGET_CONDA_ENV}"
set -u

# Make repo importable if needed
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Optional HF login (only needed for private models)
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Logging into Hugging Face..."
  huggingface-cli login --token "${HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential || true
fi

# ---------- Pre-download models and datasets from config ----------
python - <<'PY'
import os, yaml
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

repo_root = Path(os.environ["REPO_ROOT"])
cfg_path  = repo_root / "config" / "sep_sweep.yaml"

if not cfg_path.exists():
    raise FileNotFoundError(f"Config not found at {cfg_path}")

cfg = yaml.safe_load(cfg_path.read_text())
token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

# -------------------------------------------------------------------------
# 1. Download Models
# -------------------------------------------------------------------------
models = list(cfg.get("models", []))

# Add Judge Model
jm = cfg.get("judge_model")
if isinstance(jm, str):
    models.append(jm)
elif isinstance(jm, list):
    models.extend(jm or [])

# Add Entailment Model
sem = cfg.get("semantic_entropy", {}) or {}
ent_name = sem.get("entailment_model")
ent_map = {
    "deberta": "microsoft/deberta-v2-xlarge-mnli",
}
if ent_name in ent_map:
    models.append(ent_map[ent_name])

print(f"--- Downloading {len(models)} Models ---")
for m in set(models):
    print(f"Downloading snapshot for model: {m}")
    try:
        snapshot_download(repo_id=m, token=token)
    except Exception as e:
        print(f"ERROR downloading model {m}: {e}")

# -------------------------------------------------------------------------
# 2. Download Datasets
# -------------------------------------------------------------------------
datasets_list = cfg.get("datasets", [])

# Map config names to HF Hub IDs
ds_map = {
    "trivia_qa":  "TimoImhof/TriviaQA-in-SQuAD-format",
    "squad":      "squad_v2",
    "nq":         "nq_open",
    "svamp":      "ChilleD/SVAMP",
    "medical_o1": "FreedomIntelligence/medical-o1-verifiable-problem",
}

print(f"--- Downloading {len(datasets_list)} Datasets ---")
for ds_name in datasets_list:
    # 1. Skip local-only datasets
    if ds_name == "bioasq" or ds_name == "record":
        print(f"Skipping {ds_name} (expects local files)")
        continue

    # 2. Resolve the Hub ID
    hub_id = ds_map.get(ds_name, ds_name)
    
    print(f"Processing dataset: {ds_name} -> {hub_id}")
    try:
        # STEP A: Explicit snapshot download (Raw files)
        print(f"  - Snapshotting {hub_id}...")
        snapshot_download(repo_id=hub_id, repo_type="dataset", token=token)
        
        # STEP B: Load dataset (Triggers Arrow conversion/caching)
        print(f"  - Building/Caching {hub_id}...")
        if ds_name == "trivia_qa":
            load_dataset(hub_id, "unmodified", token=token)
        else:
            # Works for medical_o1, squad, nq, svamp
            load_dataset(hub_id, token=token)
            
    except Exception as e:
        print(f"ERROR processing dataset {ds_name}: {e}")

PY

echo "Setup complete."
echo "Conda env prefix: ${TARGET_CONDA_ENV}"
echo "To use later on login/compute nodes:"
echo "  module load Anaconda3"
echo "  conda activate ${TARGET_CONDA_ENV}"
