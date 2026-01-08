#!/bin/bash
set -e

# Define your models here
MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "HuggingFaceTB/SmolLM3-3B"
    "Qwen/Qwen3-4B-Instruct-2507"
    "google/gemma-3-4b-it"
    "google/gemma-7b-it"
    # "mistralai/Ministral-3-3B-Instruct-2512"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Ministral-3-8B-Instruct-2512"
    "mistralai/Ministral-8B-Instruct-2410"
    "Qwen/Qwen3-8B"
)

# Base path to your script
SCRIPT_PATH="scripts/run_sep_sweep.sh"

for m in "${MODELS[@]}"; do
    # Create a short name for the job (e.g., Llama-3.1)
    # This strips everything before the last slash
    SHORT_NAME=$(basename "$m")
    
    echo "Submitting job for: $m (Job Name: sep_$SHORT_NAME)"
    
    # Export the variable so sbatch sees it
    export TARGET_MODEL="$m"
    
    sbatch --job-name="sep_${SHORT_NAME}" "$SCRIPT_PATH"
done