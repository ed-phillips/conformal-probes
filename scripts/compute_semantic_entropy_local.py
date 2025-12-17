import argparse
import os
import pickle
import yaml
from pathlib import Path
import numpy as np

from semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    get_semantic_ids,
    cluster_assignment_entropy,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)

def find_wandb_files_dir(run_dir: Path) -> Path | None:
    """
    Find the wandb offline run's `files/` directory under `run_dir`.

    We try:
      run_dir/<user>/uncertainty/wandb/latest-run/files
      else: the most recent run_dir/**/offline-run-*/files
    """
    user = os.environ.get("USER", "")
    base_wandb_root = run_dir / user / "uncertainty" / "wandb"

    # 1) If latest-run symlink exists, prefer it
    latest = base_wandb_root / "latest-run"
    if latest.is_symlink() or latest.is_dir():
        files_dir = latest / "files"
        if files_dir.exists():
            return files_dir

    # 2) Otherwise, search for offline-run-* dirs and pick the newest
    candidates = list(run_dir.rglob("offline-run-*"))
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    files_dir = candidates[0] / "files"
    if files_dir.exists():
        return files_dir

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--runs-root", type=str, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg["models"]
    datasets = cfg["datasets"]
    sem_cfg = cfg.get("semantic_entropy", {})

    entailment_model_name = sem_cfg.get("entailment_model", "deberta")
    strict_entailment = sem_cfg.get("strict_entailment", True)
    use_all_generations = sem_cfg.get("use_all_generations", True)

    if entailment_model_name != "deberta":
        raise ValueError(
            "semantic_entropy_local only supports entailment_model=deberta (offline-friendly)."
        )
    entailment_model = EntailmentDeberta()

    runs_root = Path(args.runs_root)

    for model in models:
        for ds in datasets:
            run_dir = runs_root / f"{Path(model).name}__{ds}"
            if not run_dir.exists():
                print(f"[SE] Run dir {run_dir} does not exist, skipping.")
                continue

            files_dir = find_wandb_files_dir(run_dir)
            if files_dir is None:
                print(f"[SE] No wandb offline-run files dir found under {run_dir}, skipping.")
                continue

            val_path = files_dir / "validation_generations.pkl"
            if not val_path.exists():
                print(f"[SE] {val_path} not found, skipping.")
                continue

            print(f"[SE] Computing SE for {model} / {ds} from {val_path}")

            with val_path.open("rb") as f:
                validation_generations = pickle.load(f)

            entropy_dict = {
                "cluster_assignment_entropy": [],
                "semantic_entropy_sum": [],
                "semantic_entropy_sum-normalized": [],
                "semantic_entropy_sum-normalized-rao": [],
                "semantic_entropy_mean": [],
            }
            validation_is_false = []

            # sort keys to ensure deterministic order matching train_probes.p
            sorted_ids = sorted(validation_generations.keys())

            for tid in sorted_ids:
                ex = validation_generations[tid]
                question = ex["question"]
                context = ex["context"]
                full_responses = ex["responses"]
                most_likely = ex["most_likely_answer"]

                if use_all_generations:
                    responses = [r[0] for r in full_responses]
                    log_liks = [r[1] for r in full_responses]
                else:
                    responses = [r[0] for r in full_responses]
                    log_liks = [r[1] for r in full_responses]

                log_liks_agg = [np.mean(ll) for ll in log_liks]
                responses_for_ent = [f"{question} {r}" for r in responses]

                semantic_ids = get_semantic_ids(
                    responses_for_ent,
                    model=entailment_model,
                    strict_entailment=strict_entailment,
                    example=ex,
                )

                entropy_dict["cluster_assignment_entropy"].append(
                    cluster_assignment_entropy(semantic_ids)
                )

                entropy_dict["semantic_entropy_sum"].append(
                    predictive_entropy(
                        logsumexp_by_id(semantic_ids, log_liks_agg, agg="sum")
                    )
                )
                entropy_dict["semantic_entropy_sum-normalized"].append(
                    predictive_entropy(
                        logsumexp_by_id(semantic_ids, log_liks_agg, agg="sum_normalized")
                    )
                )
                entropy_dict["semantic_entropy_sum-normalized-rao"].append(
                    predictive_entropy_rao(
                        logsumexp_by_id(semantic_ids, log_liks_agg, agg="sum_normalized")
                    )
                )
                entropy_dict["semantic_entropy_mean"].append(
                    predictive_entropy(
                        logsumexp_by_id(semantic_ids, log_liks_agg, agg="mean")
                    )
                )

                validation_is_false.append(1.0 - most_likely["accuracy"])

            out = {
                "uncertainty_measures": entropy_dict,
                "validation_is_false": validation_is_false,
            }

            out_path = files_dir / "uncertainty_measures.pkl"
            with out_path.open("wb") as f:
                pickle.dump(out, f)
            print(f"[SE] Saved SE measures to {out_path}")


if __name__ == "__main__":
    main()
