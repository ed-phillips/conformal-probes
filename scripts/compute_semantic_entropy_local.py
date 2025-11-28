# scripts/compute_semantic_entropy_local.py

import argparse
import pickle
import yaml
from pathlib import Path
import shutil
import numpy as np

from semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    get_semantic_ids,
    cluster_assignment_entropy,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)


def find_validation_generations(run_dir: Path) -> Path | None:
    """Find validation_generations.pkl inside a run directory."""
    top = run_dir / "validation_generations.pkl"
    if top.exists():
        return top
    candidates = list(run_dir.rglob("validation_generations.pkl"))
    return candidates[0] if candidates else None


def compute_se_for_generations(gen_path: Path, entailment_model, strict_entailment, use_all_generations):
    with gen_path.open("rb") as f:
        generations = pickle.load(f)

    entropy_dict = {
        "cluster_assignment_entropy": [],
        "semantic_entropy_sum": [],
        "semantic_entropy_sum-normalized": [],
        "semantic_entropy_sum-normalized-rao": [],
        "semantic_entropy_mean": [],
    }
    validation_is_false = []

    for tid, ex in enumerate(generations.values()):
        question = ex["question"]
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
            predictive_entropy(logsumexp_by_id(semantic_ids, log_liks_agg, agg="sum"))
        )
        entropy_dict["semantic_entropy_sum-normalized"].append(
            predictive_entropy(logsumexp_by_id(semantic_ids, log_liks_agg, agg="sum_normalized"))
        )
        entropy_dict["semantic_entropy_sum-normalized-rao"].append(
            predictive_entropy_rao(logsumexp_by_id(semantic_ids, log_liks_agg, agg="sum_normalized"))
        )
        entropy_dict["semantic_entropy_mean"].append(
            predictive_entropy(logsumexp_by_id(semantic_ids, log_liks_agg, agg="mean"))
        )

        validation_is_false.append(1.0 - most_likely["accuracy"])

    return {
        "uncertainty_measures": entropy_dict,
        "validation_is_false": validation_is_false,
    }


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
        raise ValueError("semantic_entropy_local only supports entailment_model=deberta offline.")
    entailment_model = EntailmentDeberta()

    runs_root = Path(args.runs_root)

    for model in models:
        for ds in datasets:
            run_dir = runs_root / f"{Path(model).name}__{ds}"
            if not run_dir.exists():
                print(f"Run dir {run_dir} does not exist, skipping.")
                continue

            val_src = find_validation_generations(run_dir)
            if val_src is None:
                print(f"WARNING: no validation_generations.pkl under {run_dir}, skipping.")
                continue

            val_root = run_dir / "validation_generations.pkl"
            if val_root != val_src and not val_root.exists():
                print(f"Found validation_generations.pkl at {val_src}, copying to {val_root}")
                val_root.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(val_src, val_root)

            print(f"Computing SE for {model} / {ds} from {val_root}")
            out = compute_se_for_generations(val_root, entailment_model, strict_entailment, use_all_generations)

            out_path = run_dir / "uncertainty_measures.pkl"
            with out_path.open("wb") as f:
                pickle.dump(out, f)
            print(f"Saved SE measures to {out_path}")


if __name__ == "__main__":
    main()
