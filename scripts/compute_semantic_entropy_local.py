import argparse
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

    # For offline HPC we stick to DeBERTa
    if entailment_model_name != "deberta":
        raise ValueError("For offline HPC, semantic_entropy_local only supports entailment_model=deberta.")
    entailment_model = EntailmentDeberta()

    runs_root = Path(args.runs_root)

    for model in models:
        for ds in datasets:
            run_dir = runs_root / f"{Path(model).name}__{ds}"
            val_path = run_dir / "validation_generations.pkl"
            if not val_path.exists():
                print(f"WARNING: {val_path} not found, skipping.")
                continue

            print(f"Computing SE for {model} / {ds}")
            with val_path.open("rb") as f:
                validation_generations = pickle.load(f)

            entropy_dict = {
                "cluster_assignment_entropy": [],
                # optionally: other variants
                "semantic_entropy_sum": [],
                "semantic_entropy_sum-normalized": [],
                "semantic_entropy_sum-normalized-rao": [],
                "semantic_entropy_mean": [],
            }
            validation_is_false = []

            # Loop over datapoints
            for tid, ex in enumerate(validation_generations.values()):
                question = ex["question"]
                context = ex["context"]
                full_responses = ex["responses"]
                most_likely = ex["most_likely_answer"]

                if use_all_generations:
                    responses = [r[0] for r in full_responses]
                    log_liks = [r[1] for r in full_responses]
                else:
                    # e.g. limit to first N generations
                    responses = [r[0] for r in full_responses]
                    log_liks = [r[1] for r in full_responses]

                # log_liks: list of list-of-token-log-likelihoods
                # aggregate per sequence
                log_liks_agg = [np.mean(ll) for ll in log_liks]

                # Optionally condition responses on question like compute_uncertainty_measures
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

                # Standard predictive entropy on sequences
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

                # correctness labels from most_likely_answer (already computed by generate_answers)
                validation_is_false.append(1.0 - most_likely["accuracy"])

            out = {
                "uncertainty_measures": entropy_dict,
                "validation_is_false": validation_is_false,
            }

            out_path = run_dir / "uncertainty_measures_local.pkl"
            with out_path.open("wb") as f:
                pickle.dump(out, f)
            print(f"Saved SE measures to {out_path}")

if __name__ == "__main__":
    main()
