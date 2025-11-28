import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class SEPData:
    name: str
    tbg_dataset: torch.Tensor   # [layers, N, d]
    slt_dataset: torch.Tensor   # [layers, N, d]
    entropy: torch.Tensor       # [N]
    accuracies: torch.Tensor    # [N]
    # we’ll attach split_indices and b_entropy later in main()


def load_sep_dataset(run_dir: Path, entropy_key: str) -> SEPData:
    """
    Load ALL examples from validation_generations + uncertainty_measures for this run.
    Truncation to num_samples is done later in main().
    """
    with (run_dir / "validation_generations.pkl").open("rb") as f:
        generations = pickle.load(f)
    with (run_dir / "uncertainty_measures.pkl").open("rb") as f:
        measures = pickle.load(f)

    entropy = torch.tensor(
        measures["uncertainty_measures"][entropy_key], dtype=torch.float32
    )

    # Most likely answer correctness (0/1)
    accuracies = torch.tensor(
        [rec["most_likely_answer"]["accuracy"] for rec in generations.values()],
        dtype=torch.float32,
    )

    # TBG: emb_last_tok_before_gen, SLT: emb_tok_before_eos
    tbg = torch.stack(
        [rec["most_likely_answer"]["emb_last_tok_before_gen"]
         for rec in generations.values()]
    ).squeeze(-2).transpose(0, 1).to(torch.float32)

    slt = torch.stack(
        [rec["most_likely_answer"]["emb_tok_before_eos"]
         for rec in generations.values()]
    ).squeeze(-2).transpose(0, 1).to(torch.float32)

    return SEPData(
        name=run_dir.name,
        tbg_dataset=tbg,
        slt_dataset=slt,
        entropy=entropy,
        accuracies=accuracies,
    )


def best_split(entropy: torch.Tensor) -> float:
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    best_mse = np.inf
    best_split_val = splits[0]
    for s in splits:
        low = ents < s
        high = ~low
        low_mean = ents[low].mean() if low.any() else 0.0
        high_mean = ents[high].mean() if high.any() else 0.0
        mse = ((ents[low] - low_mean)**2).sum() + ((ents[high] - high_mean)**2).sum()
        if mse < best_mse:
            best_mse = mse
            best_split_val = s
    return float(best_split_val)


def binarize_entropy(entropy: torch.Tensor, thres: float) -> torch.Tensor:
    out = torch.full_like(entropy, -1.0)
    out[entropy < thres] = 0.0
    out[entropy > thres] = 1.0
    return out


def make_split_indices(N: int, train_frac: float, val_frac: float, seed: int = 42):
    """
    Create dataset-level train/val/test indices:
      - train_frac
      - val_frac
      - test_frac = 1 - train_frac - val_frac
    The same indices will be used for all layers and both token types.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    n_test = N - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return train_idx, val_idx, test_idx


def create_splits(datasets, scores, split_indices):
    """
    datasets: torch.Tensor [layers, N, d]
    scores: 1D array-like [N]
    split_indices: (train_idx, val_idx, test_idx)
    """
    X = np.array(datasets)  # [layers, N, d]
    y = np.array(scores)
    n_layers = X.shape[0]

    train_idx, val_idx, test_idx = split_indices

    X_trains, X_vals, X_tests = [], [], []
    y_trains, y_vals, y_tests = [], [], []

    for i in range(n_layers):
        X_layer = X[i]
        X_trains.append(X_layer[train_idx])
        X_vals.append(X_layer[val_idx])
        X_tests.append(X_layer[test_idx])
        y_trains.append(y[train_idx])
        y_vals.append(y[val_idx])
        y_tests.append(y[test_idx])

    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests


def train_per_layer_probes(D: SEPData, token_type: str, metric: str, split_indices):
    """
    Train a LogisticRegression probe per layer, using:
      - train split for fitting
      - test split for AUC
    """
    if token_type == "tbg":
        data = D.tbg_dataset
    else:
        data = D.slt_dataset

    if metric == "entropy":
        y = D.b_entropy.numpy()
    elif metric == "accuracy":
        y = D.accuracies.numpy()
    else:
        raise ValueError

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_splits(data, y, split_indices)

    aucs = []
    models = []
    for i, (Xt, Xv, Xte, yt, yv, yte) in enumerate(
        zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)
    ):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xt, yt)                     # train on TRAIN split only
        probs = clf.predict_proba(Xv)[:, 1]  # evaluate on validation split
        auc = roc_auc_score(yv, probs)
        aucs.append(auc)
        models.append(clf)
    return aucs, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--runs-root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg["models"]
    datasets = cfg["datasets"]

    gen_cfg = cfg.get("generation", {})
    probe_cfg = cfg.get("probes", {})

    # total number of samples we *generated* per dataset
    total_samples = gen_cfg.get("num_samples", 2000)

    # splitting is now only configured here
    train_frac = probe_cfg.get("train_frac", 0.7)
    val_frac = probe_cfg.get("val_frac", 0.15)
    test_frac = probe_cfg.get("test_frac", 0.15)
    if train_frac + val_frac + test_frac > 1.0 + 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must be <= 1.")

    entropy_key = probe_cfg.get("entropy_key", "cluster_assignment_entropy")

    runs_root = Path(args.runs_root)
    all_Ds = []

    # 1) Load all datasets
    for model in models:
        for ds in datasets:
            run_dir = runs_root / f"{Path(model).name}__{ds}"
            if not (run_dir / "validation_generations.pkl").exists():
                print(f"Skipping {run_dir}, missing validation_generations.pkl")
                continue
            if not (run_dir / "uncertainty_measures.pkl").exists():
                print(f"Skipping {run_dir}, missing uncertainty_measures.pkl")
                continue

            D = load_sep_dataset(run_dir, entropy_key)
            N = D.entropy.shape[0]
            if N == 0:
                print(f"{run_dir} has zero examples, skipping.")
                continue

            # truncate to total_samples if there are more
            N_use = min(total_samples, N)
            D.tbg_dataset = D.tbg_dataset[:, :N_use, :]
            D.slt_dataset = D.slt_dataset[:, :N_use, :]
            D.entropy = D.entropy[:N_use]
            D.accuracies = D.accuracies[:N_use]

            D.name = f"{Path(model).name}__{ds}"
            all_Ds.append(D)

    if not all_Ds:
        raise RuntimeError("No datasets found to train probes on.")

    # 2) Global best split for entropy (across all runs)
    all_entropy = torch.cat([D.entropy for D in all_Ds], dim=0)
    split = best_split(all_entropy)
    print("Best universal split:", split)

    for D in all_Ds:
        D.b_entropy = binarize_entropy(D.entropy, split)
        dummy_acc = max(D.b_entropy.mean().item(), 1 - D.b_entropy.mean().item())
        print(f"Dummy accuracy {D.name}: {dummy_acc:.4f}")

    # 3) Train probes per dataset using explicit train/val/test splits
    results = {}
    for D in all_Ds:
        print(f"\nTraining probes for {D.name}")
        N = D.entropy.shape[0]
        split_indices = make_split_indices(N, train_frac, val_frac, seed=42)
        D.split_indices = split_indices  # for debugging / later analysis

        results[D.name] = {}

        # SE probes (predict high vs low entropy)
        tb_auc, tb_models = train_per_layer_probes(D, "tbg", "entropy", split_indices)
        sb_auc, sb_models = train_per_layer_probes(D, "slt", "entropy", split_indices)

        # Accuracy probes (predict correctness)
        ta_auc, ta_models = train_per_layer_probes(D, "tbg", "accuracy", split_indices)
        sa_auc, sa_models = train_per_layer_probes(D, "slt", "accuracy", split_indices)

        results[D.name]["tb_aucs"] = tb_auc
        results[D.name]["sb_aucs"] = sb_auc
        results[D.name]["ta_aucs"] = ta_auc
        results[D.name]["sa_aucs"] = sa_auc
        results[D.name]["tb_models"] = tb_models
        results[D.name]["sb_models"] = sb_models
        results[D.name]["ta_models"] = ta_models
        results[D.name]["sa_models"] = sa_models
        results[D.name]["n_examples"] = N
        results[D.name]["train_frac"] = train_frac
        results[D.name]["val_frac"] = val_frac
        results[D.name]["test_frac"] = test_frac

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"split": split, "results": results}, f)

    print("\nSaved probe models + metrics to", out_path)


if __name__ == "__main__":
    main()
