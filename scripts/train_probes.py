import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass
import os

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class SEPData:
    name: str
    tbg_dataset: torch.Tensor   # [layers, N, d]
    slt_dataset: torch.Tensor
    entropy: torch.Tensor       # [N]
    accuracies: torch.Tensor    # [N]


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

def load_sep_dataset(
    val_path: Path,
    measures_path: Path,
    n_samples: int,
    entropy_key: str,
    name: str,
) -> SEPData:
    """
    Load embeddings + entropy from explicit file paths (not assuming run_dir root).
    """
    with val_path.open("rb") as f:
        generations = pickle.load(f)
    with measures_path.open("rb") as f:
        measures = pickle.load(f)

    entropy = torch.tensor(
        measures["uncertainty_measures"][entropy_key], dtype=torch.float32
    )

    # Most likely answer correctness
    accuracies = torch.tensor(
        [rec["most_likely_answer"]["accuracy"] for rec in generations.values()],
        dtype=torch.float32,
    )

    # TBG: emb_last_tok_before_gen, SLT: emb_tok_before_eos
    tbg = torch.stack(
        [rec["most_likely_answer"]["emb_last_tok_before_gen"] for rec in generations.values()]
    ).squeeze(-2).transpose(0, 1).to(torch.float32)
    slt = torch.stack(
        [rec["most_likely_answer"]["emb_tok_before_eos"] for rec in generations.values()]
    ).squeeze(-2).transpose(0, 1).to(torch.float32)

    # truncate
    n = min(n_samples, tbg.shape[1])
    tbg = tbg[:, n * 0 : n, :]
    slt = slt[:, n * 0 : n, :]
    entropy = entropy[:n]
    accuracies = accuracies[:n]

    return SEPData(
        name=name,
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
        mse = ((ents[low] - low_mean) ** 2).sum() + ((ents[high] - high_mean) ** 2).sum()
        if mse < best_mse:
            best_mse = mse
            best_split_val = s
    return float(best_split_val)


def binarize_entropy(entropy: torch.Tensor, thres: float) -> torch.Tensor:
    out = torch.full_like(entropy, -1.0)
    out[entropy < thres] = 0.0
    out[entropy > thres] = 1.0
    return out


def create_splits(datasets, scores):
    X = np.array(datasets)  # [layers, N, d]
    y = np.array(scores)
    n_layers = X.shape[0]

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = [], [], [], [], [], []
    for i in range(n_layers):
        X_layer = X[i]
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_layer, y, test_size=0.1, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        X_trains.append(X_train)
        X_vals.append(X_val)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_vals.append(y_val)
        y_tests.append(y_test)
    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests


def train_per_layer_probes(D: SEPData, token_type: str, metric: str):
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

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_splits(data, y)

    aucs = []
    models = []
    for Xt, Xv, Xte, yt, yv, yte in zip(
        X_trains, X_vals, X_tests, y_trains, y_vals, y_tests
    ):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xt, yt)
        probs = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, probs)
        aucs.append(auc)
        models.append(clf)
    return aucs, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--runs-root", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg["models"]
    datasets = cfg["datasets"]
    probe_cfg = cfg.get("probes", {})
    n_samples = probe_cfg.get("n_samples", 2000)
    entropy_key = probe_cfg.get("entropy_key", "cluster_assignment_entropy")

    runs_root = Path(args.runs_root)
    all_Ds = []
    run_dirs = []

    # 1) Load all datasets (robust to wandb layout)
    for model in models:
        for ds in datasets:
            run_dir = runs_root / f"{Path(model).name}__{ds}"
            if not run_dir.exists():
                print(f"Skipping {run_dir}, directory does not exist")
                continue

            files_dir = find_wandb_files_dir(run_dir)

            val_path = files_dir / "validation_generations.pkl"
            unc_path = files_dir / "uncertainty_measures.pkl"

            # val_path, unc_path = find_run_files(run_dir)
            if val_path is None or unc_path is None:
                print(f"Skipping {run_dir}, could not find both pickles")
                continue

            print(f"Loading SEP data from {val_path} and {unc_path}")
            D = load_sep_dataset(
                val_path=val_path,
                measures_path=unc_path,
                n_samples=n_samples,
                entropy_key=entropy_key,
                name=f"{Path(model).name}__{ds}",
            )
            all_Ds.append(D)
            run_dirs.append(run_dir)

    if not all_Ds:
        raise RuntimeError("No valid runs found with both generations + uncertainty_measures.")

    # 2) Global best split for entropy (across all runs)
    all_entropy = torch.cat([D.entropy for D in all_Ds], dim=0)
    split = best_split(all_entropy)
    print("Best universal split:", split)

    for D in all_Ds:
        D.b_entropy = binarize_entropy(D.entropy, split)
        dummy_acc = max(D.b_entropy.mean().item(), 1 - D.b_entropy.mean().item())
        print(f"Dummy accuracy {D.name}: {dummy_acc:.4f}")

    # 3) Train probes per dataset and save per-run probes.pkl
    global_results = {}
    for D, run_dir in zip(all_Ds, run_dirs):
        print(f"Training probes for {D.name}")
        global_results[D.name] = {}

        # SE probes
        tb_auc, tb_models = train_per_layer_probes(D, "tbg", "entropy")
        sb_auc, sb_models = train_per_layer_probes(D, "slt", "entropy")

        # Accuracy probes
        ta_auc, ta_models = train_per_layer_probes(D, "tbg", "accuracy")
        sa_auc, sa_models = train_per_layer_probes(D, "slt", "accuracy")

        global_results[D.name]["tb_aucs"] = tb_auc
        global_results[D.name]["sb_aucs"] = sb_auc
        global_results[D.name]["ta_aucs"] = ta_auc
        global_results[D.name]["sa_aucs"] = sa_auc
        global_results[D.name]["tb_models"] = tb_models
        global_results[D.name]["sb_models"] = sb_models
        global_results[D.name]["ta_models"] = ta_models
        global_results[D.name]["sa_models"] = sa_models

        # --- per-run probes file ---
        per_run_out = files_dir / "probes.pkl"
        with per_run_out.open("wb") as f:
            pickle.dump({"split": split, "results": {D.name: global_results[D.name]}}, f)
        print(f"Saved probes for {D.name} to {per_run_out}")



if __name__ == "__main__":
    main()
