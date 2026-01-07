#!/usr/bin/env python3
"""
train_probes.py 

- Loads validation_generations.pkl + uncertainty_measures.pkl
- Deterministic 3-way split: train / calibration / test
- Selects best layer for each probe (accuracy + entropy) using K-fold CV on TRAIN only
- Refits probe on all TRAIN at chosen layer
- Saves probes.pkl per run with:
    - splits
    - entropy threshold (computed on TRAIN)
    - best layers
    - CV AUC per layer
    - trained sklearn pipelines (StandardScaler + LogisticRegression)

No test leakage: test is never used for model selection.
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import yaml

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class SEPData:
    name: str
    tbg: torch.Tensor         # [L, N, D]
    slt: torch.Tensor         # [L, N, D]
    entropy: torch.Tensor     # [N] float
    y_correct: torch.Tensor   # [N] {0,1}


# -----------------------------
# Wandb files dir finder
# -----------------------------

def find_wandb_files_dir(run_dir: Path) -> Optional[Path]:
    user = os.environ.get("USER", "")
    base_wandb_root = run_dir / user / "uncertainty" / "wandb"

    latest = base_wandb_root / "latest-run"
    if latest.is_symlink() or latest.is_dir():
        files_dir = latest / "files"
        if files_dir.exists():
            return files_dir

    candidates = list(run_dir.rglob("offline-run-*"))
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    files_dir = candidates[0] / "files"
    if files_dir.exists():
        return files_dir

    return None

def find_run_directory(root_path: Path, model_name: str, dataset_name: str) -> Optional[Path]:
    safe_model_name = Path(model_name).name
    target_folder_name = f"{safe_model_name}__{dataset_name}"
    candidates: List[Path] = []

    if root_path.exists():
        flat_path = root_path / target_folder_name
        if flat_path.exists():
            candidates.append(flat_path)

        for p in root_path.iterdir():
            if p.is_dir():
                nested_path = p / target_folder_name
                if nested_path.exists():
                    candidates.append(nested_path)

    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

# -----------------------------
# Loading
# -----------------------------

def _load_generations_sorted(val_path: Path) -> list:
    with val_path.open("rb") as f:
        gen_dict = pickle.load(f)
    # Sort by key for deterministic alignment (matches SE computation script)
    keys = sorted(gen_dict.keys())
    return [gen_dict[k] for k in keys]


def _load_entropy_array(measures_path: Path, entropy_key: str) -> np.ndarray:
    with measures_path.open("rb") as f:
        measures = pickle.load(f)

    if "uncertainty_measures" not in measures or entropy_key not in measures["uncertainty_measures"]:
        raise KeyError(f"Missing entropy key '{entropy_key}' in uncertainty_measures.pkl")

    ent = np.asarray(measures["uncertainty_measures"][entropy_key], dtype=np.float32)
    return ent


def _process_emb(raw_list: list) -> torch.Tensor:
    """
    raw_list: list of tensors [L, 1, D] (or compatible)
    returns: [L, N, D]
    """
    t = torch.stack(raw_list)  # [N, L, 1, D] typically
    while t.ndim > 3:
        t = t.squeeze(-2)      # remove the singleton token dimension if present
    if t.ndim != 3:
        raise ValueError(f"Unexpected embedding tensor shape after squeeze: {tuple(t.shape)}")
    # [N, L, D] -> [L, N, D]
    return t.transpose(0, 1).to(torch.float32).contiguous()


def load_sep_dataset(
    val_path: Path,
    measures_path: Path,
    n_samples: int,
    entropy_key: str,
    name: str,
) -> SEPData:
    gens = _load_generations_sorted(val_path)
    entropy = _load_entropy_array(measures_path, entropy_key)

    # y_correct: 1 iff accuracy==1.0 (binary)
    acc = np.asarray([rec["most_likely_answer"]["accuracy"] for rec in gens], dtype=np.float32)
    y_correct = (acc >= 1.0).astype(np.int64)

    # embeddings
    tbg_raw = [rec["most_likely_answer"]["emb_last_tok_before_gen"] for rec in gens]
    slt_raw = [rec["most_likely_answer"]["emb_tok_before_eos"] for rec in gens]

    if any(x is None for x in tbg_raw):
        raise ValueError("Found None TBG embeddings in generations.")
    if any(x is None for x in slt_raw):
        raise ValueError("Found None SLT embeddings in generations.")

    tbg = _process_emb(tbg_raw)
    slt = _process_emb(slt_raw)

    n = min(n_samples, tbg.shape[1], slt.shape[1], len(entropy), len(y_correct))
    tbg = tbg[:, :n, :]
    slt = slt[:, :n, :]
    entropy = entropy[:n]
    y_correct = y_correct[:n]

    return SEPData(
        name=name,
        tbg=tbg,
        slt=slt,
        entropy=torch.tensor(entropy, dtype=torch.float32),
        y_correct=torch.tensor(y_correct, dtype=torch.long),
    )


# -----------------------------
# Splits
# -----------------------------

def get_split_indices_3way(
    n_total: int,
    train_frac: float,
    cal_frac: float,
    seed: int,
    stratify_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic 3-way split with stratification:
      train / calibration / test
    """
    test_frac = 1.0 - (train_frac + cal_frac)
    if test_frac < 0:
        raise ValueError("train_frac + cal_frac > 1.0")

    idxs = np.arange(n_total)

    train_idx, tmp_idx = train_test_split(
        idxs,
        test_size=(1.0 - train_frac),
        random_state=seed,
        stratify=stratify_y,
    )

    # stratify second split using labels restricted to tmp
    tmp_y = stratify_y[tmp_idx]
    rel_test = test_frac / (cal_frac + test_frac + 1e-12)

    cal_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=rel_test,
        random_state=seed,
        stratify=tmp_y,
    )

    return train_idx, cal_idx, test_idx


# -----------------------------
# CV layer selection
# -----------------------------

def cv_select_layer_and_refit(
    X_layers: np.ndarray,   # [L, N, D]
    y: np.ndarray,          # [N] {0,1}
    idx_train: np.ndarray,
    n_splits: int,
    seed: int,
    C: float,
    max_iter: int,
) -> Tuple[int, object, List[float]]:
    """
    For each layer:
      - K-fold CV AUROC on TRAIN only
    Select best layer by mean CV AUROC.
    Refit on all TRAIN at best layer.
    Returns: best_layer, fitted_pipeline, cv_auc_per_layer
    """
    L = X_layers.shape[0]
    y_tr = y[idx_train]
    # if only one class, AUC is undefined; fall back to layer 0
    if len(np.unique(y_tr)) < 2:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"))
        pipe.fit(X_layers[0][idx_train], y_tr)
        return 0, pipe, [0.5] * L

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_aucs: List[float] = []
    for l in range(L):
        fold_aucs = []
        Xl = X_layers[l][idx_train]
        for tr, va in skf.split(Xl, y_tr):
            Xt, Xv = Xl[tr], Xl[va]
            yt, yv = y_tr[tr], y_tr[va]

            if len(np.unique(yv)) < 2:
                continue

            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"))
            pipe.fit(Xt, yt)
            p = pipe.predict_proba(Xv)[:, 1]
            fold_aucs.append(roc_auc_score(yv, p))

        cv_aucs.append(float(np.mean(fold_aucs)) if fold_aucs else 0.5)

    best_layer = int(np.nanargmax(cv_aucs))

    best_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"))
    best_pipe.fit(X_layers[best_layer][idx_train], y_tr)

    return best_layer, best_pipe, cv_aucs


def eval_auc_on_split(
    pipe, X: np.ndarray, y: np.ndarray, idx: np.ndarray
) -> float:
    ys = y[idx]
    if len(np.unique(ys)) < 2:
        return 0.5
    p = pipe.predict_proba(X[idx])[:, 1]
    return float(roc_auc_score(ys, p))


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--runs-root", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)

    # dataset size cap
    ap.add_argument("--n-samples", type=int, default=None)

    # which SE key to use
    ap.add_argument("--entropy-key", type=str, default=None)

    # splits
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--cal-frac", type=float, default=None)

    # CV
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--max-iter", type=int, default=1000)

    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg["models"]
    datasets = cfg["datasets"]

    probe_cfg = cfg.get("probes", {})
    n_samples = args.n_samples if args.n_samples is not None else probe_cfg.get("n_samples", 2000)
    entropy_key = args.entropy_key if args.entropy_key is not None else probe_cfg.get("entropy_key", "cluster_assignment_entropy")

    train_frac = args.train_frac if args.train_frac is not None else probe_cfg.get("train_frac", 0.70)
    cal_frac = args.cal_frac if args.cal_frac is not None else probe_cfg.get("cal_frac", 0.15)

    runs_root = Path(args.runs_root)

    for model in models:
        for ds in datasets:
            run_dir = find_run_directory(runs_root, model, ds)
            if run_dir is None:
                print(f"[train_probes] No run found for {Path(model).name}__{ds}, skipping.")
                continue

            print(f"[train_probes] Processing {run_dir.name}")

            files_dir = find_wandb_files_dir(run_dir)
            if files_dir is None:
                print(f"[train_probes] No files dir under {run_dir}, skipping.")
                continue

            val_path = files_dir / "validation_generations.pkl"
            unc_path = files_dir / "uncertainty_measures.pkl"
            if not val_path.exists() or not unc_path.exists():
                print(f"[train_probes] Missing pickles in {files_dir}, skipping.")
                continue

            name = f"{Path(model).name}__{ds}"
            print(f"[train_probes] Loading {name}")

            D = load_sep_dataset(
                val_path=val_path,
                measures_path=unc_path,
                n_samples=n_samples,
                entropy_key=entropy_key,
                name=name,
            )

            # Prepare labels
            y_correct = D.y_correct.numpy().astype(np.int64)      # 1 correct, 0 halluc
            y_hall = (1 - y_correct).astype(np.int64)

            # Split indices: stratify on y_correct
            train_idx, cal_idx, test_idx = get_split_indices_3way(
                n_total=len(y_correct),
                train_frac=train_frac,
                cal_frac=cal_frac,
                seed=args.seed,
                stratify_y=y_correct,
            )
            print(f"[train_probes] Splits: train={len(train_idx)} cal={len(cal_idx)} test={len(test_idx)}")

            # Entropy binarization threshold computed on TRAIN only (per-run)
            ent = D.entropy.numpy().astype(np.float32)
            ent_thr = float(np.median(ent[train_idx]))
            y_high_ent = (ent > ent_thr).astype(np.int64)

            # Convert embeddings
            X_tbg = D.tbg.numpy().astype(np.float32)  # [L,N,D]
            X_slt = D.slt.numpy().astype(np.float32)

            # Train probes with CV layer selection on TRAIN only
            # Accuracy probe predicts y_correct
            tbg_acc_layer, tbg_acc_model, tbg_acc_cv = cv_select_layer_and_refit(
                X_layers=X_tbg,
                y=y_correct,
                idx_train=train_idx,
                n_splits=args.cv_folds,
                seed=args.seed,
                C=args.C,
                max_iter=args.max_iter,
            )
            slt_acc_layer, slt_acc_model, slt_acc_cv = cv_select_layer_and_refit(
                X_layers=X_slt,
                y=y_correct,
                idx_train=train_idx,
                n_splits=args.cv_folds,
                seed=args.seed,
                C=args.C,
                max_iter=args.max_iter,
            )

            # SE probe predicts y_high_ent
            tbg_se_layer, tbg_se_model, tbg_se_cv = cv_select_layer_and_refit(
                X_layers=X_tbg,
                y=y_high_ent,
                idx_train=train_idx,
                n_splits=args.cv_folds,
                seed=args.seed,
                C=args.C,
                max_iter=args.max_iter,
            )
            slt_se_layer, slt_se_model, slt_se_cv = cv_select_layer_and_refit(
                X_layers=X_slt,
                y=y_high_ent,
                idx_train=train_idx,
                n_splits=args.cv_folds,
                seed=args.seed,
                C=args.C,
                max_iter=args.max_iter,
            )

            # Report AUCs on calibration and test (for sanity only; not used for selection)
            def _report(pipe, Xl, label, layer, tag):
                auc_cal = eval_auc_on_split(pipe, Xl[layer], label, cal_idx)
                auc_test = eval_auc_on_split(pipe, Xl[layer], label, test_idx)
                return {"cal_auc": auc_cal, "test_auc": auc_test}

            summary = {
                "tbg_acc": _report(tbg_acc_model, X_tbg, y_correct, tbg_acc_layer, "tbg_acc"),
                "slt_acc": _report(slt_acc_model, X_slt, y_correct, slt_acc_layer, "slt_acc"),
                "tbg_se": _report(tbg_se_model, X_tbg, y_high_ent, tbg_se_layer, "tbg_se"),
                "slt_se": _report(slt_se_model, X_slt, y_high_ent, slt_se_layer, "slt_se"),
            }
            print(f"[train_probes] {name} chosen layers: "
                  f"TBG(acc={tbg_acc_layer}, se={tbg_se_layer}) "
                  f"SLT(acc={slt_acc_layer}, se={slt_se_layer})")
            print(f"[train_probes] sanity AUCs (cal/test): {summary}")

            out = {
                "meta": {
                    "name": name,
                    "seed": args.seed,
                    "n_samples": n_samples,
                    "entropy_key": entropy_key,
                    "entropy_threshold_train_median": ent_thr,
                    "train_frac": train_frac,
                    "cal_frac": cal_frac,
                    "cv_folds": args.cv_folds,
                    "C": args.C,
                    "max_iter": args.max_iter,
                },
                "splits": {
                    "train": train_idx,
                    "calibration": cal_idx,
                    "test": test_idx,
                },
                "probes": {
                    "tbg": {
                        "acc": {
                            "best_layer": tbg_acc_layer,
                            "cv_auc_per_layer": tbg_acc_cv,
                            "model": tbg_acc_model,
                        },
                        "se": {
                            "best_layer": tbg_se_layer,
                            "cv_auc_per_layer": tbg_se_cv,
                            "model": tbg_se_model,
                        },
                    },
                    "slt": {
                        "acc": {
                            "best_layer": slt_acc_layer,
                            "cv_auc_per_layer": slt_acc_cv,
                            "model": slt_acc_model,
                        },
                        "se": {
                            "best_layer": slt_se_layer,
                            "cv_auc_per_layer": slt_se_cv,
                            "model": slt_se_model,
                        },
                    },
                },
                "sanity_eval": summary,
            }

            out_path = files_dir / "probes.pkl"
            with out_path.open("wb") as f:
                pickle.dump(out, f)
            print(f"[train_probes] Saved {out_path}")


if __name__ == "__main__":
    main()
