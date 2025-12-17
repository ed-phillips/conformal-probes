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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass
class SEPData:
    name: str
    tbg_dataset: torch.Tensor   # [layers, N, d]
    slt_dataset: torch.Tensor
    entropy: torch.Tensor       # [N]
    accuracies: torch.Tensor    # [N]
    b_entropy: torch.Tensor = None 


def find_wandb_files_dir(run_dir: Path) -> Path | None:
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

def load_sep_dataset(
    val_path: Path,
    measures_path: Path,
    n_samples: int,
    entropy_key: str,
    name: str,
) -> SEPData:
    with val_path.open("rb") as f:
        # Sort by key to ensure deterministic order regardless of dictionary insertion order
        gen_dict = pickle.load(f)
        # Python 3.7+ preserves order, but sorting by ID is safer for consistency
        sorted_keys = sorted(gen_dict.keys())
        generations = [gen_dict[k] for k in sorted_keys]

    with measures_path.open("rb") as f:
        measures = pickle.load(f)

    # Entropy usually matches generation order if computed sequentially, 
    # but strictly we assume the list in measures corresponds to the list in generations.
    entropy = torch.tensor(
        measures["uncertainty_measures"][entropy_key], dtype=torch.float32
    )

    accuracies = torch.tensor(
        [rec["most_likely_answer"]["accuracy"] for rec in generations],
        dtype=torch.float32,
    )

    # Extract Hidden States
    # list of [Layers, 1, Dim]
    tbg_raw = [rec["most_likely_answer"]["emb_last_tok_before_gen"] for rec in generations]
    slt_raw = [rec["most_likely_answer"]["emb_tok_before_eos"] for rec in generations]

    # Stack -> [Batch, Layers, 1, Dim] -> Squeeze -> Transpose -> [Layers, Batch, Dim]
    def process_emb(raw_list):
        t = torch.stack(raw_list).squeeze(-2).transpose(0, 1).to(torch.float32)
        return t

    tbg = process_emb(tbg_raw)
    slt = process_emb(slt_raw)

    # Truncate
    n = min(n_samples, tbg.shape[1])
    tbg = tbg[:, :n, :]
    slt = slt[:, :n, :]
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

def get_split_indices(n_total, train_frac, val_frac, seed=42):
    """
    Generate deterministic indices for Train, Calibration (Val), and Test.
    """
    idxs = np.arange(n_total)
    
    # 1. Split (Train+Val) vs Test
    # test_frac is the remainder
    test_frac = 1.0 - (train_frac + val_frac)
    if test_frac < 0: raise ValueError("train_frac + val_frac > 1.0")
    
    # We prioritize keeping train_frac exact? 
    # Standard approach: split test off first.
    train_val_idx, test_idx = train_test_split(
        idxs, test_size=test_frac, random_state=seed
    )
    
    # 2. Split Train vs Val (Calibration)
    # The 'test_size' here is relative to the train_val chunk
    # relative_val = val_frac / (train_frac + val_frac)
    relative_val = val_frac / (1.0 - test_frac)
    
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val, random_state=seed
    )
    
    return train_idx, val_idx, test_idx


def train_per_layer_probes(dataset_tensor, y_target, idx_train, idx_val, idx_test):
    """
    Train a probe for every layer using the pre-computed indices.
    """
    # dataset_tensor: [Layers, N, Dim]
    n_layers = dataset_tensor.shape[0]
    
    X = dataset_tensor.numpy()
    y = y_target.numpy()

    aucs = []
    models = []
    
    for i in range(n_layers):
        X_layer = X[i]
        
        Xt = X_layer[idx_train]
        yt = y[idx_train]
        
        Xte = X_layer[idx_test]
        yte = y[idx_test]
        
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver='lbfgs', C=1.0) 
        )
        
        try:
            clf.fit(Xt, yt)
            
            # Evaluate
            if len(np.unique(yte)) > 1:
                # predict_proba on pipeline automatically applies scaling
                probs = clf.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, probs)
            else:
                auc = 0.5
        except Exception as e:
            print(f"Probe training failed at layer {i}: {e}")
            auc = 0.5
            
        aucs.append(auc)
        models.append(clf)
        
    return aucs, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--runs-root", type=str, required=True)
    # Optional output override, defaults to inside the run dir
    ap.add_argument("--out", type=str, default=None) 
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg["models"]
    datasets = cfg["datasets"]
    
    probe_cfg = cfg.get("probes", {})
    n_samples = probe_cfg.get("n_samples", 2000)
    entropy_key = probe_cfg.get("entropy_key", "cluster_assignment_entropy")
    
    # Read split config
    train_frac = probe_cfg.get("train_frac", 0.7)
    val_frac = probe_cfg.get("val_frac", 0.15)
    # Test is implicit remainder

    runs_root = Path(args.runs_root)
    all_Ds = []
    run_dirs = []
    files_dirs = []

    # 1) Load all datasets
    for model in models:
        for ds in datasets:
            run_dir = runs_root / f"{Path(model).name}__{ds}"
            if not run_dir.exists():
                print(f"Skipping {run_dir}, directory does not exist")
                continue

            files_dir = find_wandb_files_dir(run_dir)
            if files_dir is None:
                print(f"Skipping {run_dir}, no files dir")
                continue

            val_path = files_dir / "validation_generations.pkl"
            unc_path = files_dir / "uncertainty_measures.pkl"

            if not val_path.exists() or not unc_path.exists():
                print(f"Skipping {run_dir}, missing pickles")
                continue

            print(f"Loading {run_dir.name}")
            D = load_sep_dataset(
                val_path=val_path,
                measures_path=unc_path,
                n_samples=n_samples,
                entropy_key=entropy_key,
                name=f"{Path(model).name}__{ds}",
            )
            all_Ds.append(D)
            run_dirs.append(run_dir)
            files_dirs.append(files_dir)

    if not all_Ds:
        raise RuntimeError("No valid runs found.")

    # 2) Global best split for entropy
    all_entropy = torch.cat([D.entropy for D in all_Ds], dim=0)
    split_val = best_split(all_entropy)
    print(f"Best universal entropy split value: {split_val}")

    for D in all_Ds:
        D.b_entropy = binarize_entropy(D.entropy, split_val)

    # 3) Train and Save
    global_results = {}
    
    for D, files_dir in zip(all_Ds, files_dirs):
        print(f"Training probes for {D.name}")
        
        # A. Create Splits ONCE
        n_total = len(D.accuracies)
        idx_train, idx_val, idx_test = get_split_indices(n_total, train_frac, val_frac)
        
        print(f"  Splits: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
        
        res = {}
        
        # Save indices so notebook uses exactly the same rows
        res["splits"] = {
            "train": idx_train,
            "val": idx_val,
            "test": idx_test
        }

        # B. Train SE Probes (Target = binarized entropy)
        tb_auc, tb_models = train_per_layer_probes(
            D.tbg_dataset, D.b_entropy, idx_train, idx_val, idx_test
        )
        sb_auc, sb_models = train_per_layer_probes(
            D.slt_dataset, D.b_entropy, idx_train, idx_val, idx_test
        )

        # C. Train Accuracy Probes (Target = accuracy)
        # Note: We usually train to predict Correctness (1.0). 
        # Notebook handles inversion to P(Hallucination).
        ta_auc, ta_models = train_per_layer_probes(
            D.tbg_dataset, D.accuracies, idx_train, idx_val, idx_test
        )
        sa_auc, sa_models = train_per_layer_probes(
            D.slt_dataset, D.accuracies, idx_train, idx_val, idx_test
        )

        res["tb_aucs"] = tb_auc
        res["sb_aucs"] = sb_auc
        res["ta_aucs"] = ta_auc
        res["sa_aucs"] = sa_auc
        
        res["tb_models"] = tb_models
        res["sb_models"] = sb_models
        res["ta_models"] = ta_models
        res["sa_models"] = sa_models

        global_results[D.name] = res

        # Save per-run probes.pkl
        # Note: If --out is passed, we might overwrite, but usually we want per-run files.
        # The script arg --out is usually for a global file, but here we save per run.
        per_run_out = files_dir / "probes.pkl"
        with per_run_out.open("wb") as f:
            pickle.dump({
                "split_threshold": split_val, 
                "results": {D.name: res}
            }, f)
        print(f"  Saved to {per_run_out}")

if __name__ == "__main__":
    main()