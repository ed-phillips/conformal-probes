#!/usr/bin/env python3
"""
analyze_results.py

End-to-end analysis + plotting for:
- Hallucination detection AUROC (full + confident subset)
- Risk–coverage curves (heuristic 1D + 2D conformal frontier)
- Calibration plots (target vs realized risk)
- Coverage-at-fixed-risk tables (LaTeX + CSV)
- Probe-space acceptance-region figure (2D rectangle)

Key conventions (aligned with "accuracy probe" literature):
- y_hall = 1 means hallucination/incorrect; y_hall = 0 means correct.
- Accuracy probe outputs p_correct in [0,1] (higher = safer).
- SE probe outputs p_high_entropy in [0,1] (higher = riskier).
- Semantic entropy raw (se_raw) higher = riskier.
- Combined LR outputs p_halluc_combined in [0,1] (higher = riskier).

4-way split (no double dipping):
- train: train probes
- val: choose best probe layer(s)
- cal: conformal calibration / threshold selection ONLY
- test: evaluation ONLY

Optional: Use a one-sided upper confidence bound (Clopper–Pearson) on risk when
selecting thresholds/regions on calibration set (more defensible than using
raw empirical risk alone). If SciPy is unavailable, it falls back to empirical risk.
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration defaults
# -----------------------------

DEFAULT_TARGET_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-4b-it",
    "google/gemma-7b-it",
    "mistralai/Ministral-8B-Instruct-2410",
]

DEFAULT_TARGET_DATASETS = [
    "trivia_qa",
    # "bioasq",
    # "medical_o1",
]

# Pretty plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
plt.rcParams["font.family"] = "serif"

COLORS = {
    "Semantic Entropy": "#7f8c8d",
    "Accuracy Probe (1 - p_correct)": "#3498db",
    "SE Probe (p_high_entropy)": "#e67e22",
    "Combined (LR)": "#2ecc71",
    "Dual-Probe (2D)": "#9b59b6",
    "Ideal": "#000000",
}


# -----------------------------
# Utilities: paths + loading
# -----------------------------

def resolve_paths(args_runs_root: str) -> Tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "notebooks":
        repo_root = script_path.parent.parent
    else:
        repo_root = Path.cwd()

    runs_path = Path(args_runs_root)
    if not runs_path.is_absolute():
        runs_path = repo_root / runs_path
    return repo_root, runs_path


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


def load_pickles(run_dir: Path) -> Optional[Tuple[dict, dict]]:
    files_dir = run_dir / "files"
    if not files_dir.exists():
        try:
            candidates = list(run_dir.rglob("files"))
            if candidates:
                files_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
            else:
                files_dir = run_dir
        except Exception:
            return None

    p1 = files_dir / "validation_generations.pkl"
    p2 = files_dir / "uncertainty_measures.pkl"

    if not (p1.exists() and p2.exists()):
        return None

    with open(p1, "rb") as f:
        gens = pickle.load(f)
    with open(p2, "rb") as f:
        unc = pickle.load(f)

    return gens, unc


# -----------------------------
# Data alignment + feature extraction
# -----------------------------

def _get_entropy_array(unc: dict) -> np.ndarray:
    ent_keys = ["cluster_assignment_entropy", "semantic_entropy_sum_normalized"]
    raw_entropy = None
    for k in ent_keys:
        if "uncertainty_measures" in unc and k in unc["uncertainty_measures"]:
            raw_entropy = np.array(unc["uncertainty_measures"][k], dtype=np.float32)
            break
    if raw_entropy is None:
        raw_entropy = None
    return raw_entropy


def _get_embedding_stack(gen_values: list, embedding_key: str) -> torch.Tensor:
    """
    Expects each g["most_likely_answer"][embedding_key] to be a tensor-like object with
    shape compatible with stacking, typically [n_layers, 1, d] or [n_layers, d].
    Returns torch.Tensor with shape [n_layers, n_examples, d].
    """
    embs = []
    for g in gen_values:
        if "most_likely_answer" not in g or embedding_key not in g["most_likely_answer"]:
            raise KeyError(f"Missing embedding key {embedding_key} in most_likely_answer.")
        embs.append(g["most_likely_answer"][embedding_key])

    # Convert to torch tensors
    tlist = []
    for e in embs:
        if isinstance(e, torch.Tensor):
            t = e
        else:
            t = torch.tensor(e)
        tlist.append(t)

    stacked = torch.stack(tlist)  # [n, ...]
    # Common shapes seen:
    #   stacked: [n, n_layers, 1, d]  -> squeeze -> [n, n_layers, d]
    #   stacked: [n, n_layers, d]
    while stacked.ndim > 3:
        stacked = stacked.squeeze(-2)

    if stacked.ndim != 3:
        raise ValueError(f"Unexpected embedding tensor shape after squeeze: {tuple(stacked.shape)}")

    # stacked: [n_examples, n_layers, d] -> transpose to [n_layers, n_examples, d]
    return stacked.transpose(0, 1).contiguous()


def extract_features_aligned(
    gens: dict,
    unc: dict,
    embedding_key: str,
    n_cap: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: [n_layers, n, d] float32
      y_hall: [n] int (1=hallucination/incorrect)
      se_raw: [n] float32
    """
    gen_values = list(gens.values())
    n_total = len(gen_values)

    accuracies = np.array([g["most_likely_answer"]["accuracy"] for g in gen_values], dtype=np.float32)
    y_hall = (accuracies < 1.0).astype(np.int64)

    se_raw = _get_entropy_array(unc)
    if se_raw is None:
        se_raw = np.zeros_like(accuracies, dtype=np.float32)

    X_t = _get_embedding_stack(gen_values, embedding_key=embedding_key)  # [L, n, d]
    X = X_t.cpu().numpy().astype(np.float32)

    n = min(n_total, X.shape[1], len(y_hall), len(se_raw), n_cap)
    return X[:, :n, :], y_hall[:n], se_raw[:n]


# -----------------------------
# Splits + probe training
# -----------------------------

def make_4way_split(
    n: int,
    seed: int,
    frac_train: float = 0.55,
    frac_val: float = 0.15,
    frac_cal: float = 0.15,
    frac_test: float = 0.15,
) -> Dict[str, np.ndarray]:
    fracs = np.array([frac_train, frac_val, frac_cal, frac_test], dtype=np.float64)
    fracs = fracs / fracs.sum()

    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    n_cal = int(round(fracs[2] * n))
    # remainder to test
    n_test = n - (n_train + n_val + n_cal)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    cal_idx = idx[n_train + n_val:n_train + n_val + n_cal]
    test_idx = idx[n_train + n_val + n_cal:]

    assert len(test_idx) == n_test

    return {"train": train_idx, "val": val_idx, "cal": cal_idx, "test": test_idx}


def train_probe_across_layers(
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    C: float = 0.1,
    max_iter: int = 300,
) -> Tuple[make_pipeline, int, List[float]]:
    """
    Trains a separate logistic regression per layer, selects best by AUROC on val.
    y should be binary with higher = positive class.
    Returns:
      best_model, best_layer, val_aucs_per_layer
    """
    n_layers = X.shape[0]
    val_aucs: List[float] = []

    best_auc = -np.inf
    best_layer = 0
    best_model = None

    for l in range(n_layers):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"))
        pipe.fit(X[l][idx_train], y[idx_train])

        preds = pipe.predict_proba(X[l][idx_val])[:, 1]
        # AUROC undefined if only one class in y[idx_val]
        try:
            auc = roc_auc_score(y[idx_val], preds)
        except ValueError:
            auc = np.nan
        val_aucs.append(auc)

        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_layer = l
            best_model = pipe

    if best_model is None:
        # fall back: train on layer 0
        best_layer = 0
        best_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"))
        best_model.fit(X[best_layer][idx_train], y[idx_train])

    return best_model, best_layer, val_aucs


def process_model_dataset(
    X: np.ndarray,
    y_hall: np.ndarray,
    se_raw: np.ndarray,
    seed: int,
    entropy_binarize: str = "median_train",  # or "median_trainval"
) -> Tuple[pd.DataFrame, Dict[str, list], Dict[str, np.ndarray]]:
    """
    Produces a df with scores and split labels, plus layer stats and split indices.

    Accuracy probe: p_correct = P(correct=1 | h)
    SE probe: p_high_entropy = P(high entropy=1 | h)
    Combined LR: p_halluc_combined = P(hallucination=1 | [p_correct, p_high_entropy])
    """
    n = len(y_hall)
    splits = make_4way_split(n, seed=seed)
    idx_train, idx_val, idx_cal, idx_test = splits["train"], splits["val"], splits["cal"], splits["test"]

    # Targets
    y_correct = 1 - y_hall

    # Binarize semantic entropy to create "high entropy" target for SE probe
    if entropy_binarize == "median_train":
        thr = float(np.median(se_raw[idx_train]))
    elif entropy_binarize == "median_trainval":
        thr = float(np.median(se_raw[np.concatenate([idx_train, idx_val])]))
    else:
        raise ValueError("entropy_binarize must be median_train or median_trainval")

    y_high_ent = (se_raw > thr).astype(np.int64)

    layer_stats: Dict[str, list] = {"acc_val_aucs": [], "se_val_aucs": []}

    # 1) Train/select accuracy probe over layers (predict correct)
    acc_model, acc_layer, acc_val_aucs = train_probe_across_layers(
        X, y_correct, idx_train, idx_val
    )
    layer_stats["acc_val_aucs"] = acc_val_aucs

    # 2) Train/select entropy probe over layers (predict high entropy)
    se_model, se_layer, se_val_aucs = train_probe_across_layers(
        X, y_high_ent, idx_train, idx_val
    )
    layer_stats["se_val_aucs"] = se_val_aucs

    # Optionally refit probes on train+val after selecting layers (common practice)
    idx_trainval = np.concatenate([idx_train, idx_val])

    acc_model_refit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, solver="lbfgs"))
    acc_model_refit.fit(X[acc_layer][idx_trainval], y_correct[idx_trainval])

    se_model_refit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, solver="lbfgs"))
    se_model_refit.fit(X[se_layer][idx_trainval], y_high_ent[idx_trainval])

    # 3) Build df with splits + labels + raw entropy
    df = pd.DataFrame(index=np.arange(n))
    split_col = np.empty(n, dtype=object)
    split_col[:] = "unused"
    split_col[idx_train] = "train"
    split_col[idx_val] = "val"
    split_col[idx_cal] = "calibration"
    split_col[idx_test] = "test"
    df["split"] = split_col

    df["y_hall"] = y_hall.astype(np.int64)
    df["y_correct"] = y_correct.astype(np.int64)
    df["se_raw"] = se_raw.astype(np.float32)

    # 4) Scores from refit probes (single-pass deployable)
    df["p_correct"] = acc_model_refit.predict_proba(X[acc_layer])[:, 1].astype(np.float32)
    df["p_high_entropy"] = se_model_refit.predict_proba(X[se_layer])[:, 1].astype(np.float32)

    # 5) Combined LR trained ONLY on train+val (not on calibration/test)
    # Target is hallucination (y_hall)
    comb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, solver="lbfgs"))
    comb.fit(df.loc[idx_trainval, ["p_correct", "p_high_entropy"]], df.loc[idx_trainval, "y_hall"])
    df["p_halluc_combined"] = comb.predict_proba(df[["p_correct", "p_high_entropy"]])[:, 1].astype(np.float32)

    # For convenience in 2D selection (risk region), we use:
    # accept if p_correct >= t_acc and p_high_entropy <= t_se
    # (i.e., high accuracy + low entropy)
    layer_stats["acc_best_layer"] = acc_layer
    layer_stats["se_best_layer"] = se_layer

    return df, layer_stats, splits


# -----------------------------
# Risk–coverage curves + conformal-ish selection
# -----------------------------

def risk_coverage_curve_1d(scores: np.ndarray, y_hall: np.ndarray, safe_if_high: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a monotone risk–coverage curve by sorting examples by "safety".

    If safe_if_high=True, larger score = safer, so we sort descending and accept prefixes.
    If safe_if_high=False, smaller score = safer, so we sort ascending and accept prefixes.

    Returns:
      coverage array, risk array
    """
    scores = np.asarray(scores)
    y_hall = np.asarray(y_hall)

    order = np.argsort(scores)
    if safe_if_high:
        order = order[::-1]

    y_sorted = y_hall[order]
    n = len(y_sorted)

    cum_h = np.cumsum(y_sorted)
    accepted = np.arange(1, n + 1)
    risk = cum_h / accepted
    cov = accepted / n
    return cov, risk


def downsample_curve(cov: np.ndarray, risk: np.ndarray, n_points: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    if len(cov) <= n_points:
        return cov, risk
    idx = np.linspace(0, len(cov) - 1, n_points).astype(int)
    return cov[idx], risk[idx]


def try_clopper_pearson_ucb(k: int, n: int, delta: float) -> Optional[float]:
    """
    One-sided Clopper–Pearson upper confidence bound on Binomial proportion.
    Returns None if scipy isn't available.
    """
    try:
        from scipy.stats import beta  # type: ignore
    except Exception:
        return None

    if n <= 0:
        return 1.0
    if k < 0:
        k = 0
    if k > n:
        k = n

    # Upper bound for p with confidence 1 - delta:
    # If k == n, upper bound is 1.
    if k == n:
        return 1.0
    # Beta quantile
    return float(beta.ppf(1 - delta, k + 1, n - k))


def risk_ucb(y_hall: np.ndarray, accepted_mask: np.ndarray, delta: float, use_ucb: bool) -> float:
    m = int(accepted_mask.sum())
    if m == 0:
        return 1.0
    k = int(y_hall[accepted_mask].sum())

    if not use_ucb:
        return k / m

    ucb = try_clopper_pearson_ucb(k, m, delta=delta)
    if ucb is None:
        # fallback: empirical
        return k / m
    return ucb


def select_1d_threshold_max_coverage(
    cal_scores: np.ndarray,
    cal_y_hall: np.ndarray,
    alpha: float,
    safe_if_high: bool,
    delta: float,
    use_ucb: bool,
) -> float:
    """
    Picks a threshold on calibration set that maximizes coverage while keeping
    UCB(risk) <= alpha. Returns threshold.
    """
    cal_scores = np.asarray(cal_scores)
    cal_y_hall = np.asarray(cal_y_hall)

    # Sort by safety and consider all prefix accept sets
    order = np.argsort(cal_scores)
    if safe_if_high:
        order = order[::-1]

    y_sorted = cal_y_hall[order]
    s_sorted = cal_scores[order]

    best_cov = -1.0
    best_t = None

    cum_k = np.cumsum(y_sorted)
    for i in range(len(y_sorted)):
        m = i + 1
        k = int(cum_k[i])
        if use_ucb:
            ucb = try_clopper_pearson_ucb(k, m, delta=delta)
            if ucb is None:
                ucb = k / m
            feasible = (ucb <= alpha)
        else:
            feasible = ((k / m) <= alpha)

        if feasible:
            cov = m / len(y_sorted)
            if cov > best_cov:
                best_cov = cov
                best_t = s_sorted[i]  # accept scores at least this safe (or at most, depending)

    if best_t is None:
        # No feasible non-empty acceptance; return extreme threshold (accept nothing)
        # Caller will handle.
        return np.inf if not safe_if_high else -np.inf

    return float(best_t)


def conformal_1d_eval(
    df: pd.DataFrame,
    score_col: str,
    alpha: float,
    safe_if_high: bool,
    delta: float,
    use_ucb: bool,
) -> Tuple[float, float, float]:
    """
    Uses calibration set to pick threshold; evaluates realized risk + coverage on test.
    Returns (target, realized_risk, coverage).
    """
    cal = df[df["split"] == "calibration"]
    test = df[df["split"] == "test"]

    t = select_1d_threshold_max_coverage(
        cal[score_col].values,
        cal["y_hall"].values,
        alpha=alpha,
        safe_if_high=safe_if_high,
        delta=delta,
        use_ucb=use_ucb,
    )

    if safe_if_high:
        accept = test[score_col].values >= t
    else:
        accept = test[score_col].values <= t

    realized_risk = float(test.loc[accept, "y_hall"].mean()) if accept.sum() > 0 else 0.0
    coverage = float(accept.mean())
    return alpha, realized_risk, coverage


def find_best_2d_region(
    cal_df: pd.DataFrame,
    alpha: float,
    steps: int,
    delta: float,
    use_ucb: bool,
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Accept region: p_correct >= t_acc  AND  p_high_entropy <= t_se

    Grid search over thresholds to maximize coverage under (UCB) risk constraint.
    Returns (t_acc, t_se, best_cov). If no feasible region, returns (None, None, 0).
    """
    grid = np.linspace(0.0, 1.0, steps)

    best_cov = 0.0
    best_t_acc = None
    best_t_se = None

    pc = cal_df["p_correct"].values
    pe = cal_df["p_high_entropy"].values
    y = cal_df["y_hall"].values

    n = len(cal_df)

    for t_acc in grid:
        # prefilter p_correct >= t_acc
        mask_acc = (pc >= t_acc)
        if mask_acc.sum() == 0:
            continue

        for t_se in grid:
            mask = mask_acc & (pe <= t_se)
            m = int(mask.sum())
            if m == 0:
                continue

            # risk criterion on cal
            if use_ucb:
                ucb = risk_ucb(y, mask, delta=delta, use_ucb=True)
                feasible = (ucb <= alpha)
            else:
                feasible = (float(y[mask].mean()) <= alpha)

            if feasible:
                cov = m / n
                if cov > best_cov:
                    best_cov = cov
                    best_t_acc = float(t_acc)
                    best_t_se = float(t_se)

    return best_t_acc, best_t_se, best_cov


def conformal_2d_eval(
    df: pd.DataFrame,
    alpha: float,
    steps: int,
    delta: float,
    use_ucb: bool,
) -> Tuple[float, float, float, Optional[float], Optional[float]]:
    """
    Calibrate 2D rectangle on calibration; evaluate on test.
    Returns (target, realized_risk, coverage, t_acc, t_se)
    """
    cal = df[df["split"] == "calibration"]
    test = df[df["split"] == "test"]

    t_acc, t_se, _ = find_best_2d_region(cal, alpha=alpha, steps=steps, delta=delta, use_ucb=use_ucb)
    if t_acc is None or t_se is None:
        return alpha, 0.0, 0.0, None, None

    accept = (test["p_correct"].values >= t_acc) & (test["p_high_entropy"].values <= t_se)
    realized_risk = float(test.loc[accept, "y_hall"].mean()) if accept.sum() > 0 else 0.0
    coverage = float(accept.mean())
    return alpha, realized_risk, coverage, t_acc, t_se


# -----------------------------
# Metrics
# -----------------------------

def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """
    Returns AUROC if defined; else np.nan.
    """
    try:
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return float("nan")


def confident_subset(df_test: pd.DataFrame, q: float = 0.30) -> pd.DataFrame:
    thr = df_test["se_raw"].quantile(q)
    return df_test[df_test["se_raw"] <= thr]


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_detection_bars(df_det: pd.DataFrame, figures_dir: Path) -> None:
    for model in df_det["Model"].unique():
        for subset in ["Full Test", "Confident Subset"]:
            data = df_det[(df_det["Model"] == model) & (df_det["Subset"] == subset)].copy()
            if data.empty:
                continue

            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=data,
                x="Dataset",
                y="AUROC",
                hue="Method",
                palette=COLORS,
                edgecolor="black",
                errorbar=None,
            )
            plt.title(f"{model} — Detection AUROC ({subset})")
            plt.ylim(0.4, 1.0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            out = figures_dir / f"detection_{model}_{subset.replace(' ', '_')}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def plot_risk_coverage(curve_data: dict, figures_dir: Path) -> None:
    for model, datasets in curve_data.items():
        for ds_name, methods_data in datasets.items():
            if not methods_data:
                continue
            plt.figure(figsize=(8.5, 6.5))
            for m_name, (covs, risks) in methods_data.items():
                pairs = sorted(zip(covs, risks))
                c, r = zip(*pairs)
                plt.plot(c, r, label=m_name, color=COLORS.get(m_name, "black"), linewidth=2)

            plt.xlabel("Coverage (fraction answered)")
            plt.ylabel("Risk (hallucination rate)")
            plt.title(f"Risk–Coverage: {model} / {ds_name}")
            plt.legend()
            plt.xlim(0, 1.0)
            plt.ylim(0, 0.6)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out = figures_dir / f"risk_coverage_{model}_{ds_name}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def plot_calibration(df_cal: pd.DataFrame, figures_dir: Path) -> None:
    for model in df_cal["Model"].unique():
        plt.figure(figsize=(7.2, 7.2))
        data = df_cal[df_cal["Model"] == model].copy()
        if data.empty:
            continue

        sns.lineplot(
            data=data,
            x="Target",
            y="Realized",
            hue="Method",
            style="Dataset",
            markers=True,
            dashes=False,
            palette=COLORS,
        )
        plt.plot([0, data["Target"].max()], [0, data["Target"].max()], "k--", alpha=0.5, label="Ideal")
        plt.xlabel("Target risk (α)")
        plt.ylabel("Realized risk (test)")
        plt.title(f"Calibration: {model}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out = figures_dir / f"calibration_{model}.png"
        plt.savefig(out, dpi=300)
        plt.close()


def plot_layer_sensitivity(layer_results: list, figures_dir: Path) -> None:
    for item in layer_results:
        acc_aucs = item["Acc_AUCs"]
        se_aucs = item["SE_AUCs"]
        L = len(acc_aucs)

        x = np.linspace(0, 1, L)

        plt.figure(figsize=(8.5, 5.2))
        plt.plot(x, acc_aucs, label="Accuracy probe val AUROC", color="#3498db", linewidth=2)
        plt.plot(x, se_aucs, label="SE probe val AUROC", color="#e67e22", linestyle="--", linewidth=2)
        plt.xlabel("Layer depth (fraction)")
        plt.ylabel("AUROC (val)")
        plt.title(f"Layer sensitivity: {item['Model']} / {item['Dataset']}")
        plt.legend()
        plt.tight_layout()
        out = figures_dir / f"layers_{item['Model']}_{item['Dataset']}.png"
        plt.savefig(out, dpi=300)
        plt.close()


def plot_probe_space_region(
    df: pd.DataFrame,
    model: str,
    dataset: str,
    alpha: float,
    outpath: Path,
    steps: int,
    delta: float,
    use_ucb: bool,
) -> None:
    from matplotlib.patches import Rectangle

    cal = df[df["split"] == "calibration"].copy()
    if cal.empty:
        return

    t_acc, t_se, cov = find_best_2d_region(cal, alpha=alpha, steps=steps, delta=delta, use_ucb=use_ucb)
    if t_acc is None or t_se is None:
        print(f"[warn] No feasible 2D region for {model}/{dataset} at alpha={alpha:.2f}")
        return
    else:
        print(f"[info] 2D region for {model}/{dataset} at alpha={alpha:.2f}: t_acc={t_acc:.3f}, t_se={t_se:.3f}, cal_cov={cov:.3f}")

    y = cal["y_hall"].values
    pc = cal["p_correct"].values
    pe = cal["p_high_entropy"].values

    plt.figure(figsize=(7.0, 6.2))
    plt.scatter(pc[y == 0], pe[y == 0], alpha=0.6, s=25, label="Correct (y=0)")
    plt.scatter(pc[y == 1], pe[y == 1], alpha=0.6, s=25, label="Hallucination (y=1)")

    # Acceptance region: pc >= t_acc AND pe <= t_se
    # That's a top-left rectangle: x in [t_acc, 1], y in [0, t_se]
    rect = Rectangle((t_acc, 0.0), 1.0 - t_acc, t_se, fill=False, linewidth=2)
    plt.gca().add_patch(rect)
    plt.axvline(t_acc, linestyle="--", linewidth=1)
    plt.axhline(t_se, linestyle="--", linewidth=1)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("p_correct (accuracy probe)")
    plt.ylabel("p_high_entropy (SE probe)")
    plt.title(f"{model} / {dataset} — 2D region α={alpha:.2f} (cal cov={cov:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def latex_escape(s: str) -> str:
    if s is None:
        return s
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def pick_best_method(block: pd.DataFrame, alpha: float) -> str:
    """
    block columns expected: Method, Coverage, Realized
    returns method name to bold (risk value) for this model+alpha
    """
    b = block.dropna(subset=["Realized", "Coverage"]).copy()
    if b.empty:
        return None

    below = b[b["Realized"] <= alpha].copy()
    if not below.empty:
        # closest below target = max risk under alpha; tie-break by coverage
        below["risk_key"] = below["Realized"]  # maximize
        below = below.sort_values(["risk_key", "Coverage"], ascending=[False, False])
        return below.iloc[0]["Method"]

    # no one meets target: pick smallest realized risk; tie-break by coverage
    b = b.sort_values(["Realized", "Coverage"], ascending=[True, False])
    return b.iloc[0]["Method"]

def write_conformal_tables_pretty(
    df_cal: pd.DataFrame,
    figures_dir: Path,
    table_alphas: List[float],
) -> None:
    """
    Writes one LaTeX tabular per dataset:
      conformal_table_<dataset>.tex

    Layout:
      - One table per dataset
      - Columns: for each alpha: Cov + Risk
      - Rows: grouped by Model; within each model, list methods (5 fixed methods)
      - Bolding (per model, per alpha):
          * Coverage cell bolds: highest coverage among methods with realized risk <= alpha
          * Risk cell bolds: realized risk that is <= alpha and closest to alpha (i.e., max risk under alpha)

    If no method satisfies risk <= alpha for a given (model, alpha), then no bolding occurs for that alpha.
    """

    methods_keep = [
        "Semantic Entropy",
        "SE Probe (p_high_entropy)",
        "Accuracy Probe (1 - p_correct)",
        "Combined (LR)",
        "Dual-Probe (2D)",
    ]
    method_order = {m: i for i, m in enumerate(methods_keep)}

    df = df_cal.copy()
    df["Target"] = df["Target"].round(3)
    keep = [round(a, 3) for a in table_alphas]

    df = df[df["Target"].isin(keep)].copy()
    df = df[df["Method"].isin(methods_keep)].copy()
    if df.empty:
        return

    # Stable ordering
    df["method_rank"] = df["Method"].map(method_order)
    df = df.sort_values(["Dataset", "Model", "method_rank", "Target"]).drop(columns=["method_rank"])

    # Ensure output dir exists
    figures_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["Dataset"].unique()):
        dfd = df[df["Dataset"] == dataset].copy()
        if dfd.empty:
            continue

        alphas = sorted(set(dfd["Target"].tolist()))
        n_alpha = len(alphas)
        col_spec = "l" + "cc" * n_alpha  # Method + (Cov,Risk)*K

        lines = []
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")

        # Header row 1: multicolumn alphas
        hdr1 = [r"\textbf{Method}"]
        for a in alphas:
            hdr1.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{$\alpha={a:.2f}$}}}}")
        lines.append(" & ".join(hdr1) + r" \\")

        # cmidrules for each alpha block
        cmids = []
        start = 2
        for _ in alphas:
            cmids.append(rf"\cmidrule(lr){{{start}-{start+1}}}")
            start += 2
        lines.append(" ".join(cmids))

        # Header row 2: subcolumns
        hdr2 = [r"\textbf{Method}"]
        for _ in alphas:
            hdr2.extend([r"\textbf{Cov}", r"\textbf{Risk}"])
        lines.append(" & ".join(hdr2) + r" \\")
        lines.append(r"\midrule")

        # Helper: pick bolding winners for a given model+alpha
        def compute_bold_winners(df_block: pd.DataFrame, alpha: float) -> Dict[str, Optional[str]]:
            """
            df_block has rows for one (model, alpha) with columns Method, Coverage, Realized.
            Returns:
              {
                "best_cov_method": <method or None>,
                "best_risk_method": <method or None>
              }
            """
            b = df_block.dropna(subset=["Coverage", "Realized"]).copy()
            eligible = b[b["Realized"] <= alpha].copy()
            if eligible.empty:
                return {"best_cov_method": None, "best_risk_method": None}

            # Best coverage among eligible; tie-break by higher realized risk (closer to alpha), then method order.
            eligible["method_rank"] = eligible["Method"].map(method_order).fillna(9999).astype(int)
            elig_cov = eligible.sort_values(
                ["Coverage", "Realized", "method_rank"],
                ascending=[False, False, True],
            )
            best_cov_method = elig_cov.iloc[0]["Method"]

            # Best (closest below alpha) risk among eligible; tie-break by higher coverage, then method order.
            elig_risk = eligible.sort_values(
                ["Realized", "Coverage", "method_rank"],
                ascending=[False, False, True],
            )
            best_risk_method = elig_risk.iloc[0]["Method"]

            return {"best_cov_method": best_cov_method, "best_risk_method": best_risk_method}

        # Body: group by model
        for model in sorted(dfd["Model"].unique()):
            dfm = dfd[dfd["Model"] == model].copy()
            if dfm.empty:
                continue

            # Model header spanning all columns
            model_esc = latex_escape(model)
            lines.append(rf"\multicolumn{{{1 + 2*n_alpha}}}{{l}}{{\textbf{{{model_esc}}}}} \\")
            lines.append(r"\addlinespace[2pt]")

            # Precompute bold winners per alpha for this model
            winners = {}
            for a in alphas:
                blk = dfm[dfm["Target"] == a][["Method", "Coverage", "Realized"]]
                winners[a] = compute_bold_winners(blk, alpha=float(a))

            # Method rows (fixed order)
            for method in methods_keep:
                row = [latex_escape(method)]
                df_method = dfm[dfm["Method"] == method]

                for a in alphas:
                    cell = df_method[df_method["Target"] == a]
                    if cell.empty:
                        row.extend(["-", "-"])
                        continue

                    cov = float(cell["Coverage"].iloc[0])
                    risk = float(cell["Realized"].iloc[0])

                    cov_s = f"{cov:.3f}"
                    risk_s = f"{risk:.3f}"

                    if winners[a]["best_cov_method"] == method:
                        cov_s = rf"\textbf{{{cov_s}}}"
                    if winners[a]["best_risk_method"] == method:
                        risk_s = rf"\textbf{{{risk_s}}}"

                    row.extend([cov_s, risk_s])

                lines.append(" & ".join(row) + r" \\")

            lines.append(r"\addlinespace[4pt]")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        ds_safe = dataset.replace("/", "_")
        out_path = figures_dir / f"conformal_table_{ds_safe}.tex"
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, required=True)

    parser.add_argument("--models", type=str, nargs="*", default=DEFAULT_TARGET_MODELS)
    parser.add_argument("--datasets", type=str, nargs="*", default=DEFAULT_TARGET_DATASETS)

    parser.add_argument("--n-cap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embedding-key", type=str, default="emb_last_tok_before_gen",
                        help="Key under most_likely_answer used for hidden states. "
                             "Try emb_last_tok_before_gen (TBG) or your SLT key if available.")
    parser.add_argument("--figures-dirname", type=str, default="paper_figures_fixed_v2")

    # Conformal-ish selection params
    parser.add_argument("--use-ucb", action="store_true",
                        help="Use one-sided Clopper–Pearson upper bound on risk when selecting thresholds/regions.")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Confidence level for UCB risk bound (smaller=more conservative). Used only if --use-ucb.")
    parser.add_argument("--grid-steps-2d", type=int, default=61)

    # Which alphas to include in the paper-style table
    parser.add_argument("--table-alphas", type=float, nargs="*", default=[0.05, 0.10, 0.20, 0.25])

    # Representative probe-space plot (can override)
    parser.add_argument("--probe-plot-model", type=str, default="google/gemma-7b-it")
    parser.add_argument("--probe-plot-dataset", type=str, default="trivia_qa")
    parser.add_argument("--probe-plot-alpha", type=float, default=0.05)

    args = parser.parse_args()

    repo_root, runs_root = resolve_paths(args.runs_root)
    figures_dir = repo_root / args.figures_dirname
    figures_dir.mkdir(exist_ok=True, parents=True)

    target_models = args.models
    target_datasets = args.datasets

    print(f"Scanning runs: {runs_root}")
    print(f"Saving figures/tables to: {figures_dir}")
    if args.use_ucb:
        print(f"Using UCB risk bound with delta={args.delta} (falls back to empirical if SciPy unavailable).")

    det_results = []
    cal_results = []
    layer_results = []
    curve_data: Dict[str, Dict[str, Dict[str, Tuple[list, list]]]] = {}

    # Keep dfs for optional probe-space plot
    dfs_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}

    for model_name in target_models:
        safe_name = Path(model_name).name
        curve_data[safe_name] = {}

        for dataset_name in target_datasets:
            curve_data[safe_name][dataset_name] = {}

            run_dir = find_run_directory(runs_root, model_name, dataset_name)
            if not run_dir:
                continue

            pickles = load_pickles(run_dir)
            if not pickles:
                continue

            print(f"Processing {safe_name} / {dataset_name} ...")

            try:
                X, y_hall, se_raw = extract_features_aligned(
                    *pickles,
                    embedding_key=args.embedding_key,
                    n_cap=args.n_cap
                )
            except Exception as e:
                print(f"  [fail] feature extraction: {e}")
                continue

            try:
                df, l_stats, splits = process_model_dataset(X, y_hall, se_raw, seed=args.seed)
            except Exception as e:
                print(f"  [fail] process_model_dataset: {e}")
                continue

            dfs_by_key[(safe_name, dataset_name)] = df

            # --- Layer stats plot data ---
            layer_results.append({
                "Model": safe_name,
                "Dataset": dataset_name,
                "Acc_AUCs": l_stats["acc_val_aucs"],
                "SE_AUCs": l_stats["se_val_aucs"],
            })

            # --- Detection metrics (hallucination detection AUROC) ---
            df_test = df[df["split"] == "test"].copy()
            df_conf = confident_subset(df_test, q=0.30)

            # For hallucination detection AUROC, use "risky score" where higher = more hallucination-prone.
            # - Semantic entropy: higher entropy -> riskier
            # - Accuracy probe: risk = 1 - p_correct
            # - SE probe: risk = p_high_entropy
            # - Combined LR: risk = p_halluc_combined
            methods_risky = {
                "Semantic Entropy": ("se_raw", True),  # True means higher = riskier
                "Accuracy Probe (1 - p_correct)": ("p_correct", False),  # higher is safer; risky is (1 - p_correct)
                "SE Probe (p_high_entropy)": ("p_high_entropy", True),
                "Combined (LR)": ("p_halluc_combined", True),
            }

            for m_name, (col, higher_is_risky) in methods_risky.items():
                if m_name == "Accuracy Probe (1 - p_correct)":
                    risky_full = 1.0 - df_test[col].values
                    risky_conf = 1.0 - df_conf[col].values
                else:
                    risky_full = df_test[col].values
                    risky_conf = df_conf[col].values

                det_results.append({
                    "Model": safe_name,
                    "Dataset": dataset_name,
                    "Method": m_name,
                    "Subset": "Full Test",
                    "AUROC": safe_auc(df_test["y_hall"].values, risky_full),
                })
                det_results.append({
                    "Model": safe_name,
                    "Dataset": dataset_name,
                    "Method": m_name,
                    "Subset": "Confident Subset",
                    "AUROC": safe_auc(df_conf["y_hall"].values, risky_conf),
                })

            # --- Risk–coverage curves (heuristic 1D) ---
            # Safety direction:
            # - se_raw: safe if LOW
            # - (1 - p_correct) risky; equivalently safe if HIGH p_correct
            # - p_high_entropy: safe if LOW
            # - p_halluc_combined: safe if LOW
            curves = {
                "Semantic Entropy": (df_test["se_raw"].values, False),
                "Accuracy Probe (1 - p_correct)": (df_test["p_correct"].values, True),
                "SE Probe (p_high_entropy)": (df_test["p_high_entropy"].values, False),
                "Combined (LR)": (df_test["p_halluc_combined"].values, False),
            }
            for m_name, (scores, safe_if_high) in curves.items():
                cov, risk = risk_coverage_curve_1d(scores=scores, y_hall=df_test["y_hall"].values, safe_if_high=safe_if_high)
                cov, risk = downsample_curve(cov, risk, n_points=120)
                curve_data[safe_name][dataset_name][m_name] = (list(cov), list(risk))

            # --- Dual-probe 2D curve (sweep alpha) ---
            covs_2d, risks_2d = [], []
            alphas_curve = np.linspace(0.005, 0.6, 32)
            for a in alphas_curve:
                _, r, c, _, _ = conformal_2d_eval(
                    df,
                    alpha=float(a),
                    steps=args.grid_steps_2d,
                    delta=args.delta,
                    use_ucb=args.use_ucb,
                )
                covs_2d.append(c)
                risks_2d.append(r)
            curve_data[safe_name][dataset_name]["Dual-Probe (2D)"] = (covs_2d, risks_2d)

            # --- Calibration (target vs realized, plus coverage) ---
            # Use the same conformal-ish selection for:
            # - entropy-only (safe if LOW)
            # - accuracy-only (safe if HIGH)
            # - se-probe-only (safe if LOW)
            # - combined score (safe if LOW)
            alphas_cal = np.linspace(0.01, 0.5, 20)
            # ensure table alphas are included
            alphas_cal = np.unique(np.concatenate([alphas_cal, np.array(args.table_alphas)]))

            cal_methods = [
                ("Semantic Entropy", "se_raw", False),
                ("Accuracy Probe (1 - p_correct)", "p_correct", True),
                ("SE Probe (p_high_entropy)", "p_high_entropy", False),
                ("Combined (LR)", "p_halluc_combined", False),
            ]

            for a in alphas_cal:
                for m_name, col, safe_if_high in cal_methods:
                    tgt, realized, cov = conformal_1d_eval(
                        df=df,
                        score_col=col,
                        alpha=float(a),
                        safe_if_high=safe_if_high,
                        delta=args.delta,
                        use_ucb=args.use_ucb,
                    )
                    cal_results.append({
                        "Model": safe_name,
                        "Dataset": dataset_name,
                        "Method": m_name,
                        "Target": tgt,
                        "Realized": realized,
                        "Coverage": cov,
                    })

                # Dual-probe conformal-ish
                tgt, realized, cov, _, _ = conformal_2d_eval(
                    df=df,
                    alpha=float(a),
                    steps=args.grid_steps_2d,
                    delta=args.delta,
                    use_ucb=args.use_ucb,
                )
                cal_results.append({
                    "Model": safe_name,
                    "Dataset": dataset_name,
                    "Method": "Dual-Probe (2D)",
                    "Target": tgt,
                    "Realized": realized,
                    "Coverage": cov,
                })

    if not det_results:
        print("No results found. Check --runs-root, models, datasets, or pickle structure.")
        return

    # -----------------------------
    # Build summary frames
    # -----------------------------
    df_det = pd.DataFrame(det_results)
    df_cal = pd.DataFrame(cal_results)

    df_det.to_csv(figures_dir / "detection_auroc_summary.csv", index=False)
    df_cal.to_csv(figures_dir / "calibration_summary.csv", index=False)

    # -----------------------------
    # Plots
    # -----------------------------
    print("Generating plots...")
    plot_detection_bars(df_det, figures_dir)
    plot_risk_coverage(curve_data, figures_dir)
    plot_calibration(df_cal, figures_dir)
    plot_layer_sensitivity(layer_results, figures_dir)

    # -----------------------------
    # Tables (coverage at {0.01, 0.05, 0.10})
    # -----------------------------
    write_conformal_tables_pretty(df_cal, figures_dir, table_alphas=args.table_alphas)

    # -----------------------------
    # Probe-space region figure (one representative setting)
    # -----------------------------
    rep_model = Path(args.probe_plot_model).name
    rep_ds = args.probe_plot_dataset
    key = (rep_model, rep_ds)
    if key in dfs_by_key:
        out = figures_dir / f"probe_space_region_{rep_model}_{rep_ds}_a{args.probe_plot_alpha:.2f}.png"
        plot_probe_space_region(
            df=dfs_by_key[key],
            model=rep_model,
            dataset=rep_ds,
            alpha=float(args.probe_plot_alpha),
            outpath=out,
            steps=args.grid_steps_2d,
            delta=args.delta,
            use_ucb=args.use_ucb,
        )
    else:
        print(f"[warn] Representative probe-space plot key not found: {key}")

    print("Done.")
    print(f"- Figures: {figures_dir}")
    print(f"- LaTeX tables: {figures_dir / 'conformal_coverage_table.tex'} and {figures_dir / 'conformal_coverage_risk_table.tex'}")


if __name__ == "__main__":
    main()
