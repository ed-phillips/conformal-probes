#!/usr/bin/env python3
"""
analyze_results.py

End-to-end analysis + plotting for Hallucination Detection.

Features:
- Hallucination Detection AUROC (Bar Charts per Model).
- Layer Sensitivity Analysis.
- Risk-Coverage Curves (Dynamic Y-limits).
- Decision Boundary Visualization.
- 1D Conformal Risk Control (LR, MLP, GBM).
- LaTeX Table Generation.

Usage:
  python analyze_results.py --runs-root /path/to/runs
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# -----------------------------
# Configuration
# -----------------------------

DEFAULT_TARGET_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "google/gemma-3-4b-it",
    "google/gemma-7b-it",
    "HuggingFaceTB/SmolLM3-3B",
    "mistralai/Ministral-8B-Instruct-2410",
]

DEFAULT_TARGET_DATASETS = [
    "trivia_qa",
    "bioasq",
    "medical_o1",
]

# Plotting Aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams["font.family"] = "serif"

COLORS = {
    "Semantic Entropy": "#7f8c8d",          # Grey
    "Accuracy Probe": "#3498db",            # Blue
    "SE Probe": "#e67e22",                  # Orange
    "Combined (LR)": "#2ecc71",             # Green
    "Combined (MLP)": "#9b59b6",            # Purple
    "Combined (GBM)": "#e74c3c",            # Red
}

METHOD_ORDER = [
    "Semantic Entropy",
    "Accuracy Probe",
    "SE Probe",
    "Combined (LR)",
    "Combined (MLP)",
    "Combined (GBM)"
]

# -----------------------------
# Utilities: Paths & Loading
# -----------------------------

def resolve_paths(args_runs_root: str) -> Tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent if script_path.parent.name == "notebooks" else Path.cwd()
    runs_path = Path(args_runs_root)
    if not runs_path.is_absolute():
        runs_path = repo_root / runs_path
    return repo_root, runs_path

def find_run_directory(root_path: Path, model_name: str, dataset_name: str) -> Optional[Path]:
    safe_model_name = Path(model_name).name
    target_folder_name = f"{safe_model_name}__{dataset_name}"
    candidates = []
    
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
        candidates = list(run_dir.rglob("files"))
        if candidates:
            files_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        else:
            files_dir = run_dir

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
# Data Extraction
# -----------------------------

def _get_entropy_array(unc: dict) -> Optional[np.ndarray]:
    ent_keys = ["cluster_assignment_entropy", "semantic_entropy_sum_normalized"]
    for k in ent_keys:
        if "uncertainty_measures" in unc and k in unc["uncertainty_measures"]:
            return np.array(unc["uncertainty_measures"][k], dtype=np.float32)
    return None

def _get_embedding_stack(gen_values: list, embedding_key: str) -> torch.Tensor:
    embs = []
    for g in gen_values:
        if "most_likely_answer" not in g or embedding_key not in g["most_likely_answer"]:
            raise KeyError(f"Missing embedding key {embedding_key}")
        embs.append(g["most_likely_answer"][embedding_key])

    tlist = [e if isinstance(e, torch.Tensor) else torch.tensor(e) for e in embs]
    stacked = torch.stack(tlist)
    while stacked.ndim > 3:
        stacked = stacked.squeeze(-2)
    return stacked.transpose(0, 1).contiguous()

def extract_features_aligned(
    gens: dict, unc: dict, embedding_key: str, n_cap: int = 2000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen_values = list(gens.values())
    n_total = len(gen_values)

    # Ensure accuracy is float for comparison
    accuracies = np.array([float(g["most_likely_answer"]["accuracy"]) for g in gen_values], dtype=np.float32)
    
    # y_hall = 1 if Accuracy < 1.0 (Incorrect), else 0
    y_hall = (accuracies < 0.99).astype(np.int64)

    se_raw = _get_entropy_array(unc)
    if se_raw is None:
        se_raw = np.zeros_like(accuracies, dtype=np.float32)

    X_t = _get_embedding_stack(gen_values, embedding_key)
    X = X_t.cpu().numpy().astype(np.float32)

    n = min(n_total, X.shape[1], len(y_hall), len(se_raw), n_cap)
    return X[:, :n, :], y_hall[:n], se_raw[:n]

# -----------------------------
# Training & Processing
# -----------------------------

def make_4way_split(n: int, seed: int) -> Dict[str, np.ndarray]:
    fracs = np.array([0.55, 0.15, 0.15, 0.15])
    fracs /= fracs.sum()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    n_cal = int(round(fracs[2] * n))
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train+n_val],
        "cal": idx[n_train+n_val : n_train+n_val+n_cal],
        "test": idx[n_train+n_val+n_cal:]
    }

def train_probe_layer_selection(X, y, idx_train, idx_val, C=0.1):
    n_layers = X.shape[0]
    best_auc = -1.0
    best_layer = 0
    val_aucs = []

    for l in range(n_layers):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
        # Edge case: single class in train/val
        if len(np.unique(y[idx_train])) < 2:
            val_aucs.append(0.5)
            continue

        pipe.fit(X[l][idx_train], y[idx_train])
        preds = pipe.predict_proba(X[l][idx_val])[:, 1]
        
        try:
            auc = roc_auc_score(y[idx_val], preds)
        except:
            auc = 0.5
        val_aucs.append(auc)
        
        if auc > best_auc:
            best_auc = auc
            best_layer = l

    idx_tv = np.concatenate([idx_train, idx_val])
    if len(np.unique(y[idx_tv])) < 2:
        final_model = None
    else:
        final_model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
        final_model.fit(X[best_layer][idx_tv], y[idx_tv])
    
    return final_model, best_layer, val_aucs

def process_model_dataset(
    X: np.ndarray, y_hall: np.ndarray, se_raw: np.ndarray, seed: int
) -> Tuple[pd.DataFrame, Dict, Dict]:
    
    n = len(y_hall)
    splits = make_4way_split(n, seed)
    idx_train = splits["train"]
    idx_val = splits["val"]
    
    y_correct = 1 - y_hall
    # Use median of Training set for SE Probe target
    thr = np.median(se_raw[idx_train])
    y_high_ent = (se_raw > thr).astype(np.int64)

    # 1. Accuracy Probe
    acc_model, acc_layer, acc_aucs = train_probe_layer_selection(X, y_correct, idx_train, idx_val)
    
    # 2. SE Probe
    se_model, se_layer, se_aucs = train_probe_layer_selection(X, y_high_ent, idx_train, idx_val)

    df = pd.DataFrame(index=np.arange(n))
    df["split"] = "unused"
    for k, v in splits.items():
        df.loc[v, "split"] = k
        
    df["y_hall"] = y_hall
    df["y_correct"] = y_correct
    df["se_raw"] = se_raw

    # Inference
    if acc_model:
        df["p_correct"] = acc_model.predict_proba(X[acc_layer])[:, 1].astype(np.float32)
    else:
        df["p_correct"] = 0.5

    if se_model:
        df["p_high_entropy"] = se_model.predict_proba(X[se_layer])[:, 1].astype(np.float32)
    else:
        df["p_high_entropy"] = 0.5

    # 4. Train Combiners on Train+Val
    idx_trainval = np.concatenate([idx_train, idx_val])
    X_comb = df.loc[idx_trainval, ["p_correct", "p_high_entropy"]]
    y_comb = df.loc[idx_trainval, "y_hall"]

    combiners = {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)),
        # "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
    }

    for name, model in combiners.items():
        if len(np.unique(y_comb)) > 1:
            model.fit(X_comb, y_comb)
            # Predict Prob of Hallucination (Class 1)
            df[f"p_halluc_{name}"] = model.predict_proba(df[["p_correct", "p_high_entropy"]])[:, 1].astype(np.float32)
        else:
            df[f"p_halluc_{name}"] = 0.5

    layer_stats = {
        "acc_aucs": acc_aucs, "se_aucs": se_aucs, 
        "acc_best": acc_layer, "se_best": se_layer
    }
    
    return df, layer_stats, splits

# -----------------------------
# Conformal & Metrics
# -----------------------------

def try_clopper_pearson_ucb(k: int, n: int, delta: float) -> float:
    try:
        from scipy.stats import beta
        if k == n: return 1.0
        return float(beta.ppf(1 - delta, k + 1, n - k))
    except:
        return k / n

def select_threshold_1d(scores, labels, alpha, delta, use_ucb):
    """
    Selects threshold t such that empirical risk (or UCB) of accepted set is <= alpha.
    """
    # 1. Sort: Low Score (Safe) -> High Score (Risky)
    order = np.argsort(scores)
    y_sorted = labels[order]
    s_sorted = scores[order]
    
    # 2. Compute Risk Profile
    cum_failures = np.cumsum(y_sorted).astype(float)
    counts = np.arange(1, len(y_sorted) + 1).astype(float)
    
    if use_ucb:
        risks = np.array([try_clopper_pearson_ucb(int(k), int(n), delta) for k, n in zip(cum_failures, counts)])
    else:
        risks = cum_failures / counts

    # 3. Find Feasible Points
    feasible = risks <= alpha
    
    # 4. Select Threshold (Max Coverage)
    if not np.any(feasible):
        # Even the single safest point is too risky (or set is empty)
        return -np.inf 
    
    last_feasible_idx = np.where(feasible)[0][-1]
    return s_sorted[last_feasible_idx]

def eval_conformal(df, score_col, alpha, delta, use_ucb):
    cal = df[df["split"] == "cal"]
    test = df[df["split"] == "test"]
    
    t = select_threshold_1d(
        cal[score_col].values, cal["y_hall"].values, 
        alpha, delta, use_ucb
    )
    
    accept = test[score_col].values <= t
    cov = float(accept.mean())
    risk = float(test.loc[accept, "y_hall"].mean()) if accept.sum() > 0 else 0.0
    return risk, cov

def get_risk_coverage_curve(df_test, score_col):
    scores = df_test[score_col].values
    y = df_test["y_hall"].values
    order = np.argsort(scores) # Low (Safe) -> High
    y_sorted = y[order]
    
    n = len(y)
    accepted_counts = np.arange(1, n + 1)
    cum_risk = np.cumsum(y_sorted) / accepted_counts
    covs = accepted_counts / n
    
    # Downsample for plotting file size
    if n > 300:
        idx = np.linspace(0, n-1, 300).astype(int)
        return list(covs[idx]), list(cum_risk[idx])
    return list(covs), list(cum_risk)

# -----------------------------
# Plotting
# -----------------------------

def plot_detection_bars(df_det, figures_dir):
    # Fix: Plot PER MODEL to avoid aggregation confusion
    for model in df_det["Model"].unique():
        for subset in ["Full", "Confident"]:
            data = df_det[(df_det["Model"] == model) & (df_det["Subset"] == subset)]
            if data.empty: continue
            
            plt.figure(figsize=(9, 6))
            sns.barplot(
                data=data, x="Dataset", y="AUROC", hue="Method",
                palette=COLORS, edgecolor="black", errorbar=None,
                hue_order=[m for m in METHOD_ORDER if m in data["Method"].unique()]
            )
            plt.title(f"{model} - {subset} Subset")
            plt.ylim(0.4, 1.0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            
            safe_model = model.replace("/", "_")
            plt.savefig(figures_dir / f"detection_{safe_model}_{subset}.png", dpi=300)
            plt.close()

def plot_risk_coverage(curve_data, figures_dir):
    for model, dsets in curve_data.items():
        for ds, methods in dsets.items():
            plt.figure(figsize=(8, 6))
            
            # Track max risk for dynamic ylim
            max_r = 0.0
            for m_name, (cov, risk) in methods.items():
                if "GBM" in m_name: continue # Optional: Hide GBM to de-clutter
                plt.plot(cov, risk, label=m_name, color=COLORS.get(m_name, "black"), linewidth=2.5)
                if risk:
                    max_r = max(max_r, max(risk))
            
            plt.xlabel("Coverage")
            plt.ylabel("Hallucination Rate (Risk)")
            plt.title(f"Risk-Coverage: {model} / {ds}")
            plt.legend()
            plt.xlim(0, 1)
            # Dynamic Y-Limit (cap at 1.0)
            top_lim = min(1.0, max_r * 1.15) if max_r > 0 else 1.0
            plt.ylim(0, top_lim)
            plt.grid(True, alpha=0.3)
            
            safe_model = model.replace("/", "_")
            plt.savefig(figures_dir / f"rc_{safe_model}_{ds}.png", dpi=300)
            plt.close()

def plot_decision_boundary(df, model, ds, alpha, delta, use_ucb, out_path, combiner_type="MLP"):
    cal = df[df["split"] == "cal"]
    trainval = df[df["split"].isin(["train", "val"])]
    
    if combiner_type == "MLP":
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42))
    elif combiner_type == "GBM":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
        
    X_train = trainval[["p_correct", "p_high_entropy"]]
    y_train = trainval["y_hall"]
    
    if len(np.unique(y_train)) < 2: return
    clf.fit(X_train, y_train)
    
    score_col = f"p_halluc_{combiner_type}"
    scores_cal = df.loc[cal.index, score_col].values
    t_star = select_threshold_1d(scores_cal, cal["y_hall"].values, alpha, delta, use_ucb)
    
    if np.isinf(t_star): return

    plt.figure(figsize=(7, 6))
    mask_c = cal["y_hall"] == 0
    mask_h = cal["y_hall"] == 1
    
    plt.scatter(cal.loc[mask_c, "p_correct"], cal.loc[mask_c, "p_high_entropy"], 
                c=COLORS["Accuracy Probe"], alpha=0.3, s=20, label="Correct")
    plt.scatter(cal.loc[mask_h, "p_correct"], cal.loc[mask_h, "p_high_entropy"], 
                c=COLORS["Combined (GBM)"], alpha=0.3, s=20, label="Hallucination")
    
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(pd.DataFrame(grid, columns=["p_correct", "p_high_entropy"]))[:, 1]
    probs = probs.reshape(xx.shape)
    
    plt.contour(xx, yy, probs, levels=[t_star], colors='k', linewidths=2.5, linestyles='--')
    plt.contourf(xx, yy, probs, levels=[0, t_star], colors=[COLORS["Combined (LR)"]], alpha=0.15)
    
    plt.xlabel("Accuracy Probe ($P_{correct}$)")
    plt.ylabel("SE Probe ($P_{high\_entropy}$)")
    plt.title(f"Decision Boundary ({combiner_type}) @ $\\alpha={alpha}$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_layer_sensitivity(layer_results, figures_dir):
    for item in layer_results:
        acc_aucs = item["acc_aucs"]
        se_aucs = item["se_aucs"]
        L = len(acc_aucs)
        x = np.linspace(0, 1, L)

        plt.figure(figsize=(8, 5))
        plt.plot(x, acc_aucs, label="Accuracy Probe", color=COLORS["Accuracy Probe"], linewidth=2.5)
        plt.plot(x, se_aucs, label="SE Probe", color=COLORS["SE Probe"], linestyle="--", linewidth=2.5)
        plt.xlabel("Layer Depth (normalized)")
        plt.ylabel("Validation AUROC")
        plt.title(f"Layer Sensitivity: {item['Model']} / {item['Dataset']}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_model = item['Model'].replace("/", "_")
        plt.savefig(figures_dir / f"layers_{safe_model}_{item['Dataset']}.png", dpi=300)
        plt.close()

def plot_calibration_curves(df_cal, figures_dir):
    """
    Plots Target Risk (x) vs Realized Risk (y) for each model.
    Layout: 1 row x N datasets subplots per model.
    """
    unique_models = df_cal["Model"].unique()
    unique_datasets = sorted(df_cal["Dataset"].unique())
    n_ds = len(unique_datasets)

    for model in unique_models:
        model_data = df_cal[df_cal["Model"] == model]
        if model_data.empty: continue

        # Setup subplots (1 row, N columns)
        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5.5), sharey=True)
        if n_ds == 1: axes = [axes]

        fig.suptitle(f"Conformal Calibration: {model}", fontsize=15, y=0.98)

        for i, (ax, ds) in enumerate(zip(axes, unique_datasets)):
            ds_data = model_data[model_data["Dataset"] == ds]
            
            # Identity line (Ideal)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5)

            if not ds_data.empty:
                sns.lineplot(
                    data=ds_data, x="Target", y="Realized", hue="Method",
                    palette=COLORS, linewidth=2.5, ax=ax, legend=False
                )

            ax.set_title(ds)
            ax.set_xlabel("Target Risk ($\\alpha$)")
            ax.set_xlim(0, 0.4)
            ax.set_ylim(0, 0.4)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_ylabel("Realized Hallucination Rate")
            else:
                ax.set_ylabel("")

        # Create a single unified legend outside the subplots
        # We grab handles from the last subplot that had data
        handles, labels = [], []
        for ax in axes:
            if ax.lines: # If data was plotted (beyond the identity line)
                # Create dummy handles for the legend based on color map
                # (Seaborn makes this tricky to extract perfectly from ax without legend=True)
                # Simpler: create a dummy plot to steal handles
                dummy_fig = plt.figure()
                dummy_ax = dummy_fig.add_subplot(111)
                sns.lineplot(data=model_data, x="Target", y="Realized", hue="Method", palette=COLORS, ax=dummy_ax)
                handles, labels = dummy_ax.get_legend_handles_labels()
                plt.close(dummy_fig)
                break
        
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), frameon=False)

        plt.tight_layout()
        # Adjust layout to make room for the legend at the bottom
        plt.subplots_adjust(bottom=0.15)
        
        safe_model = model.replace("/", "_")
        plt.savefig(figures_dir / f"calibration_{safe_model}.png", dpi=300, bbox_inches='tight')
        plt.close()

# -----------------------------
# LaTeX Generation
# -----------------------------

def latex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")

def write_conformal_tables(df_cal, figures_dir, table_alphas):
    df = df_cal.copy()
    df["Target"] = df["Target"].round(3)
    keep = [round(a, 3) for a in table_alphas]
    df = df[df["Target"].isin(keep)]
    df = df[df["Method"].isin(METHOD_ORDER)]
    
    if df.empty: return

    for dataset in sorted(df["Dataset"].unique()):
        dfd = df[df["Dataset"] == dataset]
        alphas = sorted(keep)
        
        lines = []
        col_spec = "l" + "cc" * len(alphas)
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        
        h1 = [r"\textbf{Method}"] + [rf"\multicolumn{{2}}{{c}}{{\textbf{{$\alpha={a}$}}}}" for a in alphas]
        lines.append(" & ".join(h1) + r" \\")
        h2 = [r"\textbf{Method}"] + [r"\textbf{Cov}", r"\textbf{Risk}"] * len(alphas)
        lines.append(" & ".join(h2) + r" \\")
        lines.append(r"\midrule")

        for model in sorted(dfd["Model"].unique()):
            dfm = dfd[dfd["Model"] == model]
            lines.append(rf"\multicolumn{{{1 + 2*len(alphas)}}}{{l}}{{\textbf{{{latex_escape(model)}}}}} \\")
            
            winners = {}
            for a in alphas:
                blk = dfm[dfm["Target"] == a]
                # Filter valid results (Risk <= Target + epsilon)
                valid = blk[blk["Realized"] <= a + 0.005]
                
                best_cov_m = None
                if not valid.empty:
                    best_cov_m = valid.sort_values("Coverage", ascending=False).iloc[0]["Method"]
                
                best_risk_m = None
                if not valid.empty:
                    best_risk_m = valid.sort_values("Realized", ascending=False).iloc[0]["Method"]
                
                winners[a] = (best_cov_m, best_risk_m)

            for method in METHOD_ORDER:
                row = [latex_escape(method)]
                row_dat = dfm[dfm["Method"] == method]
                
                for a in alphas:
                    cell = row_dat[row_dat["Target"] == a]
                    if cell.empty:
                        row.extend(["-", "-"])
                    else:
                        c = cell.iloc[0]["Coverage"]
                        r = cell.iloc[0]["Realized"]
                        c_str = f"{c:.3f}"
                        r_str = f"{r:.3f}"
                        
                        w_cov, w_risk = winners[a]
                        # Bold Highest Coverage
                        if method == w_cov: c_str = rf"\textbf{{{c_str}}}"
                        # Bold Risk closest to target (valid)
                        if method == w_risk: r_str = rf"\textbf{{{r_str}}}"
                        
                        row.extend([c_str, r_str])
                
                lines.append(" & ".join(row) + r" \\")
            lines.append(r"\addlinespace")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        
        with open(figures_dir / f"conformal_table_{dataset}.tex", "w") as f:
            f.write("\n".join(lines))

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, required=True)
    parser.add_argument("--figures-dirname", type=str, default="analysis_output_v3")
    parser.add_argument("--use-ucb", action="store_true")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--decision-plot-method", type=str, default="MLP")
    args = parser.parse_args()

    repo_root, runs_root = resolve_paths(args.runs_root)
    figures_dir = repo_root / args.figures_dirname
    figures_dir.mkdir(exist_ok=True, parents=True)

    det_results = []
    cal_results = []
    layer_results_list = []
    curve_data = {} 

    print(f"Scanning {runs_root}...")

    for model_name in DEFAULT_TARGET_MODELS:
        safe_model = Path(model_name).name
        curve_data[safe_model] = {}
        
        for dataset_name in DEFAULT_TARGET_DATASETS:
            run_dir = find_run_directory(runs_root, model_name, dataset_name)
            if not run_dir: continue
            
            print(f"Processing {safe_model} / {dataset_name}...")
            
            try:
                pickles = load_pickles(run_dir)
                if not pickles: continue
                X, y, se = extract_features_aligned(*pickles, embedding_key="emb_last_tok_before_gen")
                df, l_stats, splits = process_model_dataset(X, y, se, seed=42)
                
                # --- DEBUG CHECK ---
                cal_df = df[df["split"] == "cal"]
                cal_risk = cal_df["y_hall"].mean()
                print(f"  [Info] Calibration set size: {len(cal_df)}, Hallucination Rate: {cal_risk:.3f}")
                if cal_risk == 0.0 or cal_risk == 1.0:
                    print("  [Warning] Calibration risk is 0 or 1. Conformal prediction may fail to find thresholds.")
                # -------------------
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue

            layer_results_list.append({
                "Model": safe_model, "Dataset": dataset_name,
                "acc_aucs": l_stats["acc_aucs"], "se_aucs": l_stats["se_aucs"]
            })

            # Scores -> Risk (High = Bad)
            df["score_SE"] = df["se_raw"]
            df["score_Acc"] = 1.0 - df["p_correct"]
            df["score_SE_Probe"] = df["p_high_entropy"]
            df["score_Comb_LR"] = df["p_halluc_LR"]
            df["score_Comb_MLP"] = df["p_halluc_MLP"]
            # df["score_Comb_GBM"] = df["p_halluc_GBM"]

            method_map = {
                "Semantic Entropy": "score_SE",
                "Accuracy Probe": "score_Acc",
                "SE Probe": "score_SE_Probe",
                "Combined (LR)": "score_Comb_LR",
                "Combined (MLP)": "score_Comb_MLP",
                # "Combined (GBM)": "score_Comb_GBM"
            }

            curve_data[safe_model][dataset_name] = {}
            df_test = df[df["split"] == "test"].copy()
            df_conf = df_test[df_test["se_raw"] <= df_test["se_raw"].quantile(0.3)].copy()

            for m_name, col in method_map.items():
                # AUROC
                try:
                    auc_full = roc_auc_score(df_test["y_hall"], df_test[col])
                except: auc_full = 0.5
                
                try:
                    if len(df_conf) > 0 and len(np.unique(df_conf["y_hall"])) > 1:
                        auc_conf = roc_auc_score(df_conf["y_hall"], df_conf[col])
                    else: auc_conf = 0.5
                except: auc_conf = 0.5
                
                det_results.append({"Model": safe_model, "Dataset": dataset_name, "Method": m_name, "Subset": "Full", "AUROC": auc_full})
                det_results.append({"Model": safe_model, "Dataset": dataset_name, "Method": m_name, "Subset": "Confident", "AUROC": auc_conf})

                # Curves
                covs, risks = get_risk_coverage_curve(df_test, col)
                curve_data[safe_model][dataset_name][m_name] = (covs, risks)

                # Conformal Calibration
                # Use a dense grid for smooth plots (0.5% to 40% risk)
                dense_alphas = np.linspace(0.005, 0.4, 40)
                
                # Ensure we include the specific table alphas so they aren't interpolated
                table_alphas = np.array([0.01, 0.05, 0.1, 0.2])
                all_alphas = np.unique(np.sort(np.concatenate([dense_alphas, table_alphas])))

                for alpha in all_alphas:
                    r, c = eval_conformal(df, col, float(alpha), args.delta, args.use_ucb)
                    cal_results.append({
                        "Model": safe_model, "Dataset": dataset_name, "Method": m_name,
                        "Target": float(alpha), "Realized": r, "Coverage": c
                    })

            plot_decision_boundary(
                df, safe_model, dataset_name, alpha=0.1, delta=args.delta, 
                use_ucb=args.use_ucb, 
                out_path=figures_dir / f"boundary_{safe_model}_{dataset_name}.png",
                combiner_type=args.decision_plot_method
            )

    df_det = pd.DataFrame(det_results)
    df_cal = pd.DataFrame(cal_results)
    df_det.to_csv(figures_dir / "auroc_summary.csv", index=False)
    df_cal.to_csv(figures_dir / "calibration_summary.csv", index=False)

    print("Generating Plots...")
    plot_detection_bars(df_det, figures_dir)
    plot_layer_sensitivity(layer_results_list, figures_dir)
    plot_risk_coverage(curve_data, figures_dir)
    plot_calibration_curves(df_cal, figures_dir)
    
    print("Generating Tables...")
    write_conformal_tables(df_cal, figures_dir, [0.01, 0.05, 0.1])
    
    print(f"Done! Artifacts saved to: {figures_dir}")

if __name__ == "__main__":
    main()