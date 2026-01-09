#!/usr/bin/env python3
"""
analyze_results.py (3-way + CV aligned)

Reads per-run trained probes + splits from probes.pkl (produced by the 3-way + CV train_probes.py),
then produces paper figures/tables:

- Hallucination Detection AUROC (Full + Confident subset) bar charts
- Layer sensitivity plots (from CV AUCs saved in probes.pkl)
- Risk–Coverage curves (+ Ideal oracle line)
- AURC (area under risk–coverage) CSV + LaTeX
- 1D "calibration" curves (target alpha vs realized risk on test) using the calibration split
- Decision boundary visualization in (p_correct, p_high_entropy) space using the SAME fitted combiner model

Conventions:
- y_hall = 1 means hallucination/incorrect, y_hall = 0 correct.
- Risk scores: higher = more risky, lower = safer.
- Thresholding accepts if score <= t.

Usage:
  python analyze_results.py --runs-root /path/to/runs --figures-dirname analysis_output_v4
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import subprocess

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


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

# Plotting aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams["font.family"] = "serif"

COLORS = {
    "Semantic Entropy": "#7f8c8d",
    "Accuracy Probe": "#3498db",
    "SE Probe": "#e67e22",
    "Combined (LR)": "#2ecc71",
    "Combined (MLP)": "#9b59b6",
    "Combined (GBM)": "#e74c3c",
}

METHOD_ORDER = [
    "Semantic Entropy",
    "Accuracy Probe",
    "SE Probe",
    "Combined (LR)",
    "Combined (MLP)",
    # "Combined (GBM)",
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


def find_wandb_files_dir(run_dir: Path) -> Path:
    """
    Mirrors your other scripts: prefer run_dir/files, else search for newest .../files.
    """
    files_dir = run_dir / "files"
    if files_dir.exists():
        return files_dir
    candidates = list(run_dir.rglob("files"))
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    return run_dir


def load_run_artifacts(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Loads:
      - validation_generations.pkl
      - uncertainty_measures.pkl
      - probes.pkl (required for 3-way+CV alignment)
    """
    files_dir = find_wandb_files_dir(run_dir)

    p_gens = files_dir / "validation_generations.pkl"
    p_unc = files_dir / "uncertainty_measures.pkl"
    p_probes = files_dir / "probes.pkl"

    if not (p_gens.exists() and p_unc.exists() and p_probes.exists()):
        return None

    with p_gens.open("rb") as f:
        gens = pickle.load(f)
    with p_unc.open("rb") as f:
        unc = pickle.load(f)
    with p_probes.open("rb") as f:
        probes = pickle.load(f)

    return {
        "files_dir": files_dir,
        "gens": gens,
        "unc": unc,
        "probes": probes,
    }


# -----------------------------
# Data Extraction
# -----------------------------

def _get_entropy_array(unc: dict) -> Optional[np.ndarray]:
    ent_keys = [
        "cluster_assignment_entropy",
        "semantic_entropy_sum_normalized",
        "semantic_entropy_sum-normalized",
        "semantic_entropy_sum-normalized-rao",
        "semantic_entropy_sum",
    ]
    if "uncertainty_measures" not in unc:
        return None
    for k in ent_keys:
        if k in unc["uncertainty_measures"]:
            return np.array(unc["uncertainty_measures"][k], dtype=np.float32)
    return None


def _get_embedding_stack_sorted(gens: dict, embedding_key: str) -> torch.Tensor:
    """
    Ensures deterministic order: sort by example id key.
    Returns tensor [n_layers, n_examples, d]
    """
    sorted_ids = sorted(gens.keys())
    gen_values = [gens[k] for k in sorted_ids]

    embs = []
    for g in gen_values:
        if "most_likely_answer" not in g or embedding_key not in g["most_likely_answer"]:
            raise KeyError(f"Missing embedding key {embedding_key} under most_likely_answer.")
        embs.append(g["most_likely_answer"][embedding_key])

    tlist = [e if isinstance(e, torch.Tensor) else torch.tensor(e) for e in embs]
    stacked = torch.stack(tlist)  # [n, ...]
    while stacked.ndim > 3:
        stacked = stacked.squeeze(-2)

    if stacked.ndim != 3:
        raise ValueError(f"Unexpected embedding tensor shape after squeeze: {tuple(stacked.shape)}")

    # [n_examples, n_layers, d] -> [n_layers, n_examples, d]
    return stacked.transpose(0, 1).contiguous()


def extract_features_aligned(
    gens: dict,
    unc: dict,
    embedding_key: str,
    n_cap: int = 2000,
    hall_acc_threshold: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X: [n_layers, n, d] float32
      y_hall: [n] int (1=hallucination)
      se_raw: [n] float32
      ids: sorted ids (length n)
    """
    sorted_ids = sorted(gens.keys())
    gen_values = [gens[k] for k in sorted_ids]
    n_total = len(gen_values)

    accuracies = np.array([float(g["most_likely_answer"]["accuracy"]) for g in gen_values], dtype=np.float32)
    y_hall = (accuracies < hall_acc_threshold).astype(np.int64)

    se_raw = _get_entropy_array(unc)
    if se_raw is None:
        se_raw = np.zeros_like(accuracies, dtype=np.float32)
    else:
        # IMPORTANT: semantic entropy script also used sorted ids; assume alignment is by sorted id order
        se_raw = se_raw.astype(np.float32)

    X_t = _get_embedding_stack_sorted(gens, embedding_key=embedding_key)  # [L, n, d]
    X = X_t.cpu().numpy().astype(np.float32)

    n = min(n_total, X.shape[1], len(y_hall), len(se_raw), n_cap)
    return X[:, :n, :], y_hall[:n], se_raw[:n], sorted_ids[:n]


# -----------------------------
# Probe loading helpers (robust)
# -----------------------------

def _dig(obj: Any, keys: List[str]) -> Optional[Any]:
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def load_splits_from_probes(probes_obj: dict) -> Dict[str, np.ndarray]:
    if "splits" not in probes_obj:
        raise KeyError("probes.pkl missing top-level 'splits' key.")

    splits = probes_obj["splits"]
    required = ["train", "calibration", "test"]

    for k in required:
        if k not in splits:
            raise KeyError(f"probes.pkl missing split '{k}'")

    return {
        "train": np.asarray(splits["train"], dtype=np.int64),
        "cal": np.asarray(splits["calibration"], dtype=np.int64),
        "test": np.asarray(splits["test"], dtype=np.int64),
    }



def load_probe_bundle(probes_obj: dict, position: str = "tbg") -> Dict[str, Any]:
    if "probes" not in probes_obj:
        raise KeyError("probes.pkl missing 'probes'")

    if position not in probes_obj["probes"]:
        raise KeyError(f"probes.pkl missing position '{position}'")

    p = probes_obj["probes"][position]

    return {
        "acc_model": p["acc"]["model"],
        "acc_best_layer": int(p["acc"]["best_layer"]),
        "acc_cv_aucs": list(p["acc"]["cv_auc_per_layer"]),
        "se_model": p["se"]["model"],
        "se_best_layer": int(p["se"]["best_layer"]),
        "se_cv_aucs": list(p["se"]["cv_auc_per_layer"]),
    }



def maybe_train_all_probes(
    runs_root: Path,
    cfg_yaml: Path,
    train_probes_script: Path,
    retrain: bool,
) -> None:
    """
    Optionally retrain probes for ALL runs found under runs_root.
    Called ONCE at analysis start.
    """
    if not retrain:
        return

    print("[analysis] Retraining probes for all available runs...")

    cmd = [
        "python",
        str(train_probes_script),
        "--config", str(cfg_yaml),
        "--runs-root", str(runs_root),
    ]

    subprocess.run(cmd, check=True)


# -----------------------------
# "Calibration" / thresholding helpers (1D)
# -----------------------------

def try_clopper_pearson_ucb(k: int, n: int, delta: float) -> float:
    try:
        from scipy.stats import beta
        if n <= 0:
            return 1.0
        if k >= n:
            return 1.0
        return float(beta.ppf(1 - delta, k + 1, n - k))
    except Exception:
        return (k / n) if n > 0 else 1.0


def select_threshold_1d(scores: np.ndarray, labels: np.ndarray, alpha: float, delta: float, use_ucb: bool) -> float:
    """
    Scores are risk scores (lower = safer).
    Accept if score <= t.
    We choose the largest t (max coverage) such that risk (or UCB) <= alpha on calibration.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    order = np.argsort(scores)  # safest -> riskiest
    y_sorted = labels[order]
    s_sorted = scores[order]

    cum_fail = np.cumsum(y_sorted).astype(float)
    counts = np.arange(1, len(y_sorted) + 1).astype(float)

    if use_ucb:
        risks = np.array([try_clopper_pearson_ucb(int(k), int(n), delta) for k, n in zip(cum_fail, counts)])
    else:
        risks = cum_fail / counts

    feasible = risks <= alpha
    if not np.any(feasible):
        return -np.inf  # accept nothing

    last_idx = np.where(feasible)[0][-1]
    return float(s_sorted[last_idx])


def eval_calibration(df: pd.DataFrame, score_col: str, alpha: float, delta: float, use_ucb: bool) -> Tuple[float, float]:
    """
    Threshold picked on cal split, evaluated on test.
    Returns (realized_risk, coverage).
    """
    cal = df[df["split"] == "cal"]
    test = df[df["split"] == "test"]

    t = select_threshold_1d(cal[score_col].values, cal["y_hall"].values, alpha, delta, use_ucb)
    if np.isinf(t) and t < 0:
        return 0.0, 0.0

    accept = test[score_col].values <= t
    cov = float(accept.mean())
    risk = float(test.loc[accept, "y_hall"].mean()) if accept.sum() > 0 else 0.0
    return risk, cov


def get_risk_coverage_curve(df_test: pd.DataFrame, score_col: str) -> Tuple[List[float], List[float]]:
    """
    Sweep thresholds by sorting risk scores ascending.
    Returns coverage and risk arrays (monotone).
    """
    scores = df_test[score_col].values
    y = df_test["y_hall"].values

    order = np.argsort(scores)  # safest -> riskiest
    y_sorted = y[order]

    n = len(y_sorted)
    accepted = np.arange(1, n + 1)
    risk = np.cumsum(y_sorted) / accepted
    cov = accepted / n

    # downsample for file size
    if n > 300:
        idx = np.linspace(0, n - 1, 300).astype(int)
        return list(cov[idx]), list(risk[idx])

    return list(cov), list(risk)

def compute_normalized_aurcc(cov: List[float], risk: List[float], base_risk: float) -> float:
    """
    Computes Normalized AURCC (nAURCC).
    nAURCC = (AURCC - AURCC_oracle) / (AURCC_worst - AURCC_oracle)
    Lower is better. 0 = Oracle, 1 = Random/Worst.
    """
    if len(cov) < 2:
        return float("nan")

    aurcc_actual = np.trapezoid(risk, cov)

    # Oracle: Risk is 0 until coverage > (1-base_risk), then rises linearly
    # Points: (0,0) -> (1-base_risk, 0) -> (1, base_risk)
    cov_oracle = [0.0, max(0.0, 1.0 - base_risk), 1.0]
    risk_oracle = [0.0, 0.0, base_risk]
    aurcc_oracle = np.trapezoid(risk_oracle, cov_oracle)

    # Worst (Inverse Oracle): Risk starts at 1.0 until hallucinations exhausted
    # Points: (0,1) -> (base_risk, 1) -> (1, base_risk)
    cov_worst = [0.0, base_risk, 1.0]
    risk_worst = [1.0, 1.0, base_risk]
    aurcc_worst = np.trapezoid(risk_worst, cov_worst)

    denom = aurcc_worst - aurcc_oracle
    if denom < 1e-9:
        return 0.0
    
    return (aurcc_actual - aurcc_oracle) / denom


# -----------------------------
# Plotting
# -----------------------------

def plot_detection_bars(df_det: pd.DataFrame, figures_dir: Path) -> None:
    for model in df_det["Model"].unique():
        for subset in ["Full", "Confident"]:
            data = df_det[(df_det["Model"] == model) & (df_det["Subset"] == subset)]
            if data.empty:
                continue

            plt.figure(figsize=(9, 6))
            sns.barplot(
                data=data,
                x="Dataset",
                y="AUROC",
                hue="Method",
                palette=COLORS,
                edgecolor="black",
                errorbar=None,
                hue_order=[m for m in METHOD_ORDER if m in data["Method"].unique()],
            )
            plt.title(f"{model} - {subset} Subset")
            plt.ylim(0.4, 1.0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            safe_model = model.replace("/", "_")
            plt.savefig(figures_dir / f"detection_{safe_model}_{subset}.png", dpi=300)
            plt.close()


def plot_layer_sensitivity(layer_results: List[dict], figures_dir: Path) -> None:
    for item in layer_results:
        acc_aucs = item.get("acc_cv_aucs", [])
        se_aucs = item.get("se_cv_aucs", [])
        L = max(len(acc_aucs), len(se_aucs))
        if L <= 1:
            continue

        x = np.linspace(0, 1, L)

        plt.figure(figsize=(8, 5))
        if acc_aucs:
            plt.plot(x[:len(acc_aucs)], acc_aucs, label="Accuracy Probe (CV AUROC)", color=COLORS["Accuracy Probe"], linewidth=2.5)
        if se_aucs:
            plt.plot(x[:len(se_aucs)], se_aucs, label="SE Probe (CV AUROC)", color=COLORS["SE Probe"], linestyle="--", linewidth=2.5)

        plt.xlabel("Layer Depth (normalized)")
        plt.ylabel("CV AUROC")
        plt.title(f"Layer Sensitivity: {item['Model']} / {item['Dataset']}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_model = item["Model"].replace("/", "_")
        plt.savefig(figures_dir / f"layers_{safe_model}_{item['Dataset']}.png", dpi=300)
        plt.close()


def plot_risk_coverage(curve_data: dict, figures_dir: Path, base_risks: dict) -> None:
    """
    Includes an "Ideal (Oracle)" line based on base risk on the test set.
    """
    for model, dsets in curve_data.items():
        for ds, methods in dsets.items():
            plt.figure(figsize=(8, 6))

            base_risk = float(base_risks.get((model, ds), 0.5))
            max_safe_cov = max(0.0, 1.0 - base_risk)

            ideal_x = [0.0, max_safe_cov, 1.0]
            ideal_y = [0.0, 0.0, base_risk]
            plt.plot(ideal_x, ideal_y, linestyle="--", color="black", label="Ideal (Oracle)", alpha=0.6, linewidth=1.5)

            max_r = base_risk
            for m_name, (cov, risk) in methods.items():
                plt.plot(cov, risk, label=m_name, color=COLORS.get(m_name, "black"), linewidth=2.5)
                if len(risk) > 0:
                    max_r = max(max_r, max(risk))

            plt.xlabel("Coverage")
            plt.ylabel("Hallucination Rate (Risk)")
            plt.title(f"Risk-Coverage: {model} / {ds}")
            plt.legend()
            plt.xlim(0, 1)

            top_lim = min(1.0, max_r * 1.15) if max_r > 0 else 1.0
            plt.ylim(0, top_lim)
            plt.grid(True, alpha=0.3)

            safe_model = model.replace("/", "_")
            plt.savefig(figures_dir / f"rc_{safe_model}_{ds}.png", dpi=300)
            plt.close()


def plot_calibration_curves(df_cal: pd.DataFrame, figures_dir: Path) -> None:
    """
    Plots Target risk (x) vs Realized risk (y), per model, with subplots per dataset.
    """
    unique_models = df_cal["Model"].unique()
    unique_datasets = sorted(df_cal["Dataset"].unique())
    n_ds = len(unique_datasets)

    for model in unique_models:
        model_data = df_cal[df_cal["Model"] == model]
        if model_data.empty:
            continue

        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5.5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        fig.suptitle(f"Calibration (threshold on cal, eval on test): {model}", fontsize=15, y=0.98)

        for i, (ax, ds) in enumerate(zip(axes, unique_datasets)):
            ds_data = model_data[model_data["Dataset"] == ds]

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5)

            if not ds_data.empty:
                sns.lineplot(
                    data=ds_data,
                    x="Target",
                    y="Realized",
                    hue="Method",
                    palette=COLORS,
                    linewidth=2.5,
                    ax=ax,
                    legend=False,
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

        # unified legend
        dummy_fig = plt.figure()
        dummy_ax = dummy_fig.add_subplot(111)
        sns.lineplot(data=model_data, x="Target", y="Realized", hue="Method", palette=COLORS, ax=dummy_ax)
        handles, labels = dummy_ax.get_legend_handles_labels()
        plt.close(dummy_fig)
        if handles:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels), frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        safe_model = model.replace("/", "_")
        plt.savefig(figures_dir / f"calibration_{safe_model}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_decision_boundary(
    df: pd.DataFrame,
    combiner_model,
    score_col: str,
    alpha: float,
    delta: float,
    use_ucb: bool,
    out_path: Path,
    title: str,
) -> None:
    """
    Uses the SAME fitted combiner_model that produced score_col.
    Draws contour at calibrated threshold t* found on cal.
    """
    cal = df[df["split"] == "cal"]
    if cal.empty:
        return

    # threshold on cal
    t_star = select_threshold_1d(cal[score_col].values, cal["y_hall"].values, alpha, delta, use_ucb)
    if np.isinf(t_star) and t_star < 0:
        return

    # scatter
    plt.figure(figsize=(7, 6))
    mask_c = cal["y_hall"] == 0
    mask_h = cal["y_hall"] == 1

    plt.scatter(cal.loc[mask_c, "p_correct"], cal.loc[mask_c, "p_high_entropy"],
                c=COLORS["Accuracy Probe"], alpha=0.3, s=20, label="Correct")
    plt.scatter(cal.loc[mask_h, "p_correct"], cal.loc[mask_h, "p_high_entropy"],
                c=COLORS["Combined (MLP)"], alpha=0.3, s=20, label="Hallucination")

    # grid
    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid, columns=["p_correct", "p_high_entropy"])

    probs = combiner_model.predict_proba(grid_df)[:, 1]
    probs = probs.reshape(xx.shape)

    plt.contour(xx, yy, probs, levels=[t_star], colors="k", linewidths=2.5, linestyles="--")
    plt.contourf(xx, yy, probs, levels=[0, t_star], colors=[COLORS["Combined (LR)"]], alpha=0.12)

    plt.xlabel("Accuracy Probe ($P_{correct}$)")
    plt.ylabel("SE Probe ($P_{high\\_entropy}$)")
    plt.title(f"{title} @ $\\alpha={alpha}$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_correlation_scatter(df: pd.DataFrame, model_name: str, ds_name: str, out_path: Path):
    """
    Plots Semantic Entropy vs Accuracy Probe Score.
    Highlights the 'Confidently Wrong' quadrant.
    """
    plt.figure(figsize=(7, 6))
    
    # Add jitter to SE because it often clamps to 0
    se_jitter = df["se_raw"] + np.random.normal(0, 0.005, size=len(df))
    
    # Scatter points
    sns.scatterplot(
        x=se_jitter, 
        y=df["p_correct"], 
        hue=df["y_hall"],
        palette={0: "#2ecc71", 1: "#e74c3c"}, # Green/Red
        style=df["y_hall"],
        markers={0: "o", 1: "X"},
        alpha=0.6,
        s=40
    )
    
    plt.axvline(x=0.05, color='gray', linestyle='--', alpha=0.5, label="Low Entropy")
    plt.xlabel("Semantic Entropy (Jittered)")
    plt.ylabel("Accuracy Probe ($P_{correct}$)")
    plt.title(f"{model_name}\n{ds_name}")
    
    # Highlight Confidently Wrong Region (Low Entropy, Low P_correct, Hallucination)
    # Just an annotation
    plt.text(0.05, 0.1, "Confidently\nWrong", color='red', fontsize=12, fontweight='bold')
    
    plt.legend(title="Hallucination")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -----------------------------
# LaTeX Tables
# -----------------------------

def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write_detection_table(df_det: pd.DataFrame, figures_dir: Path) -> None:
    """
    Writes a table comparing AUROC and AUPRC.
    Columns: Dataset -> [AUROC, AUPRC]
    """
    df = df_det[df_det["Method"].isin(METHOD_ORDER)].copy()
    if df.empty: return

    # We only care about the "Full" subset for this table now
    df = df[df["Subset"] == "Full"]

    datasets = sorted(df["Dataset"].unique())
    models = sorted(df["Model"].unique())

    # identify winners (max) for bolding
    winners = {}
    for model in models:
        for ds in datasets:
            sd = df[(df["Model"] == model) & (df["Dataset"] == ds)]
            if not sd.empty:
                winners[(model, ds, "AUROC")] = sd["AUROC"].max()
                winners[(model, ds, "AUPRC")] = sd["AUPRC"].max()

    lines = []
    col_spec = "l" + "cc" * len(datasets)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header 1: Datasets
    header_1 = [r"\textbf{Method}"]
    for ds in datasets:
        header_1.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{latex_escape(ds)}}}}}")
    lines.append(" & ".join(header_1) + r" \\")

    # Header 2: Metrics
    header_2 = [r""]
    for _ in datasets:
        header_2.extend([r"\scriptsize{AUROC}", r"\scriptsize{AUPRC}"])
    lines.append(" & ".join(header_2) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        lines.append(rf"\multicolumn{{{1 + 2*len(datasets)}}}{{l}}{{\textbf{{{latex_escape(model)}}}}} \\")
        for method in METHOD_ORDER:
            row = [latex_escape(method)]
            for ds in datasets:
                # Get Full subset rows
                row_dat = df[(df["Model"] == model) & (df["Dataset"] == ds) & (df["Method"] == method)]
                
                if row_dat.empty:
                    row.extend(["-", "-"])
                else:
                    val_auc = float(row_dat.iloc[0]["AUROC"])
                    val_prc = float(row_dat.iloc[0]["AUPRC"])
                    
                    s_auc = f"{val_auc:.3f}"
                    s_prc = f"{val_prc:.3f}"

                    # Bold logic
                    if val_auc >= winners.get((model, ds, "AUROC"), -1) - 1e-6:
                        s_auc = rf"\textbf{{{s_auc}}}"
                    if val_prc >= winners.get((model, ds, "AUPRC"), -1) - 1e-6:
                        s_prc = rf"\textbf{{{s_prc}}}"
                    
                    row.extend([s_auc, s_prc])
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "detection_metrics_table.tex", "w") as f:
        f.write("\n".join(lines))

def write_aurc_table(df_aurc: pd.DataFrame, figures_dir: Path) -> None:
    df = df_aurc[df_aurc["Method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        return

    datasets = sorted(df["Dataset"].unique())
    models = sorted(df["Model"].unique())

    winners = {}
    for model in models:
        for ds in datasets:
            sd = df[(df["Model"] == model) & (df["Dataset"] == ds)]
            winners[(model, ds)] = sd["AURC"].min() if not sd.empty else 1e9

    lines = []
    col_spec = "l" + "c" * len(datasets)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = [r"\textbf{Method}"] + [rf"\textbf{{{latex_escape(ds)}}}" for ds in datasets]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        lines.append(rf"\multicolumn{{{1 + len(datasets)}}}{{l}}{{\textbf{{{latex_escape(model)}}}}} \\")
        for method in METHOD_ORDER:
            row = [latex_escape(method)]
            for ds in datasets:
                v = df[(df["Model"] == model) & (df["Dataset"] == ds) & (df["Method"] == method)]["AURC"]
                if v.empty:
                    row.append("-")
                else:
                    val = float(v.iloc[0])
                    s = f"{val:.3f}"
                    if val <= winners.get((model, ds), 1e9) + 1e-6:
                        s = rf"\textbf{{{s}}}"
                    row.append(s)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "aurc_summary_table.tex", "w") as f:
        f.write("\n".join(lines))


def write_calibration_tables(df_cal: pd.DataFrame, figures_dir: Path, table_alphas: List[float]) -> None:
    df = df_cal.copy()
    df["Target"] = df["Target"].round(3)
    keep = [round(a, 3) for a in table_alphas]
    df = df[df["Target"].isin(keep)]
    df = df[df["Method"].isin(METHOD_ORDER)]
    if df.empty:
        return

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

            # winners per alpha (bold)
            winners = {}
            for a in alphas:
                blk = dfm[dfm["Target"] == a]
                valid = blk[blk["Realized"] <= a + 0.005]

                best_cov_m = None
                best_risk_m = None
                if not valid.empty:
                    best_cov_m = valid.sort_values("Coverage", ascending=False).iloc[0]["Method"]
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
                        c = float(cell.iloc[0]["Coverage"])
                        r = float(cell.iloc[0]["Realized"])
                        c_str = f"{c:.3f}"
                        r_str = f"{r:.3f}"
                        w_cov, w_risk = winners[a]
                        if method == w_cov:
                            c_str = rf"\textbf{{{c_str}}}"
                        if method == w_risk:
                            r_str = rf"\textbf{{{r_str}}}"
                        row.extend([c_str, r_str])
                lines.append(" & ".join(row) + r" \\")
            lines.append(r"\addlinespace")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        with open(figures_dir / f"calibration_table_{dataset}.tex", "w") as f:
            f.write("\n".join(lines))


def write_naurcc_table(df_aurc: pd.DataFrame, figures_dir: Path) -> None:
    """
    Writes nAURCC table with:
    1. Values per (Model, Dataset)
    2. Row Average (Per Model, across Datasets)
    3. Bottom Block: Column Averages (Across Models) & Grand Mean
    
    * No percentage conversion.
    * 3 decimal places.
    """
    df = df_aurc[df_aurc["Method"].isin(METHOD_ORDER)].copy()
    if df.empty: return

    # Use raw nAURCC values (0 to 1 scale)
    value_col = "nAURCC"

    datasets = sorted(df["Dataset"].unique())
    models = sorted(df["Model"].unique())
    methods = [m for m in METHOD_ORDER if m in df["Method"].unique()]

    # --- Pre-calculate Statistics & Winners ---
    stats = {}
    
    # 1. Individual entries + Row Means
    for m in models:
        stats[m] = {}
        for meth in methods:
            stats[m][meth] = {}
            row_vals = []
            for d in datasets:
                val = df[(df["Model"] == m) & (df["Dataset"] == d) & (df["Method"] == meth)][value_col]
                if not val.empty:
                    v = float(val.iloc[0])
                    stats[m][meth][d] = v
                    row_vals.append(v)
                else:
                    stats[m][meth][d] = float("nan")
            
            # Row Mean
            stats[m][meth]["Avg"] = np.mean(row_vals) if row_vals else float("nan")

    # 2. Bottom Summary (Averages across models)
    stats["AVERAGE_ALL"] = {}
    for meth in methods:
        stats["AVERAGE_ALL"][meth] = {}
        
        all_models_vals = []
        for d in datasets:
            vals = df[(df["Dataset"] == d) & (df["Method"] == meth)][value_col]
            mean_val = vals.mean() if not vals.empty else float("nan")
            stats["AVERAGE_ALL"][meth][d] = mean_val
            if not vals.empty: all_models_vals.extend(vals.values)
        
        # Grand Mean (Average of all nAURCC values for this method)
        stats["AVERAGE_ALL"][meth]["Avg"] = np.mean(all_models_vals) if all_models_vals else float("nan")

    # 3. Identify Winners (Lowest nAURCC) for bolding
    winners = {}
    all_model_keys = models + ["AVERAGE_ALL"]
    all_ds_keys = datasets + ["Avg"]
    
    for m_key in all_model_keys:
        for d_key in all_ds_keys:
            best_val = 1e9
            for meth in methods:
                val = stats[m_key][meth].get(d_key, float("nan"))
                if not np.isnan(val) and val < best_val:
                    best_val = val
            winners[(m_key, d_key)] = best_val

    # --- Write LaTeX ---
    lines = []
    col_spec = "l" + "c" * len(datasets) + "c"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header
    header = r"\textbf{Method} & " + " & ".join([rf"\textbf{{{latex_escape(ds)}}}" for ds in datasets]) + r" & \textbf{Mean} \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Body: Per Model
    for model in models:
        lines.append(rf"\multicolumn{{{len(datasets) + 2}}}{{l}}{{\textbf{{{latex_escape(model)}}}}} \\")
        for meth in methods:
            row = [latex_escape(meth)]
            for d_key in all_ds_keys:
                val = stats[model][meth].get(d_key, float("nan"))
                if np.isnan(val):
                    row.append("-")
                else:
                    s = f"{val:.3f}"
                    # Bold if within epsilon of winner
                    if val <= winners[(model, d_key)] + 1e-6:
                        s = rf"\textbf{{{s}}}"
                    row.append(s)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    # Bottom: Overall Average
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{len(datasets) + 2}}}{{l}}{{\textbf{{Overall Average (Across Models)}}}} \\")
    for meth in methods:
        row = [latex_escape(meth)]
        for d_key in all_ds_keys:
            val = stats["AVERAGE_ALL"][meth].get(d_key, float("nan"))
            if np.isnan(val):
                row.append("-")
            else:
                s = f"{val:.3f}"
                if val <= winners[("AVERAGE_ALL", d_key)] + 1e-6:
                    s = rf"\textbf{{{s}}}"
                row.append(s)
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "naurcc_table.tex", "w") as f:
        f.write("\n".join(lines))

def write_correlation_table(df_corr: pd.DataFrame, figures_dir: Path) -> None:
    """Writes a summary table of Pearson correlations between SE and Acc Probe."""
    if df_corr.empty: return

    datasets = sorted(df_corr["Dataset"].unique())
    models = sorted(df_corr["Model"].unique())

    lines = []
    col_spec = "l" + "c" * len(datasets)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & " + " & ".join([rf"\textbf{{{latex_escape(ds)}}}" for ds in datasets]) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        lines.append(rf"\textbf{{{latex_escape(model)}}}")
        row = []
        for ds in datasets:
            val = df_corr[(df_corr["Model"] == model) & (df_corr["Dataset"] == ds)]["Correlation"]
            if val.empty:
                row.append("-")
            else:
                row.append(f"{val.iloc[0]:.2f}")
        lines.append(" & " + " & ".join(row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "correlation_table.tex", "w") as f:
        f.write("\n".join(lines))

def write_calibration_error_table(df_cal: pd.DataFrame, figures_dir: Path, included_datasets: Optional[List[str]] = None) -> None:
    """
    Writes a table comparing Calibration Error (MACE) and Safety Violation (Max Excess Risk).
    Robust to cases where one mode (e.g. UCB) yields 0 coverage and is filtered out.
    """
    # Filter for valid coverage
    df = df_cal[df_cal["Coverage"] > 0.001].copy()
    if df.empty: return

    # Determine which datasets to process
    available_datasets = sorted(df["Dataset"].unique())
    if included_datasets is not None:
        datasets = [d for d in available_datasets if d in included_datasets]
        if not datasets:
            print(f"Warning: No matching datasets found for calibration table. Available: {available_datasets}")
            return
    else:
        datasets = available_datasets

    models = sorted(df["Model"].unique())
    
    # Pre-calculate stats
    stats = {}
    
    for m in models:
        stats[m] = {}
        for d in datasets:
            stats[m][d] = {}
            for meth in METHOD_ORDER:
                subset = df[(df["Model"] == m) & (df["Dataset"] == d) & (df["Method"] == meth)]
                
                res = {}
                for mode in ["Empirical", "Conservative"]:
                    sub_mode = subset[subset["CalMode"] == mode]
                    if sub_mode.empty:
                        res[mode] = (np.nan, np.nan)
                    else:
                        diffs = sub_mode["Realized"] - sub_mode["Target"]
                        mace = np.mean(np.abs(diffs))
                        max_excess = np.max(diffs)
                        res[mode] = (mace, max_excess)
                
                stats[m][d][meth] = res

    # --- Write LaTeX ---
    lines = []
    lines.append(r"\begin{tabular}{l|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c|}{\textbf{Target Adherence (Empirical)}} & \multicolumn{2}{c}{\textbf{Strict Safety (UCB)}} \\")
    lines.append(r"\textbf{Method} & \textbf{MACE} $\downarrow$ & \textbf{Max Excess} $\downarrow$ & \textbf{MACE} & \textbf{Max Excess} $\downarrow$ \\")
    lines.append(r"\midrule")

    # Helper for formatting
    def fmt_excess(val):
        if np.isnan(val): return "-"
        s = f"{val:+.3f}"
        if val > 0.005: return rf"\textcolor{{red}}{{{s}}}" 
        if val < 0: return rf"\textcolor{{blue}}{{{s}}}"
        return s

    for m in models:
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{latex_escape(m)}}}}} \\")
        
        for meth in METHOD_ORDER:
            mace_emp_vals = []
            exc_emp_vals = []
            mace_ucb_vals = []
            exc_ucb_vals = []
            
            for d in datasets:
                if meth in stats[m][d]:
                    e_m, e_x = stats[m][d][meth]["Empirical"]
                    u_m, u_x = stats[m][d][meth]["Conservative"]
                    
                    if not np.isnan(e_m):
                        mace_emp_vals.append(e_m)
                        exc_emp_vals.append(e_x)
                    if not np.isnan(u_m):
                        mace_ucb_vals.append(u_m)
                        exc_ucb_vals.append(u_x)

            # Skip row only if BOTH modes are empty
            if not mace_emp_vals and not mace_ucb_vals:
                continue

            # Compute Empirical Stats
            if mace_emp_vals:
                avg_mace_emp = np.mean(mace_emp_vals)
                max_exc_emp = np.max(exc_emp_vals)
                str_mace_emp = f"{avg_mace_emp:.3f}"
                str_exc_emp = fmt_excess(max_exc_emp)
            else:
                str_mace_emp = "-"
                str_exc_emp = "-"

            # Compute UCB Stats (Handle empty case safely)
            if mace_ucb_vals:
                avg_mace_ucb = np.mean(mace_ucb_vals)
                max_exc_ucb = np.max(exc_ucb_vals)
                str_mace_ucb = f"{avg_mace_ucb:.3f}"
                str_exc_ucb = fmt_excess(max_exc_ucb)
            else:
                str_mace_ucb = "-"
                str_exc_ucb = "-"

            row = [
                latex_escape(meth),
                str_mace_emp,
                str_exc_emp,
                str_mace_ucb,
                str_exc_ucb
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "calibration_error_table.tex", "w") as f:
        f.write("\n".join(lines))

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, required=True)
    parser.add_argument("--figures-dirname", type=str, default="analysis_output_v4")
    # parser.add_argument("--embedding-key", type=str, default="emb_last_tok_before_gen")
    parser.add_argument("--n-cap", type=int, default=2000)
    parser.add_argument("--hall-acc-threshold", type=float, default=0.99)

    parser.add_argument("--use-ucb", action="store_true")
    parser.add_argument("--delta", type=float, default=0.1)

    parser.add_argument("--decision-plot-method", type=str, default="MLP", choices=["LR", "MLP", "GBM"])
    parser.add_argument("--decision-plot-alpha", type=float, default=0.10)
    parser.add_argument(
    "--retrain-probes",
    action="store_true",
    help="Force retraining probes by calling train_probes.py before analysis."
)

    parser.add_argument(
        "--train-probes-script",
        type=str,
        default="scripts/train_probes.py",
        help="Path to train_probes.py"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config YAML used for training probes (same as generation)."
    )

    parser.add_argument(
        "--position",
        type=str,
        default="tbg",
        choices=["tbg", "slt"],
        help="Which probe position to use: "
            "tbg = token before generation (default), "
            "slt = second-to-last token in answer."
    )



    args = parser.parse_args()

    repo_root, runs_root = resolve_paths(args.runs_root)
    figures_dir = repo_root / args.figures_dirname
    figures_dir.mkdir(exist_ok=True, parents=True)

    det_results = []
    cal_results = []
    layer_results = []
    curve_data: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {}
    aurc_results = []
    corr_results = [] 
    base_risks = {}

    print(f"Scanning {runs_root} ...")
    if args.use_ucb:
        print(f"Using UCB selection with delta={args.delta}")

    # Before iterating over models/datasets
    maybe_train_all_probes(
        runs_root=runs_root,
        cfg_yaml=Path(args.config),
        train_probes_script=Path(args.train_probes_script),
        retrain=args.retrain_probes,
    )

    for model_name in DEFAULT_TARGET_MODELS:
        safe_model = Path(model_name).name
        curve_data[safe_model] = {}

        for dataset_name in DEFAULT_TARGET_DATASETS:
            run_dir = find_run_directory(runs_root, model_name, dataset_name)
            if not run_dir:
                continue

            artifacts = load_run_artifacts(run_dir)
            if artifacts is None:
                continue

            gens = artifacts["gens"]
            unc = artifacts["unc"]
            probes_obj = artifacts["probes"]

            embedding_key = (
                "emb_last_tok_before_gen"
                if args.position == "tbg"
                else "emb_tok_before_eos"
            )

            try:
                X, y_hall, se_raw, ids = extract_features_aligned(
                    gens, unc,
                    embedding_key=embedding_key,
                    n_cap=args.n_cap,
                    hall_acc_threshold=args.hall_acc_threshold,
                )
            except Exception as e:
                print(f"[skip] {safe_model}/{dataset_name} feature extraction failed: {e}")
                continue

            # load splits + probes (from probes.pkl)
            try:
                splits = load_splits_from_probes(probes_obj)
                bundle = load_probe_bundle(probes_obj, position=args.position)

            except Exception as e:
                print(f"[skip] {safe_model}/{dataset_name} probes.pkl parse failed: {e}")
                continue

            # layer sensitivity data
            layer_results.append({
                "Model": safe_model,
                "Dataset": dataset_name,
                "acc_cv_aucs": bundle.get("acc_cv_aucs", []),
                "se_cv_aucs": bundle.get("se_cv_aucs", []),
            })

            # Build df with split labels
            n = len(y_hall)
            df = pd.DataFrame(index=np.arange(n))
            df["split"] = "unused"
            for k in ["train", "cal", "test"]:
                idx = splits[k]
                idx = idx[idx < n]  # safety if n_cap truncated
                df.loc[idx, "split"] = k

            df["y_hall"] = y_hall.astype(np.int64)
            df["se_raw"] = se_raw.astype(np.float32)

            # Probe scores using stored probe models
            acc_layer = int(bundle["acc_best_layer"])
            se_layer = int(bundle["se_best_layer"])
            acc_model = bundle["acc_model"]
            se_model = bundle["se_model"]

            # Make sure layer indices are in range (defensive)
            acc_layer = max(0, min(acc_layer, X.shape[0] - 1))
            se_layer = max(0, min(se_layer, X.shape[0] - 1))

            df["p_correct"] = acc_model.predict_proba(X[acc_layer])[:, 1].astype(np.float32)
            df["p_high_entropy"] = se_model.predict_proba(X[se_layer])[:, 1].astype(np.float32)

            # Train combiners on TRAIN only (no leakage into cal/test)
            train_df = df[df["split"] == "train"]
            X_train = train_df[["p_correct", "p_high_entropy"]]
            y_train = train_df["y_hall"]

            combiners = {
                "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
                "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)),
                # "GBM": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
            }
            fitted_combiners = {}

            for name, model in combiners.items():
                if len(np.unique(y_train)) < 2:
                    df[f"p_halluc_{name}"] = 0.5
                    fitted_combiners[name] = None
                else:
                    model.fit(X_train, y_train)
                    df[f"p_halluc_{name}"] = model.predict_proba(df[["p_correct", "p_high_entropy"]])[:, 1].astype(np.float32)
                    fitted_combiners[name] = model

            # Risk scores (lower = safer)
            df["score_SE"] = df["se_raw"]
            df["score_Acc"] = 1.0 - df["p_correct"]
            df["score_SE_Probe"] = df["p_high_entropy"]
            df["score_Comb_LR"] = df["p_halluc_LR"]
            df["score_Comb_MLP"] = df["p_halluc_MLP"]

            method_map = {
                "Semantic Entropy": "score_SE",
                "Accuracy Probe": "score_Acc",
                "SE Probe": "score_SE_Probe",
                "Combined (LR)": "score_Comb_LR",
                "Combined (MLP)": "score_Comb_MLP",
            }

            df_test = df[df["split"] == "test"].copy()
            if df_test.empty:
                continue

            # Calculate Correlation (Test Set)
            if not df_test.empty:
                corr, _ = pearsonr(df_test["score_Acc"], df_test["score_SE"]) # Note: check signs. 
                # score_Acc is risk (1-p), score_SE is risk. Should be positive correlation.

    
                corr_results.append({
                    "Model": safe_model,
                    "Dataset": dataset_name,
                    "Correlation": corr
                })
                
                # Save scatter plot for the first dataset or specific ones
                if dataset_name == "trivia_qa":  # or always
                    plot_correlation_scatter(
                        df_test, safe_model, dataset_name, 
                        figures_dir / f"scatter_{safe_model}_{dataset_name}.png"
                    )

            # confident subset defined on test only
            q30 = df_test["se_raw"].quantile(0.30)
            df_conf = df_test[df_test["se_raw"] <= q30].copy()

            curve_data[safe_model][dataset_name] = {}
            base_risks[(safe_model, dataset_name)] = float(df_test["y_hall"].mean())

            # Metrics + curves + AURC + calibration sweeps
            dense_alphas = np.linspace(0.05, 0.4, 40)
            table_alphas = np.array([])
            all_alphas = np.unique(np.sort(np.concatenate([dense_alphas, table_alphas])))

            for m_name, col in method_map.items():
                # AUROC
                try:
                    auc_full = roc_auc_score(df_test["y_hall"], df_test[col])
                except Exception:
                    auc_full = 0.5

                try:
                    if len(df_conf) > 0 and len(np.unique(df_conf["y_hall"])) > 1:
                        auc_conf = roc_auc_score(df_conf["y_hall"], df_conf[col])
                    else:
                        auc_conf = 0.5
                except Exception:
                    auc_conf = 0.5

                # AUPRC
                try:
                    # average_precision_score takes (y_true, y_score)
                    # y_hall=1 is positive class. 
                    # scores are risk scores (high = hallucination). This matches.
                    auprc_full = average_precision_score(df_test["y_hall"], df_test[col])
                except Exception:
                    auprc_full = 0.0

                det_results.append({"Model": safe_model, "Dataset": dataset_name, "Method": m_name, "Subset": "Full", "AUROC": float(auc_full),  "AUPRC": float(auprc_full)})
                det_results.append({"Model": safe_model, "Dataset": dataset_name, "Method": m_name, "Subset": "Confident", "AUROC": float(auc_conf)})

                # risk-coverage curve (test)
                covs, risks = get_risk_coverage_curve(df_test, col)
                curve_data[safe_model][dataset_name][m_name] = (covs, risks)

                base_risk = base_risks[(safe_model, dataset_name)]

                # AURC raw
                aurc_val = float(np.trapezoid(risks, covs)) if len(covs) > 1 else float("nan")

                # Compute Normalized AURCC
                naurc_val = compute_normalized_aurcc(covs, risks, base_risk)

                aurc_results.append({
                    "Model": safe_model, 
                    "Dataset": dataset_name, 
                    "Method": m_name, 
                    "AURC": aurc_val,
                    "nAURCC": naurc_val,
                    "Correlation": corr  # Save correlation to this row (repeated)
                })

                # calibration: threshold on cal, evaluate on test

                # Run Calibration TWICE
                modes = [("Empirical", False), ("Conservative", True)]
                for mode_name, use_ucb_flag in modes:
                    for alpha in all_alphas:
                        r, c = eval_calibration(df, col, float(alpha), args.delta, use_ucb=use_ucb_flag)
                        cal_results.append({
                            "Model": safe_model,
                            "Dataset": dataset_name,
                            "Method": m_name,
                            "Target": float(alpha),
                            "CalMode": mode_name,
                            "Realized": float(r),
                            "Coverage": float(c),
                        })

            # Decision boundary plot for requested combiner
            dm = args.decision_plot_method
            if dm in ["LR", "MLP"]:
                comb = fitted_combiners.get(dm)
                if comb is not None:
                    score_col = "score_Comb_LR" if dm == "LR" else "score_Comb_MLP"
                    out_path = figures_dir / f"boundary_{safe_model}_{dataset_name}_{dm}.png"
                    plot_decision_boundary(
                        df=df,
                        combiner_model=comb,
                        score_col=score_col,
                        alpha=float(args.decision_plot_alpha),
                        delta=args.delta,
                        use_ucb=args.use_ucb,
                        out_path=out_path,
                        title=f"{safe_model} / {dataset_name} — Decision Boundary ({dm})",
                    )

    # Save CSVs
    df_det = pd.DataFrame(det_results)
    df_cal = pd.DataFrame(cal_results)
    df_aurc = pd.DataFrame(aurc_results)
    df_corr = pd.DataFrame(corr_results) 

    df_det.to_csv(figures_dir / "auroc_summary.csv", index=False)
    df_cal.to_csv(figures_dir / "calibration_summary.csv", index=False)
    df_aurc.to_csv(figures_dir / "aurc_summary.csv", index=False)
    df_corr.to_csv(figures_dir / "correlation_summary.csv", index=False)

    # Plots
    print("Generating plots...")
    if not df_det.empty:
        plot_detection_bars(df_det, figures_dir)
    if layer_results:
        plot_layer_sensitivity(layer_results, figures_dir)
    if curve_data:
        plot_risk_coverage(curve_data, figures_dir, base_risks)
    if not df_cal.empty:
        plot_calibration_curves(df_cal, figures_dir)

    # Tables
    print("Generating LaTeX tables...")
    if not df_det.empty:
        write_detection_table(df_det, figures_dir)
    if not df_aurc.empty:
        write_aurc_table(df_aurc, figures_dir)
        write_naurcc_table(df_aurc, figures_dir)
    if not df_cal.empty:
        # write_calibration_tables(df_cal, figures_dir, [0.05, 0.10, 0.20, 0.25])
        write_calibration_error_table(df_cal, figures_dir,
                                      included_datasets=["trivia_qa"])
    if not df_corr.empty:       
        write_correlation_table(df_corr, figures_dir)
        

    print(f"Done! Artifacts saved to: {figures_dir}")


if __name__ == "__main__":
    main()
