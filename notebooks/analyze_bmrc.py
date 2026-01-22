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

from email import parser
import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import subprocess
import shutil

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
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
    # "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    # "Qwen/Qwen3-8B",
    "google/gemma-3-4b-it",
    "google/gemma-7b-it",
    # "HuggingFaceTB/SmolLM3-3B",
    "mistralai/Ministral-8B-Instruct-2410",
]

DEFAULT_TARGET_DATASETS = [
    "trivia_qa",
    "bioasq",
    "medical_o1",
]


# -----------------------------
# Display names + global style
# -----------------------------

# Keys here should match `safe_model = Path(model_name).name`
PRETTY_MODEL_NAMES = {
    "Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "Qwen3-4B-Instruct-2507": "Qwen3 4B",
    "gemma-3-4b-it": "Gemma 3 4B",
    "gemma-7b-it": "Gemma 7B",
    "Ministral-8B-Instruct-2410": "Ministral 8B",
}

PRETTY_DATASET_NAMES = {
    "trivia_qa": "TriviaQA",
    "bioasq": "BioASQ",
    "medical_o1": "MedicalQA",
}

METHOD_DISPLAY_NAMES = {
    "Accuracy Probe": "PC Probe",
}

def pretty_method(m: str) -> str:
    return METHOD_DISPLAY_NAMES.get(m, m)

def pretty_model(model_key: str) -> str:
    return PRETTY_MODEL_NAMES.get(model_key, model_key)

def pretty_dataset(ds_key: str) -> str:
    return PRETTY_DATASET_NAMES.get(ds_key, ds_key)

def file_safe(s: str) -> str:
    """
    Make a string safe for filenames.
    Keeps alphanumerics, dash, underscore.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def configure_plotting(font_scale: float, base_fontsize: Optional[int], context: str) -> None:
    """
    Centralized plotting config. Use CLI flags to increase fonts e.g. for posters.
    """
    sns.set_theme(style="whitegrid", context=context, font_scale=font_scale)
    plt.rcParams["font.family"] = "serif"
    # Nice-to-have for publication (editable text in PDFs if you save pdfs later)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if base_fontsize is not None:
        plt.rcParams.update({
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize * 1.0,
            "axes.labelsize": base_fontsize * 1.0,
            "xtick.labelsize": base_fontsize * 1.0,
            "ytick.labelsize": base_fontsize * 1.0,
            "legend.fontsize": base_fontsize * 1.0,
            "figure.titlesize": base_fontsize * 1.0,
        })

def apply_tex_font_profile(profile: str) -> None:
    """
    Switch figure typography to match LaTeX doc fonts.
    Requires a working LaTeX install if text.usetex=True.
    Profiles:
      - "paper_times": Times-like (newtxtext/newtxmath)
      - "poster_lmodern": Latin Modern (lmodern / Computer Modern family)
      - "none": don't use LaTeX rendering
    """
    if profile == "none":
        plt.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "axes.unicode_minus": False,
        })
        return

    # default: enable latex rendering (paper), can be overridden per profile
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if profile == "paper_times":
        # Match `\usepackage{iclr2026_conference,times}`
        # (iclr style sets layout; here we only mirror font selection)
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{times}"
        )
    elif profile == "poster_gemini":
        # Match Gemini theme: Lato for body/sans; use plain mathtext.
        # This avoids needing xelatex/fontspec inside matplotlib.
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Lato", "Raleway", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
        })
    else:
        raise ValueError(f"Unknown tex font profile: {profile}")


COLORS = {
    "Semantic Entropy": "#7f8c8d",
    "Accuracy Probe": "#3498db",
    "PC Probe": "#3498db",
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
    Loads artifacts with verbose error reporting and fallback logic.
    """
    # 1. Try to find the 'files' directory
    files_dir = run_dir / "files"
    
    # Handle broken symlinks from SCP/Download:
    if files_dir.is_symlink() and not files_dir.exists():
        print(f"  [Warning] '{files_dir}' is a broken symlink. Searching for real files...")
        files_dir = run_dir # Fallback to looking in the run root

    if not files_dir.exists():
        # Fallback: WandB sometimes nests in 'wandb/latest-run/files'
        candidates = list(run_dir.rglob("uncertainty_measures.pkl"))
        if candidates:
            files_dir = candidates[0].parent
            print(f"  [Info] Found artifacts in nested path: {files_dir}")
        else:
            print(f"  [Skip] No 'files' directory or artifacts found in {run_dir.name}")
            return None

    # 2. Define expected paths
    p_gens = files_dir / "validation_generations.pkl"
    p_unc = files_dir / "uncertainty_measures.pkl"
    p_probes = files_dir / "probes.pkl"

    # 3. Check existence and print SPECIFIC missing file
    missing = []
    if not p_gens.exists(): missing.append("validation_generations.pkl")
    if not p_unc.exists(): missing.append("uncertainty_measures.pkl")
    if not p_probes.exists(): missing.append("probes.pkl")

    if missing:
        print(f"  [Skip] {run_dir.name} is missing files: {', '.join(missing)}")
        if "probes.pkl" in missing:
            print(f"         (Hint: Did you download 'probes.pkl'? Or run with --retrain-probes?)")
        return None

    # 4. Load
    try:
        with p_gens.open("rb") as f:
            gens = pickle.load(f)
        with p_unc.open("rb") as f:
            unc = pickle.load(f)
        with p_probes.open("rb") as f:
            probes = pickle.load(f)
    except Exception as e:
        print(f"  [Error] Failed to pickle load in {run_dir.name}: {e}")
        return None

    return {
        "files_dir": files_dir,
        "gens": gens,
        "unc": unc,
        "probes": probes,
    }

def save_figure(path_no_ext: Path, dpi: int = 300) -> None:
    """
    Save figures robustly.
    - Always saves PDF (best for papers/posters, no dvipng required).
    - Saves PNG only if dvipng is available or usetex is off.
    """
    # Vector: best for publications
    plt.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight")

    use_tex = bool(plt.rcParams.get("text.usetex", False))
    has_dvipng = shutil.which("dvipng") is not None
    if (not use_tex) or has_dvipng:
        plt.savefig(path_no_ext.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

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

def plot_detection_bars(df_det: pd.DataFrame, figures_dir: Path, model_order: List[str], dataset_order: List[str], dpi: int) -> None:
    # enforce categorical ordering on pretty labels
    ds_pretty_order = [pretty_dataset(d) for d in dataset_order]
    for model_key in model_order:
        for subset in ["Full", "Confident"]:
            data = df_det[(df_det["ModelKey"] == model_key) & (df_det["Subset"] == subset)].copy()
            if data.empty:
                continue
            data["Dataset"] = pd.Categorical(data["Dataset"], categories=ds_pretty_order, ordered=True)
            plt.figure(figsize=(9, 6))
            data = data.copy()
            data["MethodDisplay"] = data["Method"].map(pretty_method)
            sns.barplot(
                data=data,
                x="Dataset",
                y="AUROC",
                hue="MethodDisplay",
                palette=COLORS,
                edgecolor="black",
                errorbar=None,
                hue_order=[pretty_method(m) for m in METHOD_ORDER if m in data["Method"].unique()]
            )
            plt.title(f"{pretty_model(model_key)} - {subset} Subset")
            plt.ylim(0.4, 1.0)
            # Legend inside plot area
            leg = plt.legend(
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),   # inside axes coords
                borderaxespad=0.0,
                frameon=True,
                framealpha=0.95,
                title=None,
            )
            # Make legend box a bit more compact
            for lh in leg.legend_handles:
                try:
                    lh.set_alpha(1.0)
                except Exception:
                    pass
            plt.tight_layout()

            save_figure((figures_dir / f"detection_{file_safe(model_key)}_{subset}").with_suffix(""), dpi=dpi)
            plt.close()


def plot_layer_sensitivity(layer_results: List[dict], figures_dir: Path, dpi: int) -> None:
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
        safe_model = item["ModelKey"]
        ds_key = item["DatasetKey"]
        save_figure((figures_dir / f"layers_{file_safe(safe_model)}_{file_safe(ds_key)}").with_suffix(""), dpi=dpi)
        plt.close()


def plot_risk_coverage(curve_data: dict, figures_dir: Path, base_risks: dict, dpi: int) -> None:
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
                plt.plot(
                    cov, risk,
                    label=pretty_method(m_name),
                    color=COLORS.get(pretty_method(m_name), COLORS.get(m_name, "black")),
                    linewidth=2.5,
                )
                if len(risk) > 0:
                    max_r = max(max_r, max(risk))

            plt.xlabel("Coverage")
            plt.ylabel("Hallucination Rate")
            plt.title(f"Risk-Coverage: {pretty_model(model)} / {pretty_dataset(ds)}")
            leg = plt.legend(
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),   # inside axes coords
                borderaxespad=0.0,
                frameon=True,
                framealpha=0.95,
                title=None,
            )
            # Make legend box a bit more compact
            for lh in leg.legend_handles:
                try:
                    lh.set_alpha(1.0)
                except Exception:
                    pass
            plt.xlim(0, 1)

            top_lim = min(1.0, max_r * 1.15) if max_r > 0 else 1.0
            plt.ylim(0, top_lim)
            plt.grid(True, alpha=0.3)

            save_figure(
                (figures_dir / f"rc_{file_safe(model)}_{file_safe(ds)}").with_suffix(""),
                dpi=dpi,
            )
            plt.close()


def plot_calibration_curves(df_cal: pd.DataFrame, figures_dir: Path, model_order: List[str], dataset_order: List[str], dpi: int) -> None:
    """
    Plots Target risk (x) vs Realized risk (y).
    One Figure per (Model, Dataset).
    Subplots: [Empirical (Target Adherence)] [Conservative (Safety/UCB)]
    """
     # iterate in requested order, using keys
    unique_models = model_order
    unique_datasets = dataset_order

    for model in unique_models:
        for ds in unique_datasets:
            data = df_cal[(df_cal["ModelKey"] == model) & (df_cal["DatasetKey"] == ds)]
            if data.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            
            # Subplot 1: Empirical
            ax_emp = axes[0]
            data_emp = data[data["CalMode"] == "Empirical"]
            data_emp = data_emp.copy()
            data_emp["MethodDisplay"] = data_emp["Method"].map(pretty_method)
            
            ax_emp.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
            if not data_emp.empty:
                sns.lineplot(
                    data=data_emp, x="Target", y="Realized", hue="MethodDisplay",
                    palette=COLORS, linewidth=2.5, ax=ax_emp, estimator=None
                )
            ax_emp.set_title("Target Adherence (Empirical)")
            ax_emp.set_xlabel(r"Target Risk ($\alpha$)")
            ax_emp.set_ylabel("Realized Hallucination Rate")
            ax_emp.set_xlim(0, 0.4)
            ax_emp.set_ylim(0, 0.4)
            ax_emp.grid(True, alpha=0.3)
            ax_emp.get_legend().remove()

            # Subplot 2: Conservative (UCB)
            ax_ucb = axes[1]
            data_ucb = data[data["CalMode"] == "Conservative"]
            data_ucb = data_ucb.copy()
            data_ucb["MethodDisplay"] = data_ucb["Method"].map(pretty_method)
            
            ax_ucb.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
            if not data_ucb.empty:
                sns.lineplot(
                    data=data_ucb, x="Target", y="Realized", hue="MethodDisplay",
                    palette=COLORS, linewidth=2.5, ax=ax_ucb, estimator=None
                )
            ax_ucb.set_title(r"Strict Safety (UCB, $\delta=0.1$)")
            ax_ucb.set_xlabel(r"Target Risk ($\alpha$)")
            ax_ucb.set_xlim(0, 0.4)
            ax_ucb.set_ylim(0, 0.4)
            ax_ucb.grid(True, alpha=0.3)
            
            # Unified Legend
            handles, labels = ax_ucb.get_legend_handles_labels()
            ax_ucb.get_legend().remove()
            
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels), frameon=False)
            
            fig.suptitle(f"Calibration: {pretty_model(model)} / {pretty_dataset(ds)}", y=0.98, fontsize=14)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2) # Make room for legend

            save_figure((figures_dir / f"calibration_{file_safe(model)}_{file_safe(ds)}").with_suffix(""), dpi=dpi)
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
    dpi: int,
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
    save_figure(out_path.with_suffix(""), dpi=dpi)
    plt.close()

def plot_correlation_scatter(df: pd.DataFrame, model_name: str, ds_name: str, out_path: Path, dpi: int) -> None:
    """
    Plots Semantic Entropy vs Accuracy Probe Logits.
    Clean, consistent style with other figures.
    """
    # 1. Prepare Data for Plotting
    # Map 0/1 to strings for the Legend
    plot_df = df.copy()
    plot_df["Label"] = plot_df["y_hall"].map({0: "Correct", 1: "Hallucination"})
    
    # Jitter SE slightly to visualize density at 0.0
    # Clip to [0, inf) to keep it logical
    jitter = np.random.normal(0, 0.008, size=len(plot_df))
    plot_df["se_jittered"] = np.clip(plot_df["se_raw"] + jitter, 0, None)

    # 2. Setup Plot
    plt.figure(figsize=(8, 6))
    
    # Palette: Royal Blue (Correct) vs Red/Tomato (Hallucination)
    custom_palette = {"Correct": "#4169E1", "Hallucination": "#E74C3C"}

    # 3. Scatter Plot
    sns.scatterplot(
        data=plot_df,
        x="se_raw",
        y="logit_correct",
        hue="Label",
        style="Label",
        markers={"Correct": "o", "Hallucination": "o"}, # Keep shape simple
        palette=custom_palette,
        alpha=0.5,       # Transparency is key for density
        sizes=(100, 100),
        size="Label",
        edgecolor="w",   # Slight white edge helps separation
        linewidth=0.3
    )

    # 4. Reference Lines
    # Vertical: 50th percentile SE (Confident Subset boundary)
    q50 = plot_df["se_raw"].quantile(0.50)
    plt.axvline(x=q50, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label="50th %tile SE")
    
    # # Horizontal: Logit = 0 (Probe Decision Boundary)
    # plt.axhline(y=0, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)

    # 5. Labels & Formatting
    plt.xlabel("Semantic Entropy")
    plt.ylabel("Correctness Logit")
    plt.title(f"{model_name} / {ds_name}")

    # Move legend to best spot, usually upper right for this distribution
    plt.legend(title=None, loc="upper right", frameon=True, framealpha=0.95)
    
    plt.grid(True, alpha=0.3)
    plt.xlim(left=-0.05)
    
    # Adjust Y limits to offer a bit of headroom
    y_min, y_max = plot_df["logit_correct"].min(), plot_df["logit_correct"].max()
    plt.ylim(y_min - 0.5, y_max + 0.5)

    plt.tight_layout()
    save_figure(out_path.with_suffix(""), dpi=dpi)
    plt.close()

# -----------------------------
# LaTeX Tables
# -----------------------------

def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write_detection_table(
    df_det: pd.DataFrame,
    base_risks: Dict[Tuple[str, str], float],
    figures_dir: Path,
    model_order: List[str],
    dataset_order: List[str],
) -> None:

    """
    Writes detection metrics table (AUROC / AUPRC).
    - Rows: Methods grouped by Model.
    - Columns: Datasets (AUROC, AUPRC).
    - Extra:
      1. Model Header shows Base Accuracy (1 - base_risk).
      2. Bottom block shows Average across models.
      3. Bold = Best Overall. Italic = Best Single-Pass (non-SE).
    """
    df = df_det[df_det["Method"].isin(METHOD_ORDER)].copy()
    if df.empty: return

    # Filter to Full subset only
    df = df[df["Subset"] == "Full"]

    # Use user-requested order (keys), but render pretty names in headers
    datasets = [d for d in dataset_order if d in df["DatasetKey"].unique()]
    models = [m for m in model_order if m in df["ModelKey"].unique()]
    methods = [m for m in METHOD_ORDER if m in df["Method"].unique()]
    
    # Define Single-Pass methods for italics logic
    single_pass_methods = [m for m in methods if m != "Semantic Entropy"]

    # --- Pre-calculate Stats & Winners ---
    # Structure: stats[model_key][method][dataset] = {AUROC: x, AUPRC: y}
    # model_key can be a specific model name or "AVERAGE_ALL"
    stats = {}

    # 1. Individual Models
    for m in models:
        stats[m] = {}
        for meth in methods:
            stats[m][meth] = {}
            for ds in datasets:
                row = df[(df["ModelKey"] == m) & (df["DatasetKey"] == ds) & (df["Method"] == meth)]
                if not row.empty:
                    stats[m][meth][ds] = {
                        "AUROC": float(row.iloc[0]["AUROC"]),
                        "AUPRC": float(row.iloc[0]["AUPRC"])
                    }
                else:
                    stats[m][meth][ds] = None

    # 2. Averages
    stats["AVERAGE_ALL"] = {}
    for meth in methods:
        stats["AVERAGE_ALL"][meth] = {}
        for ds in datasets:
            # Gather all model values for this method/dataset
            aurocs = []
            auprcs = []
            for m in models:
                val = stats[m][meth][ds]
                if val:
                    aurocs.append(val["AUROC"])
                    auprcs.append(val["AUPRC"])
            
            if aurocs:
                stats["AVERAGE_ALL"][meth][ds] = {
                    "AUROC": np.mean(aurocs),
                    "AUPRC": np.mean(auprcs)
                }
            else:
                stats["AVERAGE_ALL"][meth][ds] = None

    # 3. Identify Winners (Max)
    # winners[context_key][dataset][metric] = {overall_max: x, single_pass_max: y}
    winners = {}
    all_contexts = models + ["AVERAGE_ALL"]
    
    for ctx in all_contexts:
        winners[ctx] = {}
        for ds in datasets:
            winners[ctx][ds] = {"AUROC": {"all": -1, "sp": -1}, "AUPRC": {"all": -1, "sp": -1}}
            
            # Find maxes
            for metric in ["AUROC", "AUPRC"]:
                # Overall Max
                valid_vals = [stats[ctx][m][ds][metric] for m in methods if stats[ctx][m][ds] is not None]
                if valid_vals:
                    winners[ctx][ds][metric]["all"] = max(valid_vals)
                
                # Single Pass Max
                valid_sp = [stats[ctx][m][ds][metric] for m in single_pass_methods if stats[ctx][m][ds] is not None]
                if valid_sp:
                    winners[ctx][ds][metric]["sp"] = max(valid_sp)

    # --- Write LaTeX ---
    lines = []
    # Column setup: Method | (AUROC AUPRC) * n_datasets
    col_spec = "l" + "cc" * len(datasets)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header 1: Datasets
    header_1 = [r"\textbf{Method}"]
    for ds in datasets:
        header_1.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{latex_escape(pretty_dataset(ds))}}}}}")
    lines.append(" & ".join(header_1) + r" \\")

    # Header 2: Metrics
    header_2 = [r""]
    for _ in datasets:
        header_2.extend([r"\scriptsize{AUROC}", r"\scriptsize{AUPRC}"])
    lines.append(" & ".join(header_2) + r" \\")
    lines.append(r"\midrule")

    # Helper for formatting cell
    def format_cell(val, best_all, best_sp, is_single_pass):
        if val is None: return "-"
        s = f"{val:.3f}"
        
        is_best_overall = (val >= best_all - 1e-6)
        is_best_sp = (is_single_pass and val >= best_sp - 1e-6)
        
        if is_best_overall and is_best_sp:
            # Winner of both categories: Bold + Italic
            return rf"\textit{{\textbf{{{s}}}}}"
        elif is_best_overall:
            # Overall winner only (e.g. Semantic Entropy)
            return rf"\textbf{{{s}}}"
        elif is_best_sp:
            # Single-pass winner only
            return rf"\textit{{{s}}}"
        
        return s

    # BODY: Per Model
    for model in models:
        # Model Header Row with Base Accuracy
        row_header = [rf"\textbf{{{latex_escape(pretty_model(model))}}}"]
        for ds in datasets:
            # Calculate Base Accuracy: 1 - Risk
            risk = base_risks.get((model, ds), None)
            if risk is not None:
                acc = (1.0 - risk) * 100
                acc_str = f"Acc: {acc:.1f}"
            else:
                acc_str = "Acc: -"
            row_header.append(rf"\multicolumn{{2}}{{c}}{{\scriptsize{{{acc_str}}}}}")
        
        lines.append(" & ".join(row_header) + r" \\")

        # Method Rows
        for meth in methods:
            row = [latex_escape(pretty_method(meth))]
            is_sp = (meth in single_pass_methods)
            
            for ds in datasets:
                dat = stats[model][meth][ds]
                if dat:
                    w = winners[model][ds]
                    row.append(format_cell(dat["AUROC"], w["AUROC"]["all"], w["AUROC"]["sp"], is_sp))
                    row.append(format_cell(dat["AUPRC"], w["AUPRC"]["all"], w["AUPRC"]["sp"], is_sp))
                else:
                    row.extend(["-", "-"])
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    # BOTTOM: Overall Average
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{1 + 2*len(datasets)}}}{{l}}{{\textbf{{Overall Average (Across Models)}}}} \\")
    
    for meth in methods:
        row = [latex_escape(pretty_method(meth))]
        is_sp = (meth in single_pass_methods)
        
        for ds in datasets:
            dat = stats["AVERAGE_ALL"][meth][ds]
            if dat:
                w = winners["AVERAGE_ALL"][ds]
                row.append(format_cell(dat["AUROC"], w["AUROC"]["all"], w["AUROC"]["sp"], is_sp))
                row.append(format_cell(dat["AUPRC"], w["AUPRC"]["all"], w["AUPRC"]["sp"], is_sp))
            else:
                row.extend(["-", "-"])
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "detection_metrics_table.tex", "w") as f:
        f.write("\n".join(lines))


def write_aurc_table(
    df_aurc: pd.DataFrame,
    figures_dir: Path,
    model_order: List[str],
    dataset_order: List[str],
) -> None:
    df = df_aurc[df_aurc["Method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        return

    datasets = [d for d in dataset_order if d in df["DatasetKey"].unique()]
    models = [m for m in model_order if m in df["ModelKey"].unique()]
    winners = {}
    for model in models:
        for ds in datasets:
            sd = df[(df["ModelKey"] == model) & (df["DatasetKey"] == ds)]
            winners[(model, ds)] = sd["AURC"].min() if not sd.empty else 1e9

    lines = []
    col_spec = "l" + "c" * len(datasets)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = [r"\textbf{Method}"] + [rf"\textbf{{{latex_escape(pretty_dataset(ds))}}}" for ds in datasets]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        lines.append(rf"\multicolumn{{{1 + len(datasets)}}}{{l}}{{\textbf{{{latex_escape(pretty_model(model))}}}}} \\")
        for method in METHOD_ORDER:
            row = [latex_escape(method)]
            for ds in datasets:
                v = df[(df["ModelKey"] == model) & (df["DatasetKey"] == ds) & (df["Method"] == method)]["AURC"]
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


def write_naurcc_table(
    df_aurc: pd.DataFrame,
    figures_dir: Path,
    model_order: List[str],
    dataset_order: List[str],
) -> None:

    """
    Writes nAURCC table with:
    1. Values per (Model, Dataset)
    2. Row Average (Per Model, across Datasets)
    3. Bottom Block: Column Averages (Across Models) & Grand Mean
    
    Formatting:
    - Bold: Best Overall (Lowest nAURCC).
    - Italic: Best Single-Pass (Lowest nAURCC among probes/combiners).
    """
    df = df_aurc[df_aurc["Method"].isin(METHOD_ORDER)].copy()
    if df.empty: return

    # Use raw nAURCC values (0 to 1 scale)
    value_col = "nAURCC"

    datasets = [d for d in dataset_order if d in df["DatasetKey"].unique()]
    models = [m for m in model_order if m in df["ModelKey"].unique()]
    methods = [m for m in METHOD_ORDER if m in df["Method"].unique()]

    # Define Single-Pass methods (Everything except Semantic Entropy)
    single_pass_methods = [
        "Accuracy Probe", 
        "SE Probe", 
        "Combined (LR)", 
        "Combined (MLP)"
    ]

    # --- Pre-calculate Statistics ---
    # stats[context][method][dataset_key] = value
    stats = {}
    
    # 1. Individual entries + Row Means
    for m in models:
        stats[m] = {}
        for meth in methods:
            stats[m][meth] = {}
            row_vals = []
            for d in datasets:
                val = df[(df["ModelKey"] == m) & (df["DatasetKey"] == d) & (df["Method"] == meth)][value_col]
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
            vals = df[(df["DatasetKey"] == d) & (df["Method"] == meth)][value_col]
            mean_val = vals.mean() if not vals.empty else float("nan")
            stats["AVERAGE_ALL"][meth][d] = mean_val
            if not vals.empty: all_models_vals.extend(vals.values)
        
        # Grand Mean (Average of all nAURCC values for this method)
        stats["AVERAGE_ALL"][meth]["Avg"] = np.mean(all_models_vals) if all_models_vals else float("nan")

    # 3. Identify Winners (Lowest nAURCC)
    # winners[context_key][dataset_key] = {"all": min_val, "sp": min_val_sp}
    winners = {}
    all_model_keys = models + ["AVERAGE_ALL"]
    all_ds_keys = datasets + ["Avg"]
    
    for m_key in all_model_keys:
        for d_key in all_ds_keys:
            best_all = 1e9
            best_sp = 1e9
            
            for meth in methods:
                val = stats[m_key][meth].get(d_key, float("nan"))
                if not np.isnan(val):
                    # Check Overall
                    if val < best_all:
                        best_all = val
                    
                    # Check Single Pass
                    if meth in single_pass_methods:
                        if val < best_sp:
                            best_sp = val
            
            # Handle empty cases
            if best_all == 1e9: best_all = None
            if best_sp == 1e9: best_sp = None
            
            winners[(m_key, d_key)] = {"all": best_all, "sp": best_sp}

    # --- Write LaTeX ---
    lines = []
    col_spec = "l" + "c" * len(datasets) + "c"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    
    # Header
    header = r"\textbf{Method} & " + " & ".join([rf"\textbf{{{latex_escape(pretty_dataset(ds))}}}" for ds in datasets]) + r" & \textbf{Mean} \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Helper for formatting
    def fmt_cell(val, best_all, best_sp, is_sp):
        if np.isnan(val): return "-"
        s = f"{val:.3f}"
        
        is_best_overall = (best_all is not None and val <= best_all + 1e-9)
        is_best_sp = (is_sp and best_sp is not None and val <= best_sp + 1e-9)
        
        if is_best_overall and is_best_sp:
            return rf"\textit{{\textbf{{{s}}}}}"
        elif is_best_overall:
            return rf"\textbf{{{s}}}"
        elif is_best_sp:
            return rf"\textit{{{s}}}"
        return s

    # Body: Per Model
    for model in models:
        lines.append(rf"\multicolumn{{{len(datasets) + 2}}}{{l}}{{\textbf{{{latex_escape(pretty_model(model))}}}}} \\")
        for meth in methods:
            row = [latex_escape(pretty_method(meth))]
            is_sp = (meth in single_pass_methods)
            
            for d_key in all_ds_keys:
                val = stats[model][meth].get(d_key, float("nan"))
                w = winners[(model, d_key)]
                row.append(fmt_cell(val, w["all"], w["sp"], is_sp))
                
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    # Bottom: Overall Average
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{len(datasets) + 2}}}{{l}}{{\textbf{{Overall Average (Across Models)}}}} \\")
    for meth in methods:
        row = [latex_escape(pretty_method(meth))]
        is_sp = (meth in single_pass_methods)
        
        for d_key in all_ds_keys:
            val = stats["AVERAGE_ALL"][meth].get(d_key, float("nan"))
            w = winners[("AVERAGE_ALL", d_key)]
            row.append(fmt_cell(val, w["all"], w["sp"], is_sp))
            
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "naurcc_table.tex", "w") as f:
        f.write("\n".join(lines))
        

def write_correlation_table(
    df_corr: pd.DataFrame,
    df_det: pd.DataFrame,
    figures_dir: Path,
    model_order: List[str],
    dataset_order: List[str],
) -> None:
    """
    Writes Table 1: Model | Correlation | AUROC (Acc Full) | AUROC (SE Full) | AUROC (Acc Conf) | AUROC (SE Conf)
    """
    if df_corr.empty or df_det.empty: 
        return

    # Average correlation across datasets per model for a cleaner summary, 
    # OR list (Model, Dataset) rows. The paper table seemed to be per model (averaged or representative).
    # Here we will list per Model (averaging across datasets) to match the abstract's style of "Model Performance".
    # Alternatively, if you want every pair, change logic below. 
    # Let's do One row per Model (Averaged across datasets) to keep it compact like the draft Table 1.
    
    models = [m for m in model_order if m in df_corr["ModelKey"].unique()]
    
    lines = []
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{2}{c}{\textbf{AUROC (All Samples)}} & \multicolumn{2}{c}{\textbf{AUROC (Confident Subset)}} \\")
    lines.append(r"\textbf{Model} & \textbf{Corr} & \textbf{Acc Probe} & \textbf{Sem Entropy} & \textbf{Acc Probe} & \textbf{Sem Entropy} \\")
    lines.append(r"\midrule")

    for model in models:
        # 1. Get average correlation across datasets (optionally restrict to dataset_order)
        corr_sub = df_corr[df_corr["ModelKey"] == model]
        if "DatasetKey" in corr_sub.columns:
            corr_sub = corr_sub[corr_sub["DatasetKey"].isin(dataset_order)]
        mean_corr = corr_sub["Correlation"].mean()
        
        # 2. Get AUROCs (averaged across datasets)
        # Filter for this model
        sub = df_det[df_det["ModelKey"] == model]
        if "DatasetKey" in sub.columns:
            sub = sub[sub["DatasetKey"].isin(dataset_order)]
        
        def get_auroc(method, subset):
            rows = sub[(sub["Method"] == method) & (sub["Subset"] == subset)]
            if rows.empty: return np.nan
            return rows["AUROC"].mean()

        ap_full = get_auroc("Accuracy Probe", "Full")
        se_full = get_auroc("Semantic Entropy", "Full")
        ap_conf = get_auroc("Accuracy Probe", "Confident")
        se_conf = get_auroc("Semantic Entropy", "Confident")
        
        # Formatting
        def fmt(val, is_bold=False):
            if np.isnan(val): return "-"
            s = f"{val:.2f}"
            return rf"\textbf{{{s}}}" if is_bold else s

        # Bold logic: Max in pair
        bold_full_ap = (ap_full > se_full)
        bold_full_se = (se_full > ap_full)
        bold_conf_ap = (ap_conf > se_conf)
        bold_conf_se = (se_conf > ap_conf)

        row = [
            latex_escape(pretty_model(model)),
            f"{mean_corr:.2f}" if not np.isnan(mean_corr) else "-",
            fmt(ap_full, bold_full_ap),
            fmt(se_full, bold_full_se),
            fmt(ap_conf, bold_conf_ap),
            fmt(se_conf, bold_conf_se)
        ]
        
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(figures_dir / "correlation_summary_table.tex", "w") as f:
        f.write("\n".join(lines))


def write_calibration_error_table(
    df_cal: pd.DataFrame,
    figures_dir: Path,
    model_order: List[str],
    dataset_order: List[str],
    included_datasets: Optional[List[str]] = None,
) -> None:
    """
    Writes TCE / EER table.
    - Alpha Range: [0.05, 0.30].
    - Formatting: 
        - Bold: Best score across ALL methods.
        - Italic: Best score across SINGLE-PASS methods (Acc Probe, SE Probe, Combined).
    - Includes 'Overall Average' row at the bottom.
    """
    # 1. Filter alpha range [0.05, 0.30]
    df = df_cal[
        (df_cal["Target"] >= 0.049) & 
        (df_cal["Target"] <= 0.301)
    ].copy()
    
    if df.empty: return

    available_datasets = list(df["DatasetKey"].unique())
    if included_datasets is not None:
        datasets = [d for d in dataset_order if d in available_datasets and d in included_datasets]
        if not datasets: datasets = available_datasets
    else:
        datasets = [d for d in dataset_order if d in available_datasets]

    models = [m for m in model_order if m in df["ModelKey"].unique()]
    
    # Define Single-Pass methods (Everything except the expensive sampling baseline)
    single_pass_methods = [
        "Accuracy Probe", 
        "SE Probe", 
        "Combined (LR)", 
        "Combined (MLP)"
    ]
    
    # Structure: stats[context][method] = {emp_tce, emp_eer, ...}
    stats = {}
    
    # 2. Calculate Stats Per Model
    for m in models:
        stats[m] = {}
        for meth in METHOD_ORDER:
            mace_emp_vals = []
            eer_emp_vals = []
            mace_ucb_vals = []
            eer_ucb_vals = []
            
            for d in datasets:
                subset = df[(df["ModelKey"] == m) & (df["DatasetKey"] == d) & (df["Method"] == meth)]
                
                for mode in ["Empirical", "Conservative"]:
                    sub_mode = subset[subset["CalMode"] == mode]
                    if not sub_mode.empty:
                        diffs = sub_mode["Realized"] - sub_mode["Target"]
                        mace = np.mean(np.abs(diffs))
                        eer = np.mean(np.maximum(0, diffs))
                        
                        if mode == "Empirical":
                            mace_emp_vals.append(mace)
                            eer_emp_vals.append(eer)
                        else:
                            mace_ucb_vals.append(mace)
                            eer_ucb_vals.append(eer)
            
            if mace_emp_vals or mace_ucb_vals:
                res = {}
                res["emp_tce"] = np.mean(mace_emp_vals) if mace_emp_vals else np.nan
                res["emp_eer"] = np.mean(eer_emp_vals) if eer_emp_vals else np.nan
                res["ucb_tce"] = np.mean(mace_ucb_vals) if mace_ucb_vals else np.nan
                res["ucb_eer"] = np.mean(eer_ucb_vals) if eer_ucb_vals else np.nan
                stats[m][meth] = res
            else:
                stats[m][meth] = None

    # 3. Calculate Overall Average (Across Models)
    stats["AVERAGE_ALL"] = {}
    for meth in METHOD_ORDER:
        agg = {k: [] for k in ["emp_tce", "emp_eer", "ucb_tce", "ucb_eer"]}
        for m in models:
            s = stats[m].get(meth)
            if s:
                for k in agg:
                    if not np.isnan(s[k]): agg[k].append(s[k])
        
        if any(agg.values()):
            res = {}
            for k in agg:
                res[k] = np.mean(agg[k]) if agg[k] else np.nan
            stats["AVERAGE_ALL"][meth] = res
        else:
            stats["AVERAGE_ALL"][meth] = None

    # --- Write LaTeX ---
    lines = []
    lines.append(r"\begin{tabular}{l|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c|}{\textbf{Target Alignment}} & \multicolumn{2}{c}{\textbf{Target Safety}} \\")
    lines.append(r"\textbf{Method} & \textbf{TCE} $\downarrow$ & \textbf{EER} $\downarrow$ & \textbf{TCE} $\downarrow$ & \textbf{EER} $\downarrow$ \\")
    lines.append(r"\midrule")

    # Formatting Helper
    def fmt_cell(val, best_all, best_sp, is_sp):
        if np.isnan(val): return "-"
        s = f"{val:.3f}"
        
        is_best_overall = (best_all is not None and val <= best_all + 1e-9)
        is_best_sp = (is_sp and best_sp is not None and val <= best_sp + 1e-9)
        
        if is_best_overall and is_best_sp:
            return rf"\textit{{\textbf{{{s}}}}}"
        elif is_best_overall:
            return rf"\textbf{{{s}}}"
        elif is_best_sp:
            return rf"\textit{{{s}}}"
        return s

    contexts = models + ["AVERAGE_ALL"]

    for i, ctx in enumerate(contexts):
        if ctx == "AVERAGE_ALL":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{5}{l}{\textbf{Overall Average (Across Models)}} \\")
        else:
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{latex_escape(pretty_model(ctx))}}}}} \\")
        
        # 1. Find Winners for this context
        # We track min for "All" and min for "Single Pass" (sp)
        best_vals = {
            "emp_tce": {"all": 1e9, "sp": 1e9},
            "emp_eer": {"all": 1e9, "sp": 1e9},
            "ucb_tce": {"all": 1e9, "sp": 1e9},
            "ucb_eer": {"all": 1e9, "sp": 1e9},
        }
        
        has_data = False
        
        for meth, data in stats[ctx].items():
            if data is None: continue
            has_data = True
            is_sp = (meth in single_pass_methods)
            
            for metric in best_vals:
                if not np.isnan(data[metric]):
                    # Update Global Min
                    best_vals[metric]["all"] = min(best_vals[metric]["all"], data[metric])
                    # Update Single-Pass Min
                    if is_sp:
                        best_vals[metric]["sp"] = min(best_vals[metric]["sp"], data[metric])

        # Reset sentinels
        for metric in best_vals:
            if best_vals[metric]["all"] == 1e9: best_vals[metric]["all"] = None
            if best_vals[metric]["sp"] == 1e9: best_vals[metric]["sp"] = None

        if not has_data:
            lines.append(r"\addlinespace")
            continue

        # 2. Write Rows
        for meth in METHOD_ORDER:
            data = stats[ctx].get(meth, None)
            if data is None: continue 
            
            is_sp = (meth in single_pass_methods)
            
            row = [latex_escape(pretty_method(meth))]
            metrics = ["emp_tce", "emp_eer", "ucb_tce", "ucb_eer"]
            
            for m_key in metrics:
                val = data[m_key]
                best_all = best_vals[m_key]["all"]
                best_sp = best_vals[m_key]["sp"]
                row.append(fmt_cell(val, best_all, best_sp, is_sp))
                
            lines.append(" & ".join(row) + r" \\")
        
        if ctx != "AVERAGE_ALL":
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
    parser.add_argument("--figures-dirname", type=str, default="analysis_output_local")
    # parser.add_argument("--embedding-key", type=str, default="emb_last_tok_before_gen")
    parser.add_argument("--n-cap", type=int, default=2000)
    parser.add_argument("--hall-acc-threshold", type=float, default=0.99)

    parser.add_argument("--use-ucb", action="store_true")
    parser.add_argument("--delta", type=float, default=0.1)

    parser.add_argument("--decision-plot-method", type=str, default="MLP", choices=["LR", "MLP"])
    parser.add_argument("--decision-plot-alpha", type=float, default=0.20)
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

    # ---- Global style knobs ----
    parser.add_argument("--font-scale", type=float, default=1.4,
                        help="Seaborn font_scale multiplier (quick global resize).")
    parser.add_argument("--base-fontsize", type=int, default=None,
                        help="If set, overrides Matplotlib base font size (strong global control).")
    parser.add_argument("--plot-context", type=str, default="paper",
                        choices=["paper", "talk", "poster"],
                        help="Seaborn context preset. 'poster' is helpful for A0.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for raster outputs (png). Increase for posters.")



    args = parser.parse_args()

    # Apply plotting style AFTER parsing args (so CLI controls it)
    configure_plotting(args.font_scale, args.base_fontsize, args.plot_context)
    profile = "poster_gemini" if args.plot_context == "poster" else "paper_times"
    apply_tex_font_profile(profile)

    # Desired ordering 
    model_order = [Path(m).name for m in DEFAULT_TARGET_MODELS]
    dataset_order = list(DEFAULT_TARGET_DATASETS)

    # map safe->full id when available, but allow safe names too
    safe_to_full = {Path(m).name: m for m in DEFAULT_TARGET_MODELS}

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

    for safe_model in model_order:
        model_name = safe_to_full.get(safe_model, safe_model)
        curve_data[safe_model] = {}

        for dataset_name in dataset_order:
            run_dir = find_run_directory(runs_root, model_name, dataset_name)
            if not run_dir:
                target_folder = f"{safe_model}__{dataset_name}"
                print(f"[Miss] Could not find folder '{target_folder}' in {runs_root}")
                continue

            print(f"[Found] Processing {run_dir.name}...")

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
                "ModelKey": safe_model,
                "Model": pretty_model(safe_model),
                "DatasetKey": dataset_name,
                "Dataset": pretty_dataset(dataset_name),
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

            # --- Get Probabilities ---
            df["p_correct"] = acc_model.predict_proba(X[acc_layer])[:, 1].astype(np.float32)
            df["p_high_entropy"] = se_model.predict_proba(X[se_layer])[:, 1].astype(np.float32)

            # --- Get Logits (for plotting) ---
            # Try decision_function (cleaner), else inverse-sigmoid the probabilities
            if hasattr(acc_model, "decision_function"):
                df["logit_correct"] = acc_model.decision_function(X[acc_layer]).astype(np.float32)
            else:
                from scipy.special import logit
                # Clip to avoid inf
                p_clipped = df["p_correct"].clip(1e-6, 1 - 1e-6)
                df["logit_correct"] = logit(p_clipped)

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
                # We correlate SE (Risk) with 1-P (Risk) OR -Logit (Risk)
                # This ensures a positive correlation coefficient.
                corr, _ = spearmanr(df_test["score_Acc"], df_test["score_SE"])

    
                corr_results.append({
                    "ModelKey": safe_model,
                    "Model": pretty_model(safe_model),
                    "DatasetKey": dataset_name,
                    "Dataset": pretty_dataset(dataset_name),
                    "Correlation": corr
                })
                
                # Save scatter plot for the first dataset or specific ones
                if dataset_name == "trivia_qa":  # or always
                    plot_correlation_scatter(
                        df_test, pretty_model(safe_model), pretty_dataset(dataset_name), 
                        figures_dir / f"scatter_{file_safe(safe_model)}_{file_safe(dataset_name)}.png",
                        dpi=args.dpi,
                    )

            # confident subset defined on test only
            q50 = df_test["se_raw"].quantile(0.50)
            df_conf = df_test[df_test["se_raw"] <= q50].copy()

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

                det_results.append({
                    "ModelKey": safe_model,
                    "Model": pretty_model(safe_model),
                    "DatasetKey": dataset_name,
                    "Dataset": pretty_dataset(dataset_name),
                    "Method": m_name, "Subset": "Full",
                    "AUROC": float(auc_full), "AUPRC": float(auprc_full)
                })
                det_results.append({
                    "ModelKey": safe_model,
                    "Model": pretty_model(safe_model),
                    "DatasetKey": dataset_name,
                    "Dataset": pretty_dataset(dataset_name),
                    "Method": m_name, "Subset": "Confident",
                    "AUROC": float(auc_conf)
                })

                # risk-coverage curve (test)
                covs, risks = get_risk_coverage_curve(df_test, col)
                curve_data[safe_model][dataset_name][m_name] = (covs, risks)

                base_risk = base_risks[(safe_model, dataset_name)]

                # AURC raw
                aurc_val = float(np.trapezoid(risks, covs)) if len(covs) > 1 else float("nan")

                # Compute Normalized AURCC
                naurc_val = compute_normalized_aurcc(covs, risks, base_risk)

                aurc_results.append({
                    "ModelKey": safe_model,
                    "Model": pretty_model(safe_model),
                    "DatasetKey": dataset_name,
                    "Dataset": pretty_dataset(dataset_name),
                    "Method": m_name,
                    "AURC": aurc_val,
                    "nAURCC": naurc_val,
                    "Correlation": corr
                })

                # calibration: threshold on cal, evaluate on test

                # Run Calibration TWICE
                modes = [("Empirical", False), ("Conservative", True)]
                for mode_name, use_ucb_flag in modes:
                    for alpha in all_alphas:
                        r, c = eval_calibration(df, col, float(alpha), args.delta, use_ucb=use_ucb_flag)
                        cal_results.append({
                            "ModelKey": safe_model,
                            "Model": pretty_model(safe_model),
                            "DatasetKey": dataset_name,
                            "Dataset": pretty_dataset(dataset_name),
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
                    out_path = figures_dir / f"boundary_{file_safe(safe_model)}_{file_safe(dataset_name)}_{dm}.png"
                    plot_decision_boundary(
                        df=df,
                        combiner_model=comb,
                        score_col=score_col,
                        alpha=float(args.decision_plot_alpha),
                        delta=args.delta,
                        use_ucb=args.use_ucb,
                        out_path=out_path,
                        title=f"{pretty_model(safe_model)} / {pretty_dataset(dataset_name)} — Decision Boundary ({dm})",
                        dpi=args.dpi,
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
        plot_detection_bars(df_det, figures_dir, model_order, dataset_order, dpi=args.dpi)
    if layer_results:
        plot_layer_sensitivity(layer_results, figures_dir, dpi=args.dpi)
    if curve_data:
        plot_risk_coverage(curve_data, figures_dir, base_risks, dpi=args.dpi)
    if not df_cal.empty:
        plot_calibration_curves(df_cal, figures_dir, model_order, dataset_order, dpi=args.dpi)

    # Tables
    print("Generating LaTeX tables...")
    if not df_det.empty:
        write_detection_table(df_det, base_risks, figures_dir, model_order=model_order, dataset_order=dataset_order)
    if not df_aurc.empty:
        write_aurc_table(df_aurc, figures_dir, model_order=model_order, dataset_order=dataset_order)
        write_naurcc_table(df_aurc, figures_dir, model_order=model_order, dataset_order=dataset_order)

    if not df_cal.empty:
        write_calibration_error_table(
            df_cal, figures_dir,
            model_order=model_order,
            dataset_order=dataset_order,
            included_datasets=["trivia_qa"],
        )
    if not df_corr.empty and not df_det.empty:       
        write_correlation_table(df_corr, df_det, figures_dir, model_order=model_order, dataset_order=dataset_order)
        

    print(f"Done! Artifacts saved to: {figures_dir}")


if __name__ == "__main__":
    main()
