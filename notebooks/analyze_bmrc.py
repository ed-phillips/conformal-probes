import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 0. Configuration
# ==========================================

TARGET_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-4b-it",
    "google/gemma-7b-it",
    "mistralai/Ministral-8B-Instruct-2410" 
]

TARGET_DATASETS = [
    "trivia_qa",
    # "bioasq",
    # "medical_o1",
]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
COLORS = {
    'Semantic Entropy': '#7f8c8d', 
    'Accuracy Probe': '#3498db',   
    'SE Probe': '#e67e22',         
    'Combined (LR)': '#2ecc71',    
    'Dual-Probe (2D)': '#9b59b6'   
}

# ==========================================
# 1. Path Finding & Data Loading
# ==========================================

def resolve_paths(args_runs_root):
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "notebooks":
        repo_root = script_path.parent.parent
    else:
        repo_root = Path.cwd()

    runs_path = Path(args_runs_root)
    if not runs_path.is_absolute():
        runs_path = repo_root / runs_path
        
    return repo_root, runs_path

def find_run_directory(root_path, model_name, dataset_name):
    safe_model_name = Path(model_name).name
    target_folder_name = f"{safe_model_name}__{dataset_name}"
    candidates = []
    if root_path.exists():
        flat_path = root_path / target_folder_name
        if flat_path.exists(): candidates.append(flat_path)
        for p in root_path.iterdir():
            if p.is_dir():
                nested_path = p / target_folder_name
                if nested_path.exists(): candidates.append(nested_path)
    if not candidates: return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def load_pickles(run_dir):
    files_dir = run_dir / "files"
    if not files_dir.exists():
        try:
            candidates = list(run_dir.rglob("files"))
            if candidates: files_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
            else: files_dir = run_dir
        except: return None

    p1 = files_dir / "validation_generations.pkl"
    p2 = files_dir / "uncertainty_measures.pkl"
    
    if not (p1.exists() and p2.exists()): return None
    with open(p1, "rb") as f: gens = pickle.load(f)
    with open(p2, "rb") as f: unc = pickle.load(f)
    return gens, unc

# ==========================================
# 2. Data Alignment (Fixed)
# ==========================================

def extract_features_aligned(gens, unc, n_cap=2000):
    # Use insertion order (values()) to match unc list
    gen_values = list(gens.values())
    
    accuracies = np.array([g["most_likely_answer"]["accuracy"] for g in gen_values])
    y_hallucination = (accuracies < 1.0).astype(int) 
    
    ent_keys = ["cluster_assignment_entropy", "semantic_entropy_sum_normalized"]
    raw_entropy = None
    for k in ent_keys:
        if k in unc["uncertainty_measures"]:
            raw_entropy = np.array(unc["uncertainty_measures"][k])
            break
    if raw_entropy is None: raw_entropy = np.zeros_like(accuracies)

    tbg_raw = [g["most_likely_answer"]["emb_last_tok_before_gen"] for g in gen_values]
    X_tensor = torch.stack(tbg_raw).squeeze(-2).transpose(0, 1).numpy()
    
    n = min(len(y_hallucination), X_tensor.shape[1], n_cap)
    return X_tensor[:, :n, :], y_hallucination[:n], raw_entropy[:n]

# ==========================================
# 3. Training & Statistics Collection
# ==========================================

def process_model_dataset(X, y, entropy):
    """
    Retrains probes, collects layer stats, and returns scores df + layer info.
    """
    n_total = len(y)
    indices = np.arange(n_total)
    
    # Splits
    idx_train, idx_temp = train_test_split(indices, test_size=0.3, random_state=42)
    idx_cal, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    
    # Binarize entropy for SE probe target
    train_ent = entropy[idx_train]
    split_val = np.median(train_ent)
    y_ent_bin = (entropy > split_val).astype(int)

    n_layers = X.shape[0]
    
    # Stats storage
    layer_stats = {'acc_aucs': [], 'se_aucs': []}
    
    # 1. Sweep Accuracy Probe
    best_acc_auc = 0; best_acc_model = None; best_acc_layer = 0
    for l in range(n_layers):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, C=0.1))
        pipe.fit(X[l][idx_train], y[idx_train])
        preds = pipe.predict_proba(X[l][idx_cal])[:, 1]
        auc = roc_auc_score(y[idx_cal], preds)
        layer_stats['acc_aucs'].append(auc)
        
        if auc > best_acc_auc:
            best_acc_auc = auc; best_acc_model = pipe; best_acc_layer = l

    # 2. Sweep SE Probe
    best_se_auc = 0; best_se_model = None; best_se_layer = 0
    for l in range(n_layers):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, C=0.1))
        pipe.fit(X[l][idx_train], y_ent_bin[idx_train])
        preds = pipe.predict_proba(X[l][idx_cal])[:, 1]
        auc = roc_auc_score(y_ent_bin[idx_cal], preds) # Check alignment with entropy
        layer_stats['se_aucs'].append(auc)
        
        if auc > best_se_auc:
            best_se_auc = auc; best_se_model = pipe; best_se_layer = l

    # 3. Generate Scores
    df = pd.DataFrame()
    splits = np.zeros(n_total, dtype=object)
    splits[idx_train] = 'train'; splits[idx_cal] = 'calibration'; splits[idx_test] = 'test'
    df['split'] = splits
    df['y_true'] = y
    df['se_raw'] = entropy
    
    df['s_acc'] = best_acc_model.predict_proba(X[best_acc_layer])[:, 1]
    df['s_se_probe'] = best_se_model.predict_proba(X[best_se_layer])[:, 1]
    
    # 4. Combiner (LR on Cal)
    cal_mask = (splits == 'calibration')
    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(df.loc[cal_mask, ['s_acc', 's_se_probe']], y[cal_mask])
    df['s_combined'] = lr.predict_proba(df[['s_acc', 's_se_probe']])[:, 1]
    
    return df, layer_stats

# ==========================================
# 4. Metric Calculators
# ==========================================

def get_curve_data(df, method_col, is_2d=False):
    """
    Computes Risk vs Coverage curve.
    For 1D: Sweep threshold on score.
    For 2D: Sweep alpha on conformal procedure.
    """
    cal_df = df[df['split'] == 'calibration']
    test_df = df[df['split'] == 'test']
    
    risks, coverages = [], []
    
    if not is_2d:
        # Heuristic Curve: Sort by score
        scores = test_df[method_col].values
        labels = test_df['y_true'].values
        
        # Sort ascending (low score = safe? No, high score = hallucination usually)
        # s_acc: P(Hallucination), s_se_probe: P(HighEnt), se_raw: Entropy
        # So low score = Safe. We accept if score <= t.
        
        sort_idx = np.argsort(scores) # Low to High
        sorted_labels = labels[sort_idx]
        
        # Cumulative
        n = len(labels)
        cum_hallucinations = np.cumsum(sorted_labels)
        n_accepted = np.arange(1, n + 1)
        
        calculated_risks = cum_hallucinations / n_accepted
        calculated_coverages = n_accepted / n
        
        # Downsample for plotting
        indices = np.linspace(0, n-1, 100).astype(int)
        risks = calculated_risks[indices]
        coverages = calculated_coverages[indices]
        
    else:
        # 2D Conformal Sweep
        # We sweep target alpha to trace out the Pareto frontier
        alphas = np.linspace(0.001, 0.6, 30)
        for alpha in alphas:
            r, c = conformal_2d_search(df, alpha)
            risks.append(r)
            coverages.append(c)
            
    return list(coverages), list(risks)

def conformal_2d_search(df, alpha, steps=30):
    cal_df = df[df['split'] == 'calibration']; test_df = df[df['split'] == 'test']
    t_grid = np.linspace(0, 1, steps)
    best_t1, best_t2, max_cov = -1, -1, -1
    
    # Grid search for region s_acc <= t1 & s_se <= t2
    # Optimization: vectorize?
    # Simple loop is fine for small Cal sets (~200 samples)
    
    for t1 in t_grid:
        for t2 in t_grid:
            mask = (cal_df['s_acc'] <= t1) & (cal_df['s_se_probe'] <= t2)
            if mask.sum() == 0: continue
            risk = cal_df.loc[mask, 'y_true'].mean()
            if risk <= alpha:
                cov = mask.sum() / len(cal_df)
                if cov > max_cov: max_cov = cov; best_t1, best_t2 = t1, t2
                    
    mask_test = (test_df['s_acc'] <= best_t1) & (test_df['s_se_probe'] <= best_t2)
    return (test_df.loc[mask_test, 'y_true'].mean() if mask_test.sum() > 0 else 0.0, mask_test.sum()/len(test_df))

def conformal_1d(df, col, alpha):
    cal_df = df[df['split'] == 'calibration']; test_df = df[df['split'] == 'test']
    cal_scores = np.sort(cal_df[col].values)
    cal_labels = cal_df['y_true'].values[np.argsort(cal_df[col].values)]
    cum_risk = np.cumsum(cal_labels) / np.arange(1, len(cal_labels) + 1)
    
    valid = np.where(cum_risk <= alpha)[0]
    t_star = cal_scores[valid[-1]] if len(valid) > 0 else -np.inf
    
    if t_star == -np.inf: return 0.0, 0.0
    accepted = test_df[test_df[col] <= t_star]
    return (accepted['y_true'].mean() if len(accepted) > 0 else 0.0, len(accepted)/len(test_df))

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, required=True)
    args = parser.parse_args()

    repo_root, runs_root = resolve_paths(args.runs_root)
    figures_dir = repo_root / "paper_figures_fixed"
    figures_dir.mkdir(exist_ok=True)
    
    # Storage
    det_results = []
    cal_results = []
    layer_results = []
    curve_data = {} # nested dict: model -> dataset -> method -> (cov, risk)

    print(f"Scanning {runs_root}...")
    
    for model_name in TARGET_MODELS:
        safe_name = Path(model_name).name
        curve_data[safe_name] = {}
        
        for dataset_name in TARGET_DATASETS:
            curve_data[safe_name][dataset_name] = {}
            
            run_dir = find_run_directory(runs_root, model_name, dataset_name)
            if not run_dir: continue
            
            pickles = load_pickles(run_dir)
            if not pickles: continue
            
            print(f"Processing {safe_name} / {dataset_name}...")
            
            X, y, ent = extract_features_aligned(*pickles)
            try:
                df, l_stats = process_model_dataset(X, y, ent)
            except Exception as e:
                print(f"  Failed: {e}")
                continue
                
            # 1. Layer Stats
            layer_results.append({
                'Model': safe_name, 'Dataset': dataset_name, 
                'Acc_AUCs': l_stats['acc_aucs'], 'SE_AUCs': l_stats['se_aucs']
            })
            
            # 2. Detection Metrics
            df_test = df[df['split'] == 'test']
            thresh_30 = df_test['se_raw'].quantile(0.30)
            df_conf = df_test[df_test['se_raw'] <= thresh_30]
            
            methods = {
                'Semantic Entropy': 'se_raw', 
                'Accuracy Probe': 's_acc', 
                'SE Probe': 's_se_probe', 
                'Combined (LR)': 's_combined'
            }
            
            for m_name, col in methods.items():
                # AUROC
                det_results.append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Subset': 'Full Test', 'AUROC': roc_auc_score(df_test['y_true'], df_test[col])})
                try: val = roc_auc_score(df_conf['y_true'], df_conf[col])
                except: val = 0.5
                det_results.append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Subset': 'Confident Subset', 'AUROC': val})
                
                # Curve Data (1D)
                covs, risks = get_curve_data(df, col, is_2d=False)
                curve_data[safe_name][dataset_name][m_name] = (covs, risks)
                
                # Calibration Data (Target vs Realized)
                for alpha in np.linspace(0.01, 0.4, 20):
                    r, c = conformal_1d(df, col, alpha)
                    cal_results.append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Target': alpha, 'Realized': r})

            # 3. Dual Probe (2D) Special Handling
            # Curve Data (Sweep Alpha)
            covs_2d, risks_2d = get_curve_data(df, None, is_2d=True)
            curve_data[safe_name][dataset_name]['Dual-Probe (2D)'] = (covs_2d, risks_2d)
            
            # Calibration Data
            for alpha in np.linspace(0.01, 0.4, 20):
                r, c = conformal_2d_search(df, alpha)
                cal_results.append({'Model': safe_name, 'Dataset': dataset_name, 'Method': 'Dual-Probe (2D)', 'Target': alpha, 'Realized': r})

    # ==========================================
    # 6. Plotting
    # ==========================================
    if not det_results:
        print("No results.")
        return

    print(f"\nGenerating plots in {figures_dir}...")
    
    df_det = pd.DataFrame(det_results)
    df_cal = pd.DataFrame(cal_results)
    
    # --- Plot 1: Detection Bar Charts (Per Model) ---
    for model in df_det['Model'].unique():
        for sub in ['Full Test', 'Confident Subset']:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_det[(df_det['Model']==model) & (df_det['Subset']==sub)], x='Dataset', y='AUROC', hue='Method', palette=COLORS, edgecolor='black', errorbar=None)
            plt.title(f"{model} - Detection ({sub})")
            plt.ylim(0.4, 1.0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(figures_dir / f"detection_{model}_{sub.replace(' ', '_')}.png", dpi=300)
            plt.close()

    # --- Plot 2: Risk-Coverage Curves (Per Model/Dataset) ---
    for model, datasets in curve_data.items():
        for ds_name, methods_data in datasets.items():
            plt.figure(figsize=(8, 6))
            for m_name, (covs, risks) in methods_data.items():
                # Sort by coverage for clean line
                pairs = sorted(zip(covs, risks))
                c, r = zip(*pairs)
                plt.plot(c, r, label=m_name, color=COLORS.get(m_name, 'black'), linewidth=2)
            
            plt.xlabel("Coverage (Fraction Answered)")
            plt.ylabel("Risk (Hallucination Rate)")
            plt.title(f"Risk-Coverage: {model} / {ds_name}")
            plt.legend()
            plt.xlim(0, 1.0); plt.ylim(0, 0.6)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_dir / f"risk_coverage_{model}_{ds_name}.png", dpi=300)
            plt.close()

    # --- Plot 3: Calibration (Target vs Realized) ---
    # Aggregate across datasets for cleaner plot per model
    for model in df_cal['Model'].unique():
        plt.figure(figsize=(7, 7))
        # Scatter with trendline? Just scatter for now
        sns.lineplot(data=df_cal[df_cal['Model']==model], x='Target', y='Realized', hue='Method', palette=COLORS, style='Dataset', markers=True, dashes=False)
        plt.plot([0, 0.4], [0, 0.4], 'k--', alpha=0.5, label="Ideal")
        plt.xlabel("Target Risk ($\\alpha$)")
        plt.ylabel("Realized Risk")
        plt.title(f"Calibration: {model}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(figures_dir / f"calibration_{model}.png", dpi=300)
        plt.close()

    # --- Plot 4: Layer Sensitivity (Per Model/Dataset) ---
    # Group by model to avoid clutter
    for item in layer_results:
        plt.figure(figsize=(8, 5))
        x = np.linspace(0, 1, len(item['Acc_AUCs']))
        plt.plot(x, item['Acc_AUCs'], label='Acc Probe', color=COLORS['Accuracy Probe'], linewidth=2)
        plt.plot(x, item['SE_AUCs'], label='SE Probe', color=COLORS['SE Probe'], linestyle='--', linewidth=2)
        plt.xlabel("Layer Depth")
        plt.ylabel("AUROC (Calibration)")
        plt.title(f"Layer Sensitivity: {item['Model']} / {item['Dataset']}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"layers_{item['Model']}_{item['Dataset']}.png", dpi=300)
        plt.close()

    print("Done.")

if __name__ == "__main__":
    main()