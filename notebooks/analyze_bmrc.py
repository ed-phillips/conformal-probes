import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 0. Configuration & Setup
# ==========================================

# Define exactly which models and datasets to process
TARGET_MODELS = [
#   "meta-llama/Llama-3.1-8B-Instruct",
#   "Qwen/Qwen3-8B",
  "google/gemma-7b-it",
  "meta-llama/Llama-3.2-3B-Instruct",
#   "HuggingFaceTB/SmolLM3-3B",
  "Qwen/Qwen3-4B-Instruct-2507",
  "google/gemma-3-4b-it",
  "mistralai/Ministral-8B-Instruct-2410",
]

TARGET_DATASETS = [
    "trivia_qa",
    # "bioasq",
    # "medical_o1",
]

# Plotting Styles
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
# 1. Path Finding
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

    if not candidates: return None, target_folder_name
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0], target_folder_name

def load_pickles(run_dir):
    # Find files dir
    files_dir = run_dir / "files"
    if not files_dir.exists():
        candidates = list(run_dir.rglob("files"))
        if candidates:
            files_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        else:
            files_dir = run_dir 

    p1 = files_dir / "validation_generations.pkl"
    p2 = files_dir / "uncertainty_measures.pkl"
    # We ignore probes.pkl because they are trained on misaligned data
    
    if not (p1.exists() and p2.exists()): return None
    
    with open(p1, "rb") as f: gens = pickle.load(f)
    with open(p2, "rb") as f: unc = pickle.load(f)
    return gens, unc

# ==========================================
# 2. Data Alignment (THE FIX)
# ==========================================

def extract_features_aligned(gens, unc, n_cap=2000):
    """
    CRITICAL FIX: Do NOT sort gens by key. 
    Use the dictionary iteration order to match the list order in 'unc'.
    """
    # 1. Use natural iteration order (matching compute_uncertainty_measures.py)
    gen_values = list(gens.values())
    
    # 2. Extract Labels (Aligned)
    accuracies = np.array([g["most_likely_answer"]["accuracy"] for g in gen_values])
    y_hallucination = (accuracies < 1.0).astype(int) 
    
    # 3. Extract Entropy (Aligned)
    ent_keys = ["cluster_assignment_entropy", "semantic_entropy_sum_normalized"]
    raw_entropy = None
    for k in ent_keys:
        if k in unc["uncertainty_measures"]:
            raw_entropy = np.array(unc["uncertainty_measures"][k])
            break
    if raw_entropy is None: raw_entropy = np.zeros_like(accuracies)

    # 4. Extract Embeddings (Aligned)
    tbg_raw = [g["most_likely_answer"]["emb_last_tok_before_gen"] for g in gen_values]
    X_tensor = torch.stack(tbg_raw).squeeze(-2).transpose(0, 1).numpy()
    
    # Truncate
    n = min(len(y_hallucination), X_tensor.shape[1], n_cap)
    X_tensor = X_tensor[:, :n, :]
    y_hallucination = y_hallucination[:n]
    raw_entropy = raw_entropy[:n]
    
    return X_tensor, y_hallucination, raw_entropy

# ==========================================
# 3. On-the-fly Training
# ==========================================

def retrain_probes(X, y, entropy):
    """
    Retrain probes here since the pickle ones are invalid.
    Splits: 70% Train, 15% Cal, 15% Test
    """
    n_total = len(y)
    indices = np.arange(n_total)
    
    # Standard split
    idx_train, idx_temp = train_test_split(indices, test_size=0.3, random_state=42)
    idx_cal, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    
    # Binarize entropy for SE Probe (Median split on Train)
    train_ent = entropy[idx_train]
    split_val = np.median(train_ent)
    y_ent_bin = (entropy > split_val).astype(int) # 1 = High Entropy (Bad)

    # Sweep Layers on Calibration
    n_layers = X.shape[0]
    best_acc_auc = 0; best_acc_model = None
    best_se_auc = 0; best_se_model = None
    
    # We'll just train 'best' layer found via simple scan
    # Optimization: Only train Linear Probe on Cal set to find best layer, 
    # then retrain on Train set for that layer.
    
    # 1. Find Best Layer for Accuracy Probe
    for l in range(n_layers):
        clf = LogisticRegression(max_iter=200, C=0.1)
        # Quick check on Cal (train on Train)
        # Scale? Yes, crucial.
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(X[l][idx_train], y[idx_train])
        
        preds = pipe.predict_proba(X[l][idx_cal])[:, 1]
        auc = roc_auc_score(y[idx_cal], preds) # Predicting Hallucination directly (1=Hallucination)
        # Note: If accuracy labels are 1=Correct, y needs flipping or we predict class 0.
        # Here y is y_hallucination (1=Wrong), so predict class 1.
        
        if auc > best_acc_auc:
            best_acc_auc = auc
            best_acc_model = pipe
            best_acc_layer = l

    # 2. Find Best Layer for SE Probe
    for l in range(n_layers):
        clf = LogisticRegression(max_iter=200, C=0.1)
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(X[l][idx_train], y_ent_bin[idx_train])
        
        preds = pipe.predict_proba(X[l][idx_cal])[:, 1]
        auc = roc_auc_score(y_ent_bin[idx_cal], preds)
        
        if auc > best_se_auc:
            best_se_auc = auc
            best_se_model = pipe
            best_se_layer = l
            
    # Return Dataframe with scores
    df = pd.DataFrame()
    splits = np.zeros(n_total, dtype=object)
    splits[idx_train] = 'train'; splits[idx_cal] = 'calibration'; splits[idx_test] = 'test'
    
    df['split'] = splits
    df['y_true'] = y
    df['se_raw'] = entropy
    
    # Apply Best Models
    df['s_acc'] = best_acc_model.predict_proba(X[best_acc_layer])[:, 1]
    df['s_se_probe'] = best_se_model.predict_proba(X[best_se_layer])[:, 1]
    
    # Combiner (Train on Cal)
    cal_mask = (splits == 'calibration')
    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(df.loc[cal_mask, ['s_acc', 's_se_probe']], y[cal_mask])
    df['s_combined'] = lr.predict_proba(df[['s_acc', 's_se_probe']])[:, 1]
    
    return df

# ==========================================
# 4. Metrics & Conformal (Same as before)
# ==========================================

def conformal_risk_control(df, score_col, alpha):
    cal_df = df[df['split'] == 'calibration']; test_df = df[df['split'] == 'test']
    cal_scores = np.sort(cal_df[score_col].values)
    cal_labels = cal_df['y_true'].values[np.argsort(cal_df[score_col].values)]
    cum_risk = np.cumsum(cal_labels) / np.arange(1, len(cal_labels) + 1)
    valid = np.where(cum_risk <= alpha)[0]
    t_star = cal_scores[valid[-1]] if len(valid) > 0 else -np.inf
    if t_star == -np.inf: return 0.0, 0.0
    accepted = test_df[test_df[score_col] <= t_star]
    return (accepted['y_true'].mean() if len(accepted) > 0 else 0.0, len(accepted)/len(test_df))

def conformal_2d_search(df, alpha, steps=50):
    cal_df = df[df['split'] == 'calibration']; test_df = df[df['split'] == 'test']
    t_grid = np.linspace(0, 1, steps)
    best_t1, best_t2, max_cov = -1, -1, -1
    for t1 in t_grid:
        for t2 in t_grid:
            mask = (cal_df['s_acc'] <= t1) & (cal_df['s_se_probe'] <= t2)
            if mask.sum() == 0: continue
            if cal_df.loc[mask, 'y_true'].mean() <= alpha:
                cov = mask.sum() / len(cal_df)
                if cov > max_cov: max_cov = cov; best_t1, best_t2 = t1, t2
    mask_test = (test_df['s_acc'] <= best_t1) & (test_df['s_se_probe'] <= best_t2)
    return (test_df.loc[mask_test, 'y_true'].mean() if mask_test.sum() > 0 else 0.0, mask_test.sum()/len(test_df))

# ==========================================
# 5. Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, required=True, help="Path to runs")
    args = parser.parse_args()

    repo_root, runs_root = resolve_paths(args.runs_root)
    figures_dir = repo_root / "paper_figures_fixed"
    figures_dir.mkdir(exist_ok=True)
    
    results = {'detection': [], 'conformal': []}
    
    print(f"Scanning {runs_root}...\n")
    
    for model_name in TARGET_MODELS:
        for dataset_name in TARGET_DATASETS:
            
            run_dir, expected_name = find_run_directory(runs_root, model_name, dataset_name)
            
            print(f"[{expected_name}]", end=" ")
            if not run_dir:
                print("Not found.")
                continue
            
            # Load Raw Pickles (No Probes)
            pickles = load_pickles(run_dir)
            if not pickles: 
                print("Pickles missing.")
                continue
            
            print("Loaded.", end=" ")
            
            # Extract ALIGNED data
            X, y, ent = extract_features_aligned(*pickles)
            
            # Retrain Probes (Fast)
            try:
                df = retrain_probes(X, y, ent)
                print("Retrained.")
            except Exception as e:
                print(f"Training Failed: {e}")
                continue
            
            df_test = df[df['split'] == 'test']
            thresh_30 = df_test['se_raw'].quantile(0.30)
            df_conf = df_test[df_test['se_raw'] <= thresh_30]
            
            # Detection
            safe_name = Path(model_name).name
            methods = {'Semantic Entropy': 'se_raw', 'Accuracy Probe': 's_acc', 'SE Probe': 's_se_probe', 'Combined (LR)': 's_combined'}
            for m_name, col in methods.items():
                results['detection'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Subset': 'Full Test', 'AUROC': roc_auc_score(df_test['y_true'], df_test[col])})
                try: val = roc_auc_score(df_conf['y_true'], df_conf[col])
                except: val = 0.5
                results['detection'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Subset': 'Confident Subset', 'AUROC': val})
            
            # Conformal
            for alpha in [0.01, 0.05, 0.10]:
                for m_name, col in methods.items():
                    r, c = conformal_risk_control(df, col, alpha)
                    results['conformal'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Target_Alpha': alpha, 'Realized_Risk': r, 'Coverage': c})
                r2, c2 = conformal_2d_search(df, alpha)
                results['conformal'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': 'Dual-Probe (2D)', 'Target_Alpha': alpha, 'Realized_Risk': r2, 'Coverage': c2})

    # Saving
    if not results['detection']:
        print("No valid results.")
        return

    print(f"\nSaving figures to: {figures_dir}")
    df_det = pd.DataFrame(results['detection'])
    df_conf = pd.DataFrame(results['conformal'])
    df_det.to_csv(figures_dir / "results_detection.csv", index=False)
    df_conf.to_csv(figures_dir / "results_conformal.csv", index=False)

    # Plot
    for model in df_det['Model'].unique():
        for sub in ['Full Test', 'Confident Subset']:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_det[(df_det['Model']==model) & (df_det['Subset']==sub)], x='Dataset', y='AUROC', hue='Method', palette=COLORS, edgecolor='black', errorbar=None)
            plt.title(f"{model} - Detection ({sub})")
            plt.ylim(0.4, 1.0)
            plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(figures_dir / f"detection_{model}_{sub.replace(' ', '_')}.png", dpi=300)
            plt.close()

    if not df_conf.empty:
        sub = df_conf[df_conf['Target_Alpha'] == 0.05]
        for model in sub['Model'].unique():
            plt.figure(figsize=(10, 6))
            sns.barplot(data=sub[sub['Model']==model], x='Dataset', y='Coverage', hue='Method', palette=COLORS, edgecolor='black', errorbar=None)
            plt.title(f"{model} - Coverage @ Risk=0.05")
            plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(figures_dir / f"coverage_{model}_0.05.png", dpi=300)
            plt.close()

    print("Done.")

if __name__ == "__main__":
    main()