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
# 1. Path & Data Handling
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
    """
    Finds the run directory.
    Returns the path to the folder named {Model}__{Dataset}.
    """
    safe_model_name = Path(model_name).name
    target_folder_name = f"{safe_model_name}__{dataset_name}"
    
    candidates = []
    
    if root_path.exists():
        # 1. Check Root (Flat)
        flat_path = root_path / target_folder_name
        if flat_path.exists():
            candidates.append(flat_path)
        
        # 2. Check Job IDs (Nested)
        # Iterate over immediate children (Job IDs)
        for p in root_path.iterdir():
            if p.is_dir() and p.name != target_folder_name:
                nested_path = p / target_folder_name
                if nested_path.exists():
                    candidates.append(nested_path)

    if not candidates:
        return None, target_folder_name
    
    # 3. Sort by modification time to get latest
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0], target_folder_name

def find_wandb_files_explicit(run_dir):
    """
    Explicitly looks for the 'files' directory structure used on HPC.
    Structure: run_dir / <USER> / uncertainty / wandb / latest-run / files
    """
    # 1. Check if 'files' is directly inside (Local runs)
    if (run_dir / "files").exists():
        return run_dir / "files"

    # 2. Check HPC Structure: Look for 'uncertainty/wandb' inside any user folder
    # We iterate immediate children of run_dir to find the user folder
    try:
        for user_dir in run_dir.iterdir():
            if user_dir.is_dir():
                wandb_root = user_dir / "uncertainty" / "wandb"
                if wandb_root.exists():
                    # Check 'latest-run' symlink
                    latest = wandb_root / "latest-run" / "files"
                    if latest.exists():
                        return latest
                    
                    # Check 'offline-run-*' folders
                    offlines = list(wandb_root.glob("offline-run-*"))
                    if offlines:
                        # Sort by name (roughly timestamp) or mtime
                        latest_offline = sorted(offlines, key=lambda p: p.stat().st_mtime)[-1]
                        if (latest_offline / "files").exists():
                            return latest_offline / "files"
    except (PermissionError, OSError):
        pass

    return None

def load_run_data(run_dir):
    if not run_dir: return None
    
    # Use explicit finder instead of rglob
    files_dir = find_wandb_files_explicit(run_dir)
    
    if not files_dir:
        # Last ditch: try rglob if explicit structure failed (e.g. diff config)
        try:
            candidates = list(run_dir.rglob("files"))
            if candidates:
                files_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        except:
            pass
            
    if not files_dir:
        return None

    try:
        p1 = files_dir / "validation_generations.pkl"
        p2 = files_dir / "uncertainty_measures.pkl"
        p3 = files_dir / "probes.pkl"
        
        if not (p1.exists() and p2.exists() and p3.exists()):
            return None
            
        with open(p1, "rb") as f: gens = pickle.load(f)
        with open(p2, "rb") as f: unc = pickle.load(f)
        with open(p3, "rb") as f: probes = pickle.load(f)
        return gens, unc, probes
    except Exception:
        return None

# ==========================================
# 2. Data Processing (Same as before)
# ==========================================

def extract_features(gens, unc, probes, n_cap=2000):
    sorted_keys = sorted(gens.keys())
    gen_values = [gens[k] for k in sorted_keys]
    
    accuracies = np.array([g["most_likely_answer"]["accuracy"] for g in gen_values])
    y_hallucination = (accuracies < 1.0).astype(int) 
    
    ent_keys = ["cluster_assignment_entropy", "semantic_entropy_sum_normalized"]
    raw_entropy = None
    for k in ent_keys:
        if k in unc["uncertainty_measures"]:
            raw_entropy = np.array(unc["uncertainty_measures"][k])
            break
    if raw_entropy is None: 
        raw_entropy = np.zeros_like(accuracies)

    tbg_raw = [g["most_likely_answer"]["emb_last_tok_before_gen"] for g in gen_values]
    if tbg_raw[0] is None:
        raise ValueError("Embeddings missing")
    
    X_tensor = torch.stack(tbg_raw).squeeze(-2).transpose(0, 1).numpy()
    
    n = min(len(y_hallucination), X_tensor.shape[1], n_cap)
    X_tensor = X_tensor[:, :n, :]
    y_hallucination = y_hallucination[:n]
    raw_entropy = raw_entropy[:n]
    
    res_key = list(probes['results'].keys())[0]
    splits = probes['results'][res_key]['splits']
    models_data = probes['results'][res_key]
    
    return {
        'X': X_tensor, 'y': y_hallucination, 'entropy': raw_entropy,
        'splits': splits, 'models_data': models_data
    }

def select_best_layers(data):
    X = data['X']; y = data['y']; cal_idx = data['splits']['val']
    acc_models = data['models_data']['ta_models']
    se_models = data['models_data']['tb_models']
    total_layers = min(len(acc_models), len(se_models), X.shape[0])

    acc_aucs = []
    best_acc_auc = 0.0; best_acc_layer = -1
    for layer_idx in range(total_layers):
        X_cal = X[layer_idx][cal_idx]; y_cal = y[cal_idx]
        if len(np.unique(y_cal)) < 2: acc_aucs.append(0.5); continue
        preds = acc_models[layer_idx].predict_proba(X_cal)[:, 0] 
        auc = roc_auc_score(y_cal, preds)
        acc_aucs.append(auc)
        if auc > best_acc_auc: best_acc_auc = auc; best_acc_layer = layer_idx

    se_aucs = []
    best_se_auc = 0.0; best_se_layer = -1
    for layer_idx in range(total_layers):
        X_cal = X[layer_idx][cal_idx]; y_cal = y[cal_idx]
        if len(np.unique(y_cal)) < 2: se_aucs.append(0.5); continue
        preds = se_models[layer_idx].predict_proba(X_cal)[:, 1] 
        auc = roc_auc_score(y_cal, preds)
        se_aucs.append(auc)
        if auc > best_se_auc: best_se_auc = auc; best_se_layer = layer_idx
            
    return best_acc_layer, best_se_layer, acc_aucs, se_aucs, total_layers

def get_scores_dataframe(data, best_acc_layer, best_se_layer):
    df = pd.DataFrame()
    splits = data['splits']
    
    idx_map = np.zeros(len(data['y']), dtype=object)
    idx_map[splits['train']] = 'train'
    idx_map[splits['val']] = 'calibration'
    idx_map[splits['test']] = 'test'
    
    df['split'] = idx_map
    df['y_true'] = data['y']
    df['se_raw'] = data['entropy']
    
    df['s_acc'] = data['models_data']['ta_models'][best_acc_layer].predict_proba(data['X'][best_acc_layer])[:, 0]
    df['s_se_probe'] = data['models_data']['tb_models'][best_se_layer].predict_proba(data['X'][best_se_layer])[:, 1]
    return df

def train_combiner(df):
    cal_df = df[df['split'] == 'calibration']
    if len(cal_df['y_true'].unique()) < 2:
        df['s_combined'] = df['s_acc']
        return df
    
    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(cal_df[['s_acc', 's_se_probe']], cal_df['y_true'])
    
    df['s_combined'] = lr.predict_proba(df[['s_acc', 's_se_probe']])[:, 1]
    return df

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
# 3. Main Loop
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, required=True, help="Path to runs (e.g., outputs/runs)")
    args = parser.parse_args()

    repo_root, runs_root = resolve_paths(args.runs_root)

    # Output directory for figures
    figures_dir = repo_root / "paper_figures"
    figures_dir.mkdir(exist_ok=True)
    
    results = {'detection': [], 'conformal': [], 'layers': [], 'layer_log': []}
    
    print(f"Scanning {runs_root}...\n")
    
    for model_name in TARGET_MODELS:
        for dataset_name in TARGET_DATASETS:
            
            run_dir, expected_name = find_run_directory(runs_root, model_name, dataset_name)
            
            print(f"[{expected_name}]", end=" ")
            if not run_dir:
                print("Not found.")
                continue
            
            raw_data = load_run_data(run_dir)
            if not raw_data: 
                print("Found but files missing.")
                continue
            else:
                print("Loaded.")

            # Processing
            data_dict = extract_features(*raw_data)
            
            safe_name = Path(model_name).name
            b_acc, b_se, acc_aucs, se_aucs, n_layers = select_best_layers(data_dict)
            results['layers'].append({'Model': safe_name, 'Dataset': dataset_name, 'Acc_AUCs': acc_aucs, 'SE_AUCs': se_aucs})
            results['layer_log'].append({'Model': safe_name, 'Dataset': dataset_name, 'Best_Acc': b_acc, 'Best_SE': b_se, 'Total': n_layers})
            
            df = train_combiner(get_scores_dataframe(data_dict, b_acc, b_se))
            df_test = df[df['split'] == 'test']
            if df_test.empty: continue
            
            thresh_30 = df_test['se_raw'].quantile(0.30)
            df_conf = df_test[df_test['se_raw'] <= thresh_30]
            
            methods = {'Semantic Entropy': 'se_raw', 'Accuracy Probe': 's_acc', 'SE Probe': 's_se_probe', 'Combined (LR)': 's_combined'}
            for m_name, col in methods.items():
                try: auc_f = roc_auc_score(df_test['y_true'], df_test[col])
                except: auc_f = 0.5
                try: auc_c = roc_auc_score(df_conf['y_true'], df_conf[col])
                except: auc_c = 0.5
                results['detection'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Subset': 'Full Test', 'AUROC': auc_f})
                results['detection'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Subset': 'Confident Subset', 'AUROC': auc_c})
            
            for alpha in [0.01, 0.05, 0.10]:
                for m_name, col in methods.items():
                    r, c = conformal_risk_control(df, col, alpha)
                    results['conformal'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': m_name, 'Target_Alpha': alpha, 'Realized_Risk': r, 'Coverage': c})
                r2, c2 = conformal_2d_search(df, alpha)
                results['conformal'].append({'Model': safe_name, 'Dataset': dataset_name, 'Method': 'Dual-Probe (2D)', 'Target_Alpha': alpha, 'Realized_Risk': r2, 'Coverage': c2})

    # ==========================================
    # 4. Plotting
    # ==========================================
    if not results['detection']:
        print("\nNo results gathered.")
        return

    print(f"\nSaving figures to: {figures_dir}")
    df_det = pd.DataFrame(results['detection'])
    df_conf = pd.DataFrame(results['conformal'])
    df_layers = pd.DataFrame(results['layer_log'])

    if not df_layers.empty: df_layers.to_csv(figures_dir / "layer_selection_log.csv", index=False)
    if not df_det.empty: df_det.to_csv(figures_dir / "results_detection.csv", index=False)
    if not df_conf.empty: df_conf.to_csv(figures_dir / "results_conformal.csv", index=False)

    # 1. Detection Bar Plots (PER MODEL)
    unique_models = df_det['Model'].unique()
    
    for model in unique_models:
        model_df = df_det[df_det['Model'] == model]
        
        for sub in ['Full Test', 'Confident Subset']:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=model_df[model_df['Subset']==sub], 
                x='Dataset', y='AUROC', hue='Method', 
                palette=COLORS, edgecolor='black', errorbar=None
            )
            plt.title(f"{model} - Detection ({sub})")
            plt.ylim(0.4, 1.0)
            plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            
            # Safe filename
            safe_sub = sub.replace(' ', '_')
            plt.savefig(figures_dir / f"detection_{model}_{safe_sub}.png", dpi=300)
            plt.close()

    # 2. Coverage Bar Plots (PER MODEL)
    if not df_conf.empty:
        target_alpha = 0.05
        conf_sub = df_conf[df_conf['Target_Alpha'] == target_alpha]
        
        unique_models_conf = conf_sub['Model'].unique()
        
        for model in unique_models_conf:
            model_conf_df = conf_sub[conf_sub['Model'] == model]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=model_conf_df, 
                x='Dataset', y='Coverage', hue='Method', 
                palette=COLORS, edgecolor='black', errorbar=None
            )
            plt.title(f"{model} - Coverage @ Risk={target_alpha}")
            plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(figures_dir / f"coverage_{model}_alpha{target_alpha}.png", dpi=300)
            plt.close()

    # 3. Layer Sensitivity (Aggregate Plot still fine, or split if too many)
    if results['layers']:
        n = len(results['layers'])
        # If too many plots, maybe split? For now keep aggregated grid.
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        for i, item in enumerate(results['layers']):
            ax = axes[i]
            x = np.linspace(0, 1, len(item['Acc_AUCs']))
            ax.plot(x, item['Acc_AUCs'], label='Acc Probe', color=COLORS['Accuracy Probe'], linewidth=2)
            ax.plot(x, item['SE_AUCs'], label='SE Probe', color=COLORS['SE Probe'], linestyle='--', linewidth=2)
            ax.set_title(f"{item['Model']}\n{item['Dataset']}")
            ax.set_ylim(0.4, 1.0)
            if i==0: ax.legend()
        for j in range(i+1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(figures_dir / "layer_sensitivity_dual.png", dpi=300)
        plt.close()

    print("Done.")

if __name__ == "__main__":
    main()