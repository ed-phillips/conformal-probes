import argparse
import subprocess
import yaml
from pathlib import Path
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--runs-root", type=str, required=True)
    p.add_argument("--no-compute-uncertainties", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = cfg["models"]
    datasets = cfg["datasets"]
    gen = cfg.get("generation", {})

    judge_model = cfg.get("judge_model")  

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        # e.g. model="meta-llama/Llama-3.1-8B-Instruct"
        # HuggingfaceModel expects just "Llama-3.1-8B-Instruct"
        model_name_for_sep = Path(model).name

        for ds in datasets:
            run_dir = runs_root / f"{model_name_for_sep}__{ds}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Always run as a module; we’re in REPO_ROOT so this is fine
            cmd = [
                "python",
                "-m", "semantic_uncertainty.generate_answers",
                "--model_name", model_name_for_sep,
                "--dataset", ds,
                "--num_samples", str(gen.get("num_samples", 2000)),
                "--num_generations", str(gen.get("num_generations", 10)),
                "--num_few_shot", str(gen.get("num_few_shot", 0)),
                "--temperature", str(gen.get("temperature", 1.0)),
                "--model_max_new_tokens", str(gen.get("model_max_new_tokens", 50)),
                "--metric", "hf_judge",
                "--brief_prompt", gen.get("brief_prompt", "default"),
            ]

            # HF-judge labeler
            if judge_model is not None:
                cmd.extend(["--judge_model_name", judge_model])

            if not gen.get("use_context", False):
                cmd.append("--no-use_context")

            # IMPORTANT: disable calling compute_uncertainty_measures in __main__
            if args.no_compute_uncertainties:
                cmd.append("--no-compute_uncertainties")

            # Inherit current env & override SCRATCH_DIR for this run
            env = os.environ.copy()
            env["SCRATCH_DIR"] = str(run_dir)

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()