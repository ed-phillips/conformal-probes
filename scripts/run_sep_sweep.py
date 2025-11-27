import argparse
import subprocess
import yaml
from pathlib import Path

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

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        for ds in datasets:
            run_dir = runs_root / f"{Path(model).name}__{ds}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", "-m", "semantic_uncertainty.generate_answers"
                if Path("semantic_uncertainty/generate_answers.py").exists()
                else "semantic_uncertainty/generate_answers.py",
                "--model_name", Path(model).name,  # because HuggingfaceModel reconstructs base prefix
                "--dataset", ds,
                "--num_samples", str(gen.get("num_samples", 2000)),
                "--num_generations", str(gen.get("num_generations", 10)),
                "--num_few_shot", str(gen.get("num_few_shot", 0)),
                "--temperature", str(gen.get("temperature", 1.0)),
                "--model_max_new_tokens", str(gen.get("model_max_new_tokens", 50)),
                "--metric", "hf_judge",
                "--judge_model_name", cfg.get("judge_model", None),
                "--brief_prompt", gen.get("brief_prompt", "default"),
            ]

            if not gen.get("use_context", False):
                cmd.append("--no-use_context")

            # IMPORTANT: disable calling compute_uncertainty_measures in __main__
            if args.no_compute_uncertainties:
                cmd.append("--no-compute_uncertainties")

            # We rely on SCRATCH_DIR and wandb.run.dir for paths,
            # so set SCRATCH_DIR to this run_dir's parent and use wandb offline
            env = dict(**dict(Path.cwd().env if hasattr(Path.cwd(), "env") else {}))
            env.update(**{
                "SCRATCH_DIR": str(run_dir),
            })

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
