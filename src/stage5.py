"""Stage 5 validation-only sweep: search at 224, reuse each architecture's LR, then high resolution."""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from src import train_wholeimage as tw
from src.config import load_config
from src.records import get_default_runs_dir, require_clean_code, reusable_run
from src.split import read_folds

MODELS = ("inceptionv3", "efficientnetv2s", "convnext_tiny", "swin_tiny")


def run_one(config, lr, r):
    name = tw.run_name(config, lr, r)
    _, checksum = read_folds(config)
    if not reusable_run(
        name, {**config, "lr": lr, "round": r}, checksum, ("model.pt", "metrics.json", "predictions.csv")
    ):
        folder = get_default_runs_dir() / name
        if folder.exists():
            # Preserve interrupted artifacts rather than silently deleting run history.
            n = 1
            while folder.with_name(f"{name}_interrupted{n}").exists():
                n += 1
            folder.rename(folder.with_name(f"{name}_interrupted{n}"))
        tw.train(config, lr, r)
    return name, json.loads((get_default_runs_dir() / name / "metrics.json").read_text())["val"]["macro_f1"]


def select_lr(trials):
    """Trials contain only five-round validation scores. Ties follow the declared LR order."""
    if any(len(t["validation_scores"]) != 5 for t in trials):
        raise ValueError("learning-rate selection requires all five validation rounds")
    return max(trials, key=lambda t: float(np.mean(t["validation_scores"])))


def run():
    require_clean_code()
    selections, configurations = {}, []
    for model in MODELS:
        config = load_config(f"configs/wholeimage/{model}_224.yaml")
        trials = []
        for lr in config["wholeimage"]["lrs"]:
            results = [run_one(config, lr, r) for r in range(1, 6)]
            trials.append(
                {"lr": lr, "runs": [n for n, _ in results], "validation_scores": [s for _, s in results]}
            )
        best = select_lr(trials)
        selections[model] = {
            "trials": trials,
            "selected_lr": best["lr"],
            "mean_val_macro_f1": float(np.mean(best["validation_scores"])),
        }
        configurations.append({"name": f"{model}_224", "runs": best["runs"]})
        for suffix in ("128", "224_otsu"):
            other = load_config(f"configs/wholeimage/{model}_{suffix}.yaml")
            runs = [run_one(other, best["lr"], r)[0] for r in range(1, 6)]
            configurations.append({"name": f"{model}_{suffix}", "runs": runs})
    # Choose the high-resolution architecture at the common 224px, no-Otsu condition, on validation only.
    best_model = max(MODELS, key=lambda m: selections[m]["mean_val_macro_f1"])
    config = load_config("configs/wholeimage/best_448.yaml")
    config = copy.deepcopy(config)
    config["wholeimage"]["model"] = best_model
    lr = selections[best_model]["selected_lr"]
    runs = [run_one(config, lr, r)[0] for r in range(1, 6)]
    configurations.append({"name": f"{best_model}_448", "runs": runs})
    result = {
        "selection_partition": "val",
        "architectures": selections,
        "best_448_model": best_model,
        "configurations": configurations,
    }
    (get_default_runs_dir() / "stage5_selection.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("Stage 5 validation sweeps complete:", run()["best_448_model"])


if __name__ == "__main__":
    main()
