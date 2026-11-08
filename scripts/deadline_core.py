"""Tonight's complete core comparisons. No reduced search or early-stopping budgets.

Train three independent 40-trial MIL studies and the two whole-photo probes. Final fits use the
headline seed; additional stability seeds and ablation caches are explicitly deferred. Cloud
whole-image runs are imported only if their records and all artifacts are complete.
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate, stats, train_mil
from src.config import REPO_ROOT, load_config
from src.inventory_report import md_table
from src.records import get_default_runs_dir, require_clean_code
from src.sanity_baselines import summarise
from src.split import read_folds


def train_worker(aggregator):
    c = load_config(f"configs/mil/{aggregator}.yaml")
    train_mil.tune(c)
    train_mil.final(c, [c["seed"]])


def probe_worker():
    from src import linear_probe

    for size in (224, 512):
        linear_probe.run(load_config(f"configs/mil/probe_e1_{size}.yaml"))


def train(source):
    version = require_clean_code()
    config = load_config("configs/mil/attention.yaml")
    folds, _ = read_folds(config)
    # Build shared packed arrays once before parallel studies can request the same files.
    train_mil.store_for(config, folds, "cm")
    from src.embed import completed_cache

    completed_cache(config, "e1_dinov3_vitb16")
    print(f"Core MIL studies begin at commit {version}; original budgets", flush=True)

    def launch(action):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), action],
                       env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, check=True)

    def wait_for_wholephoto():
        # Keep the already-running core cache extraction, then stop the pipeline before its
        # out-of-scope alternative caches begin.
        while not (source / "cache/e1_wholephoto/complete.json").exists():
            print("Waiting for the whole-photo core cache before running probe fits", flush=True)
            time.sleep(10)
        status_path = source / "runs/pipeline_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("stage") == "6" and status.get("state") == "running":
                os.kill(status["pid"], signal.SIGINT)
        completed_cache(config, "e1_wholephoto")
        launch("probes")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(launch, name) for name in ("mean", "max", "attention")]
        futures.append(pool.submit(wait_for_wholephoto))
        for future in futures:
            future.result()
    (get_default_runs_dir() / "deadline_core_training_complete.json").write_text(
        json.dumps(
            {
                "code_version": version,
                "tuning_trials_each": 40,
                "rounds": 5,
                "deferred": [
                    "additional stability seeds",
                    "Stage 5 128/Otsu/448",
                    "alternative caches",
                    "Stages 8–10",
                ],
            },
            indent=2,
        )
    )


def configurations():
    root = get_default_runs_dir()
    items = []
    incomplete = []
    for model in ("inceptionv3", "efficientnetv2s", "convnext_tiny", "swin_tiny"):
        trials = []
        for lr in (3e-5, 1e-4, 3e-4):
            names = [f"stage5_{model}_224_lr{lr:g}_r{r}" for r in range(1, 6)]
            if not all(
                (root / name / "record.json").exists()
                and json.loads((root / name / "record.json").read_text())["status"] == "completed"
                and all((root / name / f).exists() for f in ("metrics.json", "predictions.csv", "model.pt"))
                for name in names
            ):
                break
            score = np.mean(
                [json.loads((root / name / "metrics.json").read_text())["val"]["macro_f1"] for name in names]
            )
            trials.append((score, names))
        if len(trials) != 3:
            incomplete.append(model)
        else:
            items.append({"name": model + "_224", "runs": max(trials, key=lambda t: t[0])[1]})
    for size in (224, 512):
        selected = json.loads((root / f"stage7a_selection_e1_{size}.json").read_text())
        items.append({"name": f"probe_{size}", "runs": selected["runs"]})
    for method in ("mean", "max", "attention"):
        items.append({"name": method, "runs": [f"stage7_{method}_r{r}" for r in range(1, 6)]})
    return items, incomplete


def report():
    config = load_config("configs/pipeline.yaml")
    root = get_default_runs_dir()
    items, incomplete = configurations()
    if not (root / "deadline_core_training_complete.json").exists():
        raise RuntimeError("all original core MIL/probe budgets must complete before test evaluation")
    plan = root / "deadline_evaluation_plan.json"
    evaluate.freeze_plan([name for item in items for name in item["runs"]], plan)
    rows, tables = [], {}
    target = REPO_ROOT / "results/tables"
    target.mkdir(parents=True, exist_ok=True)
    for item in items:
        for name in item["runs"]:
            folder = root / name
            if not ((folder / "test_metrics.json").exists() and (folder / "test_predictions.csv").exists()):
                evaluate.evaluate(name, plan)
        table = pd.concat(
            [
                pd.read_csv(root / name / filename)
                for name in item["runs"]
                for filename in ("predictions.csv", "test_predictions.csv")
            ],
            ignore_index=True,
        )
        result = summarise(table, config)
        tables[item["name"]] = table
        (target / f"deadline_{item['name']}_full_metrics.json").write_text(json.dumps(result, indent=2))
        rows.append(
            {
                "method": item["name"],
                **{
                    f"{key}_{s}": value
                    for key, summary in result["test"].items()
                    for s, value in summary.items()
                },
                "ece": result["ece"],
                "hispa_recall": result["pooled_test"]["per_class"]["hispa"]["recall"],
            }
        )
    comparisons = []
    for name, table in tables.items():
        if name == "attention":
            continue
        for label, scorer in (
            ("macro_f1", stats.macro_f1(config["classes"])),
            ("balanced_accuracy", stats.balanced_accuracy),
        ):
            comparisons.append(
                {
                    "A": "attention",
                    "B": name,
                    "metric": label,
                    **stats.compare(
                        tables["attention"],
                        table,
                        scorer,
                        config["evaluation"]["bootstrap"],
                        config["evaluation"]["comparison"],
                        config["seed"],
                    ),
                }
            )
    frame, paired = pd.DataFrame(rows), pd.DataFrame(comparisons)
    frame.to_csv(target / "deadline_core_scores.csv", index=False)
    paired.to_csv(target / "deadline_core_paired.csv", index=False)
    report = REPO_ROOT / "docs/reports/deadline_core_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Deadline core results\n\nOriginal tuning/epoch budgets and five rounds; "
        "seed-0 headline models only. Test was evaluated once after validation selection. "
        "Stability seeds, 128/Otsu/448 whole-image settings, ablations, partition-order and Paddy "
        "evaluation are deferred. Incomplete whole-image searches are excluded rather than selected "
        f"from a reduced budget: {incomplete}. Values below are proportions.\n\n"
        + md_table(frame)
        + "\n\n"
        + md_table(paired)
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "mean", "max", "attention", "probes", "report"])
    parser.add_argument("--source", type=Path, default=Path("/home/aztrek/projects/ricetein"))
    args = parser.parse_args()
    if args.action in ("mean", "max", "attention"):
        train_worker(args.action)
    elif args.action == "probes":
        probe_worker()
    elif args.action == "train":
        train(args.source)
    else:
        report()
