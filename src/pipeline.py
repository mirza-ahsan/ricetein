"""Run the declared experiments through Stage 7, resumably. Stage 8 is intentionally separate.

    uv run python -m src.pipeline --config configs/pipeline.yaml --stage all

Complete the Stage 2 audit and commit the locked splits/code before starting. While this process runs,
do not edit src/: worker processes import it from disk. Failures stop the pipeline and preserve artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import torch

from src.config import REPO_ROOT, load_config, output_root
from src.records import get_default_runs_dir, require_clean_code


def mark(stage, state):
    path = output_root() / "runs/pipeline_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "state": state,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": require_clean_code(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)
    print(f"Stage {stage}: {state}", flush=True)


def preflight(config):
    require_clean_code()
    for path in (
        "data/groups.csv",
        "data/label_conflicts.csv",
        "data/cross_dataset_overlap.csv",
        "splits/folds.csv",
        "splits/paddy4.csv",
        "data/duplicate_miss_rule.json",
    ):
        if not (REPO_ROOT / path).exists():
            raise FileNotFoundError(f"finish audits and locked splits before experiments: {path}")
    if not json.loads((REPO_ROOT / "data/duplicate_miss_rule.json").read_text())["applied"]:
        raise ValueError("duplicate miss rule must be applied before experiments")
    torch.set_num_threads(config["pipeline"]["cpu_threads"])


def execute(config, stage):
    preflight(config)
    workers = config["pipeline"]["workers"]
    if stage in ("all", "4"):
        mark("4", "running")
        from src import sanity_baselines as sanity

        c = load_config("configs/sanity.yaml")
        sanity.report(c, sanity.run(c, workers=workers))
        mark("4", "completed")
    if stage in ("all", "6"):
        mark("6", "running")
        from src import cache_report, embed

        for name in config["pipeline"]["caches"]:
            embed.run(load_config(f"configs/embed/{name}.yaml"), workers)
        cache_report.run(config)
        mark("6", "completed_checks_galleries_await_visual_review")
    if stage in ("all", "7"):
        mark("7", "training")
        from src import linear_probe, train_mil

        for size in (224, 512):
            linear_probe.run(load_config(f"configs/mil/probe_e1_{size}.yaml"))
        for aggregator in ("mean", "max", "attention"):
            c = load_config(f"configs/mil/{aggregator}.yaml")
            train_mil.tune(c)
            train_mil.final(c, [c["seed"], *c["mil"]["stability_seeds"]])
        mark("7", "validation_and_stability_fits_completed")
    if stage in ("all", "5"):
        mark("5", "training")
        from src import stage5

        stage5.run()
        mark("5", "validation_selection_completed")
    if stage in ("all", "evaluate"):
        mark("5_and_7", "freezing_test_plan")
        from src import evaluate, experiment_report

        five, seven = experiment_report.configurations()
        names = [name for item in five + seven for name in item["runs"]]
        path = get_default_runs_dir() / "evaluation_plan.json"
        evaluate.freeze_plan(names, path)
        for name in names:
            folder = get_default_runs_dir() / name
            if (folder / "test_metrics.json").exists() and (folder / "test_predictions.csv").exists():
                continue
            evaluate.evaluate(name, path)
        experiment_report.build(config)
        mark("5_and_7", "test_evaluation_and_reports_completed")
    if stage == "report":
        from src import experiment_report

        experiment_report.build(config)
    mark(stage, "action_completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--stage", choices=["all", "4", "5", "6", "7", "evaluate", "report"], default="all")
    args = parser.parse_args()
    execute(load_config(args.config), args.stage)


if __name__ == "__main__":
    main()
