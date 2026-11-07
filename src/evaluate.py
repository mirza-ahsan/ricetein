"""The one-time test evaluation (golden rule 4): a selected, frozen configuration scores its test fold.

For each run folder it loads the saved best checkpoint, predicts the photos whose role is ``test`` in that
run's round, and writes ``test_predictions.csv`` and ``test_metrics.json`` next to the run's validation
results. It refuses to run twice on the same run, refuses a run trained on a different split, and refuses
code that is not a clean commit (test scores must trace to exact code).

    uv run python -m src.evaluate --runs stage5_convnext_tiny_224_lr0.0001_r1 ...
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
import torch

from src import metrics
from src.records import get_code_version, get_default_runs_dir
from src.split import read_folds


def evaluate_stage5(run_dir, record: dict[str, Any]) -> pd.DataFrame:
    from src import train_wholeimage as tw

    config, round_ = record["config"], record["config"]["round"]
    folds, _ = read_folds(config)
    device = torch.device("cuda")
    model = tw.build_model(config, config["wholeimage"]["size"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    loader = tw.loaders(config, folds, round_, ("test",))["test"]
    rows, probs = tw.predict(model.to(device), loader, config, device)
    return tw.prediction_table(folds, rows, probs, round_, "test", config["classes"])


EVALUATORS = {"stage5": evaluate_stage5}


def evaluate(name: str) -> dict[str, Any]:
    run_dir = get_default_runs_dir() / name
    if (run_dir / "test_predictions.csv").exists():
        raise SystemExit(f"{name}: already evaluated on test (it is evaluated once)")
    record = json.loads((run_dir / "record.json").read_text())
    _, checksum = read_folds(record["config"])
    if record["split_checksum"] != checksum:
        raise SystemExit(f"{name}: trained on a different split ({record['split_checksum']} vs {checksum})")
    predictions = EVALUATORS[name.split("_")[0]](run_dir, record)
    result = metrics.scores(predictions["true_class"], predictions["pred_class"], record["config"]["classes"])
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "test_metrics.json").write_text(json.dumps({**result, "code_version": get_code_version()},
                                                          indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    version = get_code_version()
    if version.endswith("-dirty") or version == "unknown":
        raise SystemExit(f"code version {version}: commit before evaluating on test")
    for name in args.runs:
        evaluate(name)
        print(f"{name}: test evaluated")  # scores go to the run folder, not the screen


if __name__ == "__main__":
    main()
