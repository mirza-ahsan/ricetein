"""One-time test evaluation of completed runs listed in a frozen evaluation plan.

Selection is based exclusively on validation. The plan pins checkpoint hashes and split checksums before
any test score is computed. A started marker prevents concurrent evaluation or silent retries after failure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src import metrics
from src.records import get_default_runs_dir, require_clean_code, reusable_run
from src.seeding import seed_everything
from src.split import read_folds


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(run_dir):
    return run_dir / ("model.pkl" if run_dir.name.startswith("stage7a_") else "model.pt")


def freeze_plan(names: list[str], path: Path) -> dict:
    """Call only once all Stage 5 and Stage 7 validation searches/final fits are finished."""
    version = require_clean_code()
    entries = {}
    for name in names:
        folder = get_default_runs_dir() / name
        record = json.loads((folder / "record.json").read_text())
        if (
            record["status"] != "completed"
            or record["code_version"] == "unknown"
            or record["code_version"].endswith("-dirty")
        ):
            raise ValueError(f"{name}: incomplete or untraceable training run")
        if not reusable_run(name, record["config"], record["split_checksum"],
                            ("metrics.json", "predictions.csv", checkpoint(folder).name)):
            raise ValueError(f"{name}: missing completed training artifacts")
        entries[name] = {
            "record_sha256": digest(folder / "record.json"),
            "checkpoint_sha256": digest(checkpoint(folder)),
            "split_checksum": record["split_checksum"],
        }
    plan = {"selection_partition": "val", "code_version": version, "runs": entries}
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("the frozen test plan cannot be changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as out:
            json.dump(plan, out, indent=2)
    return plan


def restore_size_keys(config: dict) -> dict:
    """A copy of a recorded Stage 5 config whose size tables are keyed by integers again.

    record.json is JSON, which turns the integer keys of ``batch_size`` and of each model's
    ``size_kwargs`` into strings; looked up by an integer size, the first raises and the second
    silently returns nothing (a Swin model at 128 or 448 px would be built for 224 px).
    """
    config = copy.deepcopy(config)
    whole = config["wholeimage"]
    whole["batch_size"] = {int(size): batch for size, batch in whole["batch_size"].items()}
    for spec in whole["models"].values():
        if "size_kwargs" in spec:
            spec["size_kwargs"] = {int(size): kwargs for size, kwargs in spec["size_kwargs"].items()}
    return config


def evaluate_stage5(run_dir, record):
    from src import train_wholeimage as tw

    config = restore_size_keys(record["config"])
    round_ = config["round"]
    seed_everything(config["seed"], config["wholeimage"]["determinism"])
    folds, _ = read_folds(config)
    device = torch.device("cuda")
    model = tw.build_model(config, config["wholeimage"]["size"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    loader = tw.loaders(config, folds, round_, ("test",))["test"]
    rows, probs = tw.predict(model.to(device), loader, config, device)
    return tw.prediction_table(folds, rows, probs, round_, "test", config["classes"])


def evaluate_probe(run_dir, record):
    from src import linear_probe as lp
    from src.split import round_column
    from src.train_wholeimage import prediction_table

    config = record["config"]
    seed_everything(config["seed"], config["determinism"])
    folds, _ = read_folds(config)
    x = lp.descriptors(config, folds, 1)[0]
    with (run_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)  # only our own local, checksum-verified checkpoint
    rows = np.flatnonzero((folds[round_column(config["round"])] == "test").to_numpy())
    probs = model.predict_proba(x[rows])[:, np.argsort(model.classes_)]
    return prediction_table(folds, rows, probs, config["round"], "test", config["classes"])


def evaluate_mil(run_dir, record):
    from src import train_mil as tm
    from src.mil_models import MILModel
    from src.train_wholeimage import prediction_table

    config = record["config"]
    seed_everything(config.get("run_seed", config["seed"]), config["determinism"])
    mil, params, r = config["mil"], config["params"], config["round"]
    folds, _ = read_folds(config)
    clean_config = {**config, "mil": {**mil, "versions": 1}}
    store = tm.store_for(clean_config, folds, params["descriptor"])
    device = torch.device(mil.get("device", "cuda"))
    model = MILModel(store.dim, params["width"], params["dropout"], mil["aggregator"], len(config["classes"]))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    bags = tm.Bags(store, folds, r, "test", config["classes"], 1, 0.0, mil["batch"], config["seed"])
    rows, probs, weights = tm.predict(model.to(device), bags, device)
    if weights[0] is not None:
        np.savez_compressed(
            run_dir / "attention.npz", **{folds.photo_id.iloc[row]: w for row, w in zip(rows, weights)}
        )
    return prediction_table(folds, rows, probs, r, "test", config["classes"])


EVALUATORS = {"stage5": evaluate_stage5, "stage7a": evaluate_probe, "stage7": evaluate_mil}


def evaluate(name: str, plan_path: Path | None = None) -> dict:
    version = require_clean_code()
    run_dir = get_default_runs_dir() / name
    if any(
        (run_dir / f).exists() for f in ("test_predictions.csv", "test_metrics.json", "test_started.json")
    ):
        raise ValueError(f"{name}: test already evaluated or started; no silent retry")
    plan_path = plan_path or get_default_runs_dir() / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["selection_partition"] != "val" or name not in plan["runs"]:
        raise ValueError(f"{name}: not selected in the frozen validation-only evaluation plan")
    expected = plan["runs"][name]
    record = json.loads((run_dir / "record.json").read_text())
    _, checksum = read_folds(record["config"])
    if checksum != expected["split_checksum"] or checksum != record["split_checksum"]:
        raise ValueError(f"{name}: trained on a different split")
    if (
        digest(run_dir / "record.json") != expected["record_sha256"]
        or digest(checkpoint(run_dir)) != expected["checkpoint_sha256"]
    ):
        raise ValueError(f"{name}: training record or checkpoint changed after selection")
    if (
        record["status"] != "completed"
        or record["code_version"] == "unknown"
        or record["code_version"].endswith("-dirty")
    ):
        raise ValueError(f"{name}: incomplete or untraceable training")
    with (run_dir / "test_started.json").open("x") as out:
        json.dump({"evaluation_code": version, "plan_sha256": digest(plan_path)}, out)
    predictions = EVALUATORS[name.split("_")[0]](run_dir, record)
    result = {
        "partition": "test",
        **metrics.scores(predictions.true_class, predictions.pred_class, record["config"]["classes"]),
    }
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "test_metrics.json").write_text(json.dumps({**result, "code_version": version}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    for name in args.runs:
        evaluate(name, args.plan)
        print(f"{name}: test evaluated")  # scores are read by the reporting step after all tuning is frozen


if __name__ == "__main__":
    main()
