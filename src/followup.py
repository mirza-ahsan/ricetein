"""The follow-up study declared on 2026-11-14 (handbook §24; manuals/followup_plan.md).

    uv run python -m src.followup <action>

baselines       the extra shortcut controls on both splits
session-cnn     the four core whole-image models retrained on the session-blocked split
session-heads   the two core probes and three core MIL heads refitted on the session-blocked split
extend-cnn      edge rule for the whole-image learning rate (locked split)
extend-probes   edge rule for the probe L2 strength (locked split)
wide <name>     the second 40-trial study of one aggregator, then its final fits
whole <size>    the same-head whole-photo control: bag cache, 40-trial study, final fits
seeds           the declared stability seeds of the three core MIL heads (validation only)
select          write runs/followup_selection.json from validation scores only
evaluate        freeze runs/followup_evaluation_plan.json, then test each new configuration once
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from src import evaluate, linear_probe, shortcut_baselines, stage5, train_mil, wholephoto_bags
from src.config import REPO_ROOT, load_config
from src.embed import completed_cache
from src.records import RunRecorder, completed_run, get_default_runs_dir, require_clean_code, reusable_run
from src.split import read_folds
from src.train_wholeimage import prediction_table, run_name

MODELS = stage5.MODELS
AGGREGATORS = ("mean", "max", "attention")
SIZES = (224, 512)           # whole-photo sizes of the two probes and the two whole-photo controls
ROUNDS = range(1, 6)
LR_LADDER = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
L2_LADDER = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
MAX_EXTENSIONS = 3


def read_json(path) -> Any:
    return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
    (get_default_runs_dir() / name).write_text(json.dumps(data, indent=2) + "\n")


def core_methods() -> dict[str, list[str]]:
    """The nine frozen configurations of the 2026-11-08 core study: method -> its five run names."""
    provenance = read_json(REPO_ROOT / "results/tables/deadline_report_provenance.json")
    return {method["name"]: method["runs"] for method in provenance["methods"]}


def core_config(method: str) -> dict[str, Any]:
    return read_json(get_default_runs_dir() / core_methods()[method][0] / "record.json")["config"]


def best_value(scores: dict[float, float]) -> float:
    """The value with the highest score; the smaller value wins a tie."""
    return max(sorted(scores), key=lambda value: scores[value])


def next_value(scores: dict[float, float], ladder: list[float], declared: list[float]) -> float | None:
    """The edge rule (D6): the next ladder value to try, or None when the search is finished."""
    values = sorted(scores)
    if len(values) - len(declared) >= MAX_EXTENSIONS:
        return None
    best = best_value(scores)
    at = ladder.index(best)
    if best == values[-1] and at + 1 < len(ladder):
        return ladder[at + 1]
    if best == values[0] and at > 0:
        return ladder[at - 1]
    return None


def fit_rounds(config: dict[str, Any], params: dict[str, Any], seed: int) -> list[str]:
    """Five final fits of one MIL head with given settings. Same steps as train_mil.final, without its
    requirement that the tuning study used this config's split (the session runs reuse locked-split settings)."""
    require_clean_code()
    config = {**config, "cache_spec_sha256": completed_cache(config, config["mil"]["cache"])["spec_sha256"]}
    mil, classes = config["mil"], config["classes"]
    folds, checksum = read_folds(config)
    store = train_mil.store_for(config, folds, params["descriptor"])
    names = []
    for r in ROUNDS:
        name = f"stage7_{mil['name']}_r{r}"
        names.append(name)
        run_config = {**config, "params": params, "round": r, "run_seed": seed}
        if reusable_run(name, run_config, checksum, ("model.pt", "metrics.json", "predictions.csv")):
            continue
        recorder = RunRecorder(name, config=run_config, split_checksum=checksum, overwrite=True,
                               data_version={"primary": config["datasets"]["primary"]["version"]})
        best = train_mil.fit(store, folds, r, params, config, seed, log=recorder.log_epoch)
        torch.save(best["state"], recorder.run_dir / "model.pt")
        recorder.save_predictions(prediction_table(folds, best["rows"], best["probs"], r, "val", classes))
        if best["weights"][0] is not None:
            np.savez_compressed(recorder.run_dir / "attention_val.npz",
                                **{folds.photo_id.iloc[row]: w for row, w in zip(best["rows"], best["weights"])})
        recorder.save_metrics({"best_epoch": best["epoch"], "val_macro_f1": best["f1"], "params": params,
                               "partition": "val"})
        recorder.finish()
    return names


def baselines() -> None:
    for path in ("configs/shortcut.yaml", "configs/session/shortcut.yaml"):
        shortcut_baselines.run(load_config(path))


def session_cnn() -> None:
    for model in MODELS:
        lr = core_config(f"{model}_224")["lr"]
        config = load_config(f"configs/session/{model}_224.yaml")
        for r in ROUNDS:
            stage5.run_one(config, lr, r)


def session_heads() -> None:
    for size in SIZES:
        config = load_config(f"configs/session/probe_e1_{size}.yaml")
        strength = core_config(f"probe_{size}")["l2_strength"]
        config["probe"] = {**config["probe"], "l2_strengths": [strength]}
        linear_probe.run(config)
    for name in AGGREGATORS:
        params = read_json(get_default_runs_dir() / f"stage7_tune_{name}" / "metrics.json")["best_params"]
        config = load_config(f"configs/session/{name}.yaml")
        fit_rounds(config, params, config["seed"])


def extend_cnn() -> None:
    searched = {}
    for model in MODELS:
        config = load_config(f"configs/wholeimage/{model}_224.yaml")
        declared = config["wholeimage"]["lrs"]

        def mean_score(lr: float) -> float:
            return float(np.mean([stage5.run_one(config, lr, r)[1] for r in ROUNDS]))

        scores = {lr: mean_score(lr) for lr in declared}   # the 60 core runs are reused, not retrained
        while (lr := next_value(scores, LR_LADDER, declared)) is not None:
            scores[lr] = mean_score(lr)
        searched[model] = {f"{lr:g}": score for lr, score in sorted(scores.items())}
    write_json("followup_extend_cnn.json", searched)


def extend_probes() -> None:
    for size in SIZES:
        config = load_config(f"configs/mil/probe_e1_{size}.yaml")
        declared = config["probe"]["l2_strengths"]
        core = read_json(get_default_runs_dir() / f"stage7a_selection_e1_{size}.json")
        scores = {trial["l2_strength"]: trial["mean_val_macro_f1"] for trial in core["trials"]}
        while (strength := next_value(scores, L2_LADDER, declared)) is not None:
            extra = {**config, "probe": {**config["probe"], "name": f"e1_{size}_ext", "l2_strengths": [strength]}}
            scores[strength] = linear_probe.run(extra)["trials"][0]["mean_val_macro_f1"]
        write_json(f"followup_extend_probe_{size}.json",
                   {f"{s:g}": score for s, score in sorted(scores.items())})


def study(config_name: str) -> None:
    config = load_config(f"configs/mil/{config_name}.yaml")
    if "whole_bag" in config:
        wholephoto_bags.build(config)
    train_mil.tune(config)
    train_mil.final(config, [config["seed"]])


def seeds() -> None:
    for name in AGGREGATORS:
        config = load_config(f"configs/mil/{name}.yaml")
        train_mil.final(config, [config["seed"], *config["mil"]["stability_seeds"]])


def completed(name: str, checkpoint: str) -> bool:
    return completed_run(name, ("metrics.json", "predictions.csv", checkpoint))


def select() -> dict[str, Any]:
    """Every choice below reads validation scores only."""
    runs, core = get_default_runs_dir(), core_methods()
    methods: dict[str, Any] = {}
    cnn = read_json(runs / "followup_extend_cnn.json")
    for model in MODELS:
        scores = {float(lr): score for lr, score in cnn[model].items()}
        chosen = best_value(scores)
        names = [run_name(load_config(f"configs/wholeimage/{model}_224.yaml"), chosen, r) for r in ROUNDS]
        methods[f"{model}_224"] = {"searched": cnn[model], "selected": f"{chosen:g}", "runs": names,
                                   "changed": names != core[f"{model}_224"]}
    for size in SIZES:
        searched = read_json(runs / f"followup_extend_probe_{size}.json")
        scores = {float(s): score for s, score in searched.items()}
        chosen = best_value(scores)
        declared = load_config(f"configs/mil/probe_e1_{size}.yaml")["probe"]["l2_strengths"]
        stem = f"e1_{size}" if chosen in declared else f"e1_{size}_ext"
        names = [f"stage7a_probe_{stem}_l2{chosen:g}_r{r}" for r in ROUNDS]
        methods[f"probe_{size}"] = {"searched": searched, "selected": f"{chosen:g}", "runs": names,
                                    "changed": names != core[f"probe_{size}"]}
    for name in AGGREGATORS:
        studies = {kind: read_json(runs / f"stage7_tune_{name}{suffix}" / "metrics.json")
                   for kind, suffix in (("declared", ""), ("wide", "_wide"))}
        scores = {kind: s["best_mean_val_macro_f1"] for kind, s in studies.items()}
        kind = "wide" if scores["wide"] > scores["declared"] else "declared"
        stem = f"{name}_wide" if kind == "wide" else name
        methods[name] = {"searched": scores, "selected": kind, "params": studies[kind]["best_params"],
                         "runs": [f"stage7_{stem}_r{r}" for r in ROUNDS], "changed": kind == "wide"}
    for size in SIZES:
        methods[f"whole{size}"] = {"runs": [f"stage7_whole{size}_r{r}" for r in ROUNDS], "changed": True}
    session = {}
    for model in MODELS:
        lr = core_config(f"{model}_224")["lr"]
        config = load_config(f"configs/session/{model}_224.yaml")
        session[f"{model}_224"] = [run_name(config, lr, r) for r in ROUNDS]
    for size in SIZES:
        strength = core_config(f"probe_{size}")["l2_strength"]
        session[f"probe_{size}"] = [f"stage7a_probe_e1_{size}_session_l2{strength:g}_r{r}" for r in ROUNDS]
    for name in AGGREGATORS:
        session[name] = [f"stage7_{name}_session_r{r}" for r in ROUNDS]
    for names in [m["runs"] for m in methods.values()] + list(session.values()):
        for run in names:
            if not completed(run, evaluate.checkpoint(get_default_runs_dir() / run).name):
                raise RuntimeError(f"{run}: not complete; finish every training action before selecting")
    selection = {"partition": "val", "code_version": require_clean_code(), "methods": methods, "session": session}
    write_json("followup_selection.json", selection)
    return selection


def evaluate_new() -> None:
    selection = read_json(get_default_runs_dir() / "followup_selection.json")
    names = [run for runs in selection["session"].values() for run in runs]
    names += [run for method in selection["methods"].values() if method["changed"] for run in method["runs"]]
    plan = get_default_runs_dir() / "followup_evaluation_plan.json"
    evaluate.freeze_plan(names, plan)
    for name in names:
        folder = get_default_runs_dir() / name
        if (folder / "test_metrics.json").exists() and (folder / "test_predictions.csv").exists():
            continue
        evaluate.evaluate(name, plan)
        print(f"{name}: test evaluated", flush=True)   # scores are read by src.followup_report only


ACTIONS = {"baselines": baselines, "session-cnn": session_cnn, "session-heads": session_heads,
           "extend-cnn": extend_cnn, "extend-probes": extend_probes, "seeds": seeds, "select": select,
           "evaluate": evaluate_new}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=[*ACTIONS, "wide", "whole"])
    parser.add_argument("name", nargs="?", help="aggregator for 'wide'; 224 or 512 for 'whole'")
    args = parser.parse_args()
    if args.action == "wide":
        if args.name not in AGGREGATORS:
            parser.error("wide needs mean, max or attention")
        study(f"{args.name}_wide")
    elif args.action == "whole":
        if args.name not in ("224", "512"):
            parser.error("whole needs 224 or 512")
        study(f"whole{args.name}")
    else:
        ACTIONS[args.action]()
    print(f"FOLLOWUP {args.action} {args.name or ''} DONE", flush=True)


if __name__ == "__main__":
    main()
