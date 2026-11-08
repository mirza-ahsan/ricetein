"""Generate Stage 5 and 7 tables and paired comparisons solely from frozen, completed run records."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import stats
from src.config import load_config, output_root
from src.inventory_report import md_table
from src.records import get_default_runs_dir
from src.sanity_baselines import summarise


def configurations():
    runs = get_default_runs_dir()
    stage5 = json.loads((runs / "stage5_selection.json").read_text())["configurations"]
    probes = [
        {
            "name": f"probe_{size}",
            "runs": json.loads((runs / f"stage7a_selection_e1_{size}.json").read_text())["runs"],
        }
        for size in (224, 512)
    ]
    mil = [
        {"name": aggregator, "runs": [f"stage7_{aggregator}_r{r}" for r in range(1, 6)]}
        for aggregator in ("mean", "max", "attention")
    ]
    return stage5, probes + mil


def predictions(names):
    runs = get_default_runs_dir()
    tables = []
    checksums = set()
    for name in names:
        folder = runs / name
        record = json.loads((folder / "record.json").read_text())
        if record["status"] != "completed":
            raise ValueError(f"{name}: incomplete run")
        checksums.add(record["split_checksum"])
        tables += [pd.read_csv(folder / "predictions.csv"), pd.read_csv(folder / "test_predictions.csv")]
    if len(checksums) != 1:
        raise ValueError("inconsistent split checksums")
    return pd.concat(tables, ignore_index=True)


def build(config):
    stage5, stage7 = configurations()
    tables, rows = {}, []
    for item in stage5 + stage7:
        table = predictions(item["runs"])
        tables[item["name"]] = table
        result = summarise(table, config)
        rows.append(
            {
                "method": item["name"],
                "partition": "test",
                **{key: result["test"][key]["mean"] for key in ("macro_f1", "balanced_accuracy", "accuracy")},
                "macro_f1_sd": result["test"]["macro_f1"]["std"],
                "balanced_accuracy_sd": result["test"]["balanced_accuracy"]["std"],
                "macro_f1_ci_low": result["pooled_macro_f1"]["low"],
                "macro_f1_ci_high": result["pooled_macro_f1"]["high"],
                "ece": result["ece"],
                "hispa_recall": result["pooled_test"]["per_class"]["hispa"]["recall"],
            }
        )
    frame = pd.DataFrame(rows)
    root = output_root()
    target = root / "results/tables"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "stage7_main.csv", index=False)
    comparisons = []
    for name, table in tables.items():
        if name == "attention":
            continue
        for metric, scorer in (
            ("macro_f1", stats.macro_f1(config["classes"])),
            ("balanced_accuracy", stats.balanced_accuracy),
        ):
            result = stats.compare(
                tables["attention"],
                table,
                scorer,
                config["evaluation"]["bootstrap"],
                config["evaluation"]["comparison"],
                config["seed"],
            )
            comparisons.append({"A": "attention", "B": name, "metric": metric, "partition": "test", **result})
    paired = pd.DataFrame(comparisons)
    paired.to_csv(target / "stage7_paired.csv", index=False)
    stability = []
    for aggregator in ("mean", "max", "attention"):
        for seed in (config["seed"], 1, 2):
            for r in range(1, 6):
                name = f"stage7_{aggregator}{'' if seed == config['seed'] else f'_s{seed}'}_r{r}"
                scores = json.loads((get_default_runs_dir() / name / "metrics.json").read_text())
                stability.append(
                    {
                        "aggregator": aggregator,
                        "seed": seed,
                        "round": r,
                        "partition": "val",
                        "macro_f1": scores["val_macro_f1"],
                    }
                )
    stable = pd.DataFrame(stability)
    stable.to_csv(target / "stage7_stability.csv", index=False)
    report = root / "docs/reports"
    report.mkdir(parents=True, exist_ok=True)
    display = frame.copy()
    for column in display.columns:
        if column not in ("method", "partition", "ece"):
            display[column] = display[column].map(lambda x: f"{100 * x:.1f}")
    text = (
        "# Stage 7 main comparison\n\nGenerated from recorded runs selected on validation. "
        "Scores in percent on the locked **test** folds; SD is across five rounds and CIs are "
        "class-stratified pooled out-of-fold bootstrap intervals. Stability runs are reported on "
        "**validation**, separately. Attention weights are qualitative explanations, "
        "not disease severity.\n\n"
        + md_table(display)
        + "\n\n## Paired comparisons\n\n"
        + md_table(paired)
        + "\n\n## Stability (validation)\n\n"
        + md_table(stable.groupby(["aggregator", "seed"]).macro_f1.agg(["mean", "std"]).reset_index())
        + "\n"
    )
    (report / "mil_report.md").write_text(text)
    (report / "wholeimage_report.md").write_text(
        "# Stage 5 whole-image comparison\n\n"
        "Generated from frozen validation selections; **test** scores "
        "in percent over five rounds.\n\n" + md_table(display.iloc[: len(stage5)]) + "\n"
    )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    build(load_config(args.config))
    print("Stage 5 and Stage 7 reports written")


if __name__ == "__main__":
    main()
