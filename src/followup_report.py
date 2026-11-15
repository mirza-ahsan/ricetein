"""Tables and the report of the follow-up study (declared 2026-11-14; manuals/followup_plan.md).

    uv run python -m src.followup_report

Reads saved predictions and run metrics only: it never fits or evaluates anything, so rerunning it is safe.
It refuses to run until every run in ``runs/followup_evaluation_plan.json`` has its test predictions.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src import sanity_baselines, stats
from src.config import REPO_ROOT, load_config
from src.followup import AGGREGATORS, MAX_EXTENSIONS, MODELS, ROUNDS, core_methods, read_json
from src.inventory_report import md_table
from src.records import get_default_runs_dir

TABLES = REPO_ROOT / "results/tables"
REPORT = REPO_ROOT / "docs/reports/followup_report.md"
SHORTCUTS = ("majority", "random_prior", "colour_only", "background_only", "properties_only", "camera_only",
             "time_neighbour", "background_strict")
STAGE4_SIX = SHORTCUTS[:6]   # already run on the locked split in Stage 4b, under the name stage4_<baseline>
METRICS = ("macro_f1", "balanced_accuracy")


def block_of() -> pd.Series:
    """photo_id -> capture block (the group_id column of the session-blocked fold file)."""
    table = pd.read_csv(REPO_ROOT / "splits/session_blocked/folds.csv", dtype=str)
    return table.set_index("photo_id")["group_id"]


def block_samples(blocks: np.ndarray, resamples: int, seed: int) -> list[np.ndarray]:
    """Bootstrap index sets that resample whole capture blocks with replacement."""
    rng = np.random.default_rng(seed)
    ids, inverse = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(ids))]
    return [np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))])
            for _ in range(resamples)]


def block_interval(true, pred, blocks, metric, settings, seed=0, other=None) -> dict[str, float]:
    """metric(pred), or metric(pred) - metric(other) on the same photos, with its block-bootstrap range."""
    true, pred, blocks = np.asarray(true), np.asarray(pred), np.asarray(blocks)
    other = None if other is None else np.asarray(other)

    def value(i):
        return metric(true[i], pred[i]) - (0.0 if other is None else metric(true[i], other[i]))

    draws = np.array([value(i) for i in block_samples(blocks, settings["resamples"], seed)])
    low, high = stats.interval(draws, settings["confidence"])
    return {"value": value(np.arange(len(true))), "low": low, "high": high}


def require_tested() -> None:
    root = get_default_runs_dir()
    plan = root / "followup_evaluation_plan.json"
    if not plan.exists():
        raise RuntimeError("runs/followup_evaluation_plan.json is missing: run `followup select` and `evaluate`")
    missing = [name for name in read_json(plan)["runs"] if not (root / name / "test_predictions.csv").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} planned runs have no test predictions yet, e.g. {missing[0]}")


def read_predictions(names: list[str]) -> pd.DataFrame:
    """Validation and test predictions of the five runs of one method, as scripts/deadline_core.py does."""
    root = get_default_runs_dir()
    return pd.concat([pd.read_csv(root / name / filename) for name in names
                      for filename in ("predictions.csv", "test_predictions.csv")], ignore_index=True)


def pooled_test(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["partition"] == "test"].set_index("photo_id").sort_index()


def scorers(classes: list[str]) -> dict[str, Any]:
    return {"macro_f1": stats.macro_f1(classes), "balanced_accuracy": stats.balanced_accuracy}


def verdict(low: float, wins: int, comparison: dict[str, Any]) -> str:
    """The declared rule of ``stats.compare``, with the range passed in: entirely above 0 and enough round wins."""
    ok = low > 0 if comparison["ci_excludes_zero"] else True
    return "A beats B" if ok and wins >= comparison["min_round_wins"] else "no clear difference"


def method_scores(table: pd.DataFrame, blocks: pd.Series, config: dict[str, Any]) -> dict[str, float]:
    """Per-round mean and SD, the declared photo range and the block range of one method on test."""
    summary = sanity_baselines.summarise(table, config)
    test, boot = pooled_test(table), config["evaluation"]["bootstrap"]
    row: dict[str, float] = {"ece": summary["ece"]}
    for metric, scorer in scorers(config["classes"]).items():
        photo = summary[f"pooled_{metric}"]
        block = block_interval(test["true_class"], test["pred_class"], blocks.loc[test.index], scorer, boot,
                               config["seed"])
        row.update({f"{metric}_mean": summary["test"][metric]["mean"], f"{metric}_sd": summary["test"][metric]["std"],
                    f"{metric}_pooled": photo["value"],
                    f"{metric}_photo_low": photo["low"], f"{metric}_photo_high": photo["high"],
                    f"{metric}_block_low": block["low"], f"{metric}_block_high": block["high"]})
    return row


def paired_rows(a: pd.DataFrame, others: dict[str, pd.DataFrame], blocks: pd.Series,
                config: dict[str, Any]) -> pd.DataFrame:
    """A against every other method: the declared paired rule and the same rule with the block range."""
    boot, comparison = config["evaluation"]["bootstrap"], config["evaluation"]["comparison"]
    test_a, rows, metric_scorers = pooled_test(a), [], scorers(config["classes"])
    for name, b in others.items():
        test_b = pooled_test(b).loc[test_a.index]
        for metric, scorer in metric_scorers.items():
            photo = stats.compare(a, b, scorer, boot, comparison, config["seed"])
            block = block_interval(test_a["true_class"], test_a["pred_class"], blocks.loc[test_a.index], scorer,
                                   boot, config["seed"], other=test_b["pred_class"])
            rows.append({"B": name, "metric": metric, "difference": photo["difference"],
                         "photo_low": photo["low"], "photo_high": photo["high"],
                         "block_low": block["low"], "block_high": block["high"],
                         "round_wins": photo["round_wins"], "rounds": photo["rounds"],
                         "verdict_photo": photo["verdict"],
                         "verdict_block": verdict(block["low"], photo["round_wins"], comparison)})
    return pd.DataFrame(rows)


def shortcut_table() -> pd.DataFrame:
    root, rows = get_default_runs_dir(), []
    for name in SHORTCUTS:
        row: dict[str, Any] = {"baseline": name}
        folders = {"locked": f"stage4_{name}" if name in STAGE4_SIX else f"stage4b_{name}",
                   "session": f"stage4b_{name}_session"}
        for split, folder in folders.items():
            test = read_json(root / folder / "metrics.json")["test"]
            for metric in METRICS:
                row[f"{split}_{metric}_mean"], row[f"{split}_{metric}_sd"] = test[metric]["mean"], test[metric]["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def search_table(selection: dict[str, Any]) -> pd.DataFrame:
    declared = {f"{model}_224": load_config(f"configs/wholeimage/{model}_224.yaml")["wholeimage"]["lrs"]
                for model in MODELS}
    declared |= {f"probe_{size}": load_config(f"configs/mil/probe_e1_{size}.yaml")["probe"]["l2_strengths"]
                 for size in (224, 512)}
    rows = []
    for method, entry in selection["methods"].items():
        grid = declared.get(method)   # the aggregators have two studies instead of a grid
        for value, score in entry.get("searched", {}).items():
            rows.append({"method": method, "value": value, "mean_val_macro_f1": score,
                         "declared": value == "declared" if grid is None else float(value) in grid,
                         "selected": value == entry["selected"], "selection_changed": entry["changed"]})
    return pd.DataFrame(rows)


def stability_table() -> pd.DataFrame:
    root, rows = get_default_runs_dir(), []
    for name in AGGREGATORS:
        config = load_config(f"configs/mil/{name}.yaml")
        seeds = [config["seed"], *config["mil"]["stability_seeds"]]
        means = []
        for index, seed in enumerate(seeds):
            scores = [read_json(root / f"stage7_{name}{'' if index == 0 else f'_s{seed}'}_r{r}" / "metrics.json")
                      ["val_macro_f1"] for r in ROUNDS]
            rows += [{"aggregator": name, "seed": seed, "round": r, "val_macro_f1": s}
                     for r, s in enumerate(scores, 1)]
            means.append(float(np.mean(scores)))
            rows.append({"aggregator": name, "seed": seed, "round": "mean", "val_macro_f1": means[-1]})
        rows.append({"aggregator": name, "seed": "all", "round": "mean ± SD over seeds",
                     "val_macro_f1": float(np.mean(means)), "sd": float(np.std(means, ddof=1))})
    return pd.DataFrame(rows)


def percent(value: float) -> str:
    return f"{100 * value:.1f}"


def plus_minus(mean: float, sd: float) -> str:
    return f"{percent(mean)} ± {percent(sd)}"


def interval_text(low: float, high: float) -> str:
    return f"[{percent(low)}, {percent(high)}]"


def score_columns(frame: pd.DataFrame, column: str, label: str) -> dict[str, list[str]]:
    """Display columns of one score: mean ± SD, the declared photo range and the block range."""
    def texts(make) -> list[str]:
        return [make(r) for r in frame.to_dict("records")]

    return {f"{label} (mean ± SD)": texts(lambda r: plus_minus(r[f"{column}_mean"], r[f"{column}_sd"])),
            f"{label} photo range": texts(lambda r: interval_text(r[f"{column}_photo_low"], r[f"{column}_photo_high"])),
            f"{label} block range": texts(lambda r: interval_text(r[f"{column}_block_low"], r[f"{column}_block_high"]))}


def paired_text(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"A (attention) vs": frame["B"], "metric": frame["metric"],
                         "difference (points)": frame["difference"].map(percent),
                         "photo range": [interval_text(a, b) for a, b in zip(frame["photo_low"], frame["photo_high"])],
                         "block range": [interval_text(a, b) for a, b in zip(frame["block_low"], frame["block_high"])],
                         "rounds won": [f"{w}/{n}" for w, n in zip(frame["round_wins"], frame["rounds"])],
                         "verdict (photo range)": frame["verdict_photo"],
                         "verdict (block range)": frame["verdict_block"]})


def changed_verdicts(frame: pd.DataFrame) -> str:
    differing = frame[frame["verdict_photo"] != frame["verdict_block"]]
    if differing.empty:
        return "The verdict is the same under both ranges in every row."
    return "The verdict differs between the two ranges for: " + "; ".join(
        f"{r.B} ({r.metric}): photo range '{r.verdict_photo}', block range '{r.verdict_block}'"
        for r in differing.itertuples()) + "."


def build_report(tables: dict[str, pd.DataFrame]) -> str:
    shortcuts, sensitivity, final = tables["shortcuts"], tables["split_sensitivity"], tables["final"]
    shown_shortcuts = pd.DataFrame({"baseline": shortcuts["baseline"]} | {
        f"{split} {label}": [plus_minus(m, s) for m, s in zip(shortcuts[f"{split}_{metric}_mean"], shortcuts[f"{split}_{metric}_sd"])]
        for split in ("locked", "session") for metric, label in (("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced acc."))})
    shown_sensitivity = pd.DataFrame({"method": sensitivity["method"]}
                                     | score_columns(sensitivity, "locked_macro_f1", "locked macro-F1")
                                     | score_columns(sensitivity, "session_macro_f1", "session macro-F1")
                                     | {"drop (points)": sensitivity["drop_points"].map(lambda x: f"{x:.1f}")})
    search = tables["search"]
    shown_search = pd.DataFrame({"method": search["method"], "value tried": search["value"],
                                 "mean val macro-F1": search["mean_val_macro_f1"].map(percent),
                                 "in declared grid / study": search["declared"].map({True: "yes", False: "added"}),
                                 "selected": search["selected"].map({True: "yes", False: ""}),
                                 "selection changed": search["selection_changed"].map({True: "yes", False: "no"})})
    shown_final = pd.DataFrame({"method": final["method"]} | score_columns(final, "macro_f1", "macro-F1")
                               | score_columns(final, "balanced_accuracy", "balanced acc.")
                               | {"ECE": final["ece"].map(lambda x: f"{x:.3f}")})
    stability = tables["stability"].copy()
    stability["val macro-F1"] = [plus_minus(v, s) if pd.notna(s) else percent(v)
                                 for v, s in zip(stability["val_macro_f1"], stability["sd"])]
    shown_stability = stability[["aggregator", "seed", "round", "val macro-F1"]]
    return f"""# Follow-up study: split sensitivity, extended searches and controls

Built by `src/followup_report.py` from `runs/` and `splits/session_blocked/folds.csv`; do not edit by hand.
Study plan: `manuals/followup_plan.md`. All scores are percentages.

**Which photos each number comes from**

- Sections 3 and 7 are **validation** scores (used for selection only).
- Sections 4 and 5 are **test** scores of the **locked split** (`splits/folds.csv`).
- Section 6 is **test** scores of the **session-blocked split** (`splits/session_blocked/folds.csv`).
- Sections 1 and 2 show both splits side by side (test).
- "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks (photos
  shot without a pause of more than 60 s from one camera) and is wider when neighbouring photos are alike.

**Limits to read first**

- The wider searches (edge rule, second aggregator study) and the same-head whole-photo control were declared
  **after** the core test results were seen (2026-11-14). Their selections use validation scores only.
- The session split retrains the nine core configurations with the settings chosen on locked-split validation
  photos. Some of those photos are session-split test photos, so these scores are a sensitivity analysis, not a
  clean new tuning round.
- Class shares on the session split are reported, not enforced (`docs/reports/session_split_report.md`).
- Tuning budgets are not equal in section 5: each aggregator keeps the better of two 40-trial studies, the
  whole-photo controls had one 40-trial study, and a CNN may have up to {MAX_EXTENSIONS} added learning rates.
  Differences in section 5 therefore include a difference in search effort.
- The photo range is stratified by class; the block range resamples whole blocks and is not, so it also
  contains class-count variation. That is part of why it is wider.

## 1. Shortcut controls (test, mean ± SD over 5 rounds)

{md_table(shown_shortcuts)}

## 2. Locked split against session-blocked split (test macro-F1, nine core configurations)

{md_table(shown_sensitivity)}

## 3. Searches (validation macro-F1, mean over 5 rounds)

{md_table(shown_search)}

## 4. Final selections on the locked split (test)

Core runs where the selection is unchanged, new runs where it changed; plus the two whole-photo controls.

{md_table(shown_final)}

## 5. Attention against every other final method (locked split, test)

{md_table(paired_text(tables["paired"]))}

{changed_verdicts(tables["paired"])}

## 6. Attention against the other eight configurations (session-blocked split, test)

{md_table(paired_text(tables["paired_session"]))}

{changed_verdicts(tables["paired_session"])}

## 7. Stability seeds (validation macro-F1)

Seed 0 is the declared run; seeds 1 and 2 are the declared stability seeds. Validation only.

{md_table(shown_stability)}
"""


def main() -> None:
    require_tested()
    config, root = load_config("configs/pipeline.yaml"), get_default_runs_dir()
    selection, blocks = read_json(root / "followup_selection.json"), block_of()
    TABLES.mkdir(parents=True, exist_ok=True)
    predictions: dict[tuple[str, ...], pd.DataFrame] = {}
    scored: dict[tuple[str, ...], dict[str, float]] = {}

    def table(names: list[str]) -> pd.DataFrame:
        key = tuple(names)
        if key not in predictions:
            predictions[key] = read_predictions(names)
        return predictions[key]

    def scores(names: list[str]) -> dict[str, float]:
        key = tuple(names)
        if key not in scored:
            scored[key] = method_scores(table(names), blocks, config)
        return scored[key]

    shortcuts, stability = shortcut_table(), stability_table()   # cheap and most likely to fail: do them first
    core = core_methods()
    sensitivity = pd.DataFrame([
        {"method": method,
         **{f"locked_{k}": v for k, v in scores(core[method]).items()},
         **{f"session_{k}": v for k, v in scores(selection["session"][method]).items()}} for method in core])
    sensitivity["drop_points"] = 100 * (sensitivity["locked_macro_f1_mean"] - sensitivity["session_macro_f1_mean"])
    search = search_table(selection)
    final = pd.DataFrame([{"method": method, **scores(entry["runs"])} for method, entry in selection["methods"].items()])
    runs_final = {method: table(entry["runs"]) for method, entry in selection["methods"].items()}
    paired = paired_rows(runs_final["attention"], {m: t for m, t in runs_final.items() if m != "attention"},
                         blocks, config)
    paired_session = paired_rows(table(selection["session"]["attention"]),
                                 {m: table(runs) for m, runs in selection["session"].items() if m != "attention"},
                                 blocks, config)
    tables = {"shortcuts": shortcuts, "split_sensitivity": sensitivity, "search": search, "final": final,
              "paired": paired, "paired_session": paired_session, "stability": stability}
    for name, frame in tables.items():
        frame.to_csv(TABLES / f"followup_{name}.csv", index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(tables), encoding="utf-8")
    print(json.dumps({name: len(frame) for name, frame in tables.items()}))


if __name__ == "__main__":
    main()
