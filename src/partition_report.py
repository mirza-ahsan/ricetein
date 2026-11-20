"""Stage 9 inflation table and report (declared 2026-11-20, handbook §24). Reads scored predictions only.

    uv run python -m src.partition_report report   # results/tables/partition_order_inflation.csv and the report
    uv run python -m src.partition_report leaks    # the leak counter of every condition, from the split files alone

A separate module from ``src.partition_order`` on purpose: ``records.code_paths`` hashes the training module and
what it imports, so a fix to the report text never makes a finished run look changed.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src import metrics, stats
from src.config import REPO_ROOT, load_config
from src.inventory import read_inventory
from src.inventory_report import md_table
from src.partition_order import (
    ROUNDS,
    all_runs,
    get_default_runs_dir,
    reference_runs,
    require_declared,
    run_name,
)
from src.partition_splits import leak_counter, photo_keys, read_condition

TABLE, REPORT = "results/tables/partition_order_inflation.csv", "docs/reports/partition_order_report.md"


MEANING = {"A": "correct: grouped by capture block, augmentation on training photos only",
           "L": "reference: locked split (duplicates grouped, capture sessions not), its own selected settings",
           "B": "augment first, then split the copies at random",
           "B_clean": "as B, scored on the clean copy of each photo only",
           "C": "no grouping: photos split at random", "D": "train on every photo, score on every photo"}


def confusion_by_photo(table: pd.DataFrame, photos: pd.Index, classes: list[str]) -> np.ndarray:
    """(photos, k*k) counts: for each photo, how many of its scored rows fall in each (true, predicted) cell."""
    k, code = len(classes), {c: i for i, c in enumerate(classes)}
    at = photos.get_indexer(table["photo_id"])
    if (at < 0).any():
        raise ValueError("scored rows of photos that are not in the reference set")
    counts = np.zeros((len(photos), k * k))
    np.add.at(counts, (at, table["true_class"].map(code).to_numpy() * k + table["pred_class"].map(code).to_numpy()), 1)
    return counts


def scores_from_confusion(counts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 (over all k classes, 0 where undefined) and balanced accuracy of confusion counts (..., k*k)."""
    c = counts.reshape(*counts.shape[:-1], k, k)
    tp, actual, predicted = np.diagonal(c, axis1=-2, axis2=-1), c.sum(-1), c.sum(-2)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros_like(tp), where=actual + predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    present = actual > 0
    return f1.mean(-1), (recall * present).sum(-1) / present.sum(-1)


def inflation_ranges(counts: dict[str, np.ndarray], photo_class: np.ndarray, settings: dict[str, Any], seed: int,
                     k: int, reference: str = "A") -> dict[str, dict[str, dict[str, float]]]:
    """Per label and metric: the pooled value with its range, and the difference to ``reference`` with its range.
    The ranges come from one set of bootstrap draws over source photos (resampled within class, all copies and
    repeats of a photo kept together), so every difference is paired."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(photo_class == c) for c in np.unique(photo_class)]
    draws = np.zeros((settings["resamples"], len(photo_class)))
    for i in range(settings["resamples"]):
        draws[i] = np.bincount(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]),
                               minlength=len(photo_class))
    point = {label: scores_from_confusion(c.sum(0), k) for label, c in counts.items()}
    boot = {label: scores_from_confusion(draws @ c, k) for label, c in counts.items()}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for label in counts:
        result[label] = {}
        for m, metric in enumerate(("macro_f1", "balanced_accuracy")):
            low, high = stats.interval(boot[label][m], settings["confidence"])
            d_low, d_high = stats.interval(boot[label][m] - boot[reference][m], settings["confidence"])
            result[label][metric] = {"value": float(point[label][m]), "low": low, "high": high,
                                     "difference": float(point[label][m] - point[reference][m]),
                                     "difference_low": d_low, "difference_high": d_high}
    return result


def round_scores(table: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Macro-F1 and balanced accuracy of each round (or repeat), indexed by round."""
    return pd.DataFrame({int(r): {m: s[m] for m in ("macro_f1", "balanced_accuracy")}
                         for r, g in table.groupby("round")
                         for s in [metrics.scores(g["true_class"], g["pred_class"], classes)]}).T


def reading(difference_low: float, difference_high: float, above: int, below: int, minimum: int) -> str:
    """The declared decision rule for 'higher than A': the range excludes 0 and at least
    ``minimum`` rounds agree. Otherwise there is no clear difference."""
    if difference_low > 0 and above >= minimum:
        return "clearly above A"
    if difference_high < 0 and below >= minimum:
        return "clearly below A"
    return "no clear difference"


def leak_shares(units: pd.DataFrame, keys: pd.DataFrame, scored_where: np.ndarray | None = None) -> dict[str, float]:
    """The leak counter averaged over the five rounds, weighted by the number of scored units."""
    rows = [leak_counter(units, r, keys, scored_where) for r in ROUNDS]
    weight = np.array([row["scored_units"] for row in rows], float)
    return {key: float(np.average([row[key] for row in rows], weights=weight)) for key in rows[0] if key != "scored_units"}


def read_scored(names: list[str], filename: str) -> pd.DataFrame:
    root = get_default_runs_dir()
    return pd.concat([pd.read_csv(root / n / filename) for n in names], ignore_index=True)


def inflation_table(config: dict[str, Any], tables: dict[str, pd.DataFrame], leaks: dict[str, dict[str, float]],
                    model: str) -> pd.DataFrame:
    """One row per condition for one model; ``tables`` maps a label to its scored predictions (A first)."""
    classes, boot = config["classes"], config["evaluation"]["bootstrap"]
    minimum = config["evaluation"]["comparison"]["min_round_wins"]
    photos = pd.Index(sorted(tables["A"]["photo_id"].unique()))
    counts = {label: confusion_by_photo(t, photos, classes) for label, t in tables.items()}
    for label, c in counts.items():
        if (c.sum(1) == 0).any():
            raise ValueError(f"{label}: some photos have no scored row; the table needs every photo in every condition")
    photo_class = tables["A"].drop_duplicates("photo_id").set_index("photo_id").loc[photos, "true_class"]
    ranges = inflation_ranges(counts, photo_class.map({c: i for i, c in enumerate(classes)}).to_numpy(), boot,
                              config["seed"], len(classes))
    per_round = {label: round_scores(t, classes) for label, t in tables.items()}
    rows = []
    for label, t in tables.items():
        row: dict[str, Any] = {"model": model, "condition": label, "meaning": MEANING[label], "scored_rows": len(t),
                               "photos": t["photo_id"].nunique(), "rounds": t["round"].nunique()}
        for metric in ("macro_f1", "balanced_accuracy"):
            mine, ref = per_round[label][metric], per_round["A"][metric]
            above, below = int((mine > ref.reindex(mine.index)).sum()), int((mine < ref.reindex(mine.index)).sum())
            r = ranges[label][metric]
            row.update({f"{metric}_pooled": r["value"], f"{metric}_low": r["low"], f"{metric}_high": r["high"],
                        f"{metric}_round_mean": mine.mean(), f"{metric}_round_sd": mine.std(ddof=1),
                        f"{metric}_inflation": r["difference"], f"{metric}_inflation_low": r["difference_low"],
                        f"{metric}_inflation_high": r["difference_high"], f"{metric}_rounds_above_A": above,
                        f"{metric}_reading": "reference" if label == "A" else
                        reading(r["difference_low"], r["difference_high"], above, below, minimum)})
        rows.append({**row, **{f"leak_{k}": v for k, v in leaks[label].items()}})
    return pd.DataFrame(rows)


def report(config: dict[str, Any]) -> None:
    require_declared(config)
    root = get_default_runs_dir()
    missing = [n for n, *_ in all_runs(config) if not (root / n / "score_metrics.json").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} runs are not scored yet, e.g. {missing[0]}")
    leaks = leak_tables(config)
    frames = []
    for model in config["partition_order"]["models"]:
        scored = {c: read_scored([run_name(model, c, r) for r in ROUNDS], "score_predictions.csv") for c in "BCD"}
        tables = {"A": read_scored(reference_runs(config, model, "a"), "test_predictions.csv"),
                  "L": read_scored(reference_runs(config, model, "l"), "test_predictions.csv"),
                  "B": scored["B"], "B_clean": scored["B"][scored["B"]["version"] == 0],
                  "C": scored["C"], "D": scored["D"]}
        frames.append(inflation_table(config, tables, leaks, model))
    table = pd.concat(frames, ignore_index=True)
    (REPO_ROOT / TABLE).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPO_ROOT / TABLE, index=False)
    (REPO_ROOT / REPORT).write_text(report_text(table))
    print(f"report: {len(table)} rows", flush=True)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def report_text(table: pd.DataFrame) -> str:
    """The neutral report, built from the table only."""
    sections = []
    for model, title in (("efficientnetv2s", "EfficientNetV2-S, whole photo at 224 px"),
                         ("attention", "Main tile model (E1 tiles + gated-attention head)")):
        part = table[table["model"] == model]
        shown = pd.DataFrame({
            "condition": part["condition"], "what it does": part["meaning"],
            "scored rows": part["scored_rows"].map("{:,}".format),
            "macro-F1, pooled [95% range]": [f"{pct(v)} [{pct(lo)}, {pct(hi)}]" for v, lo, hi in
                                             zip(part.macro_f1_pooled, part.macro_f1_low, part.macro_f1_high)],
            "mean ± SD over rounds": [f"{pct(m)} ± {pct(s)}" for m, s in zip(part.macro_f1_round_mean, part.macro_f1_round_sd)],
            "inflation vs A (points) [95% range]": ["–" if c == "A" else f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                                                    for c, d, lo, hi in zip(part.condition, part.macro_f1_inflation,
                                                                            part.macro_f1_inflation_low, part.macro_f1_inflation_high)],
            "rounds above A": ["–" if c == "A" else f"{n}/{r}" for c, n, r in zip(part.condition, part.macro_f1_rounds_above_A, part.rounds)],
            "reading": part["macro_f1_reading"]})
        leak = pd.DataFrame({
            "condition": part["condition"], "same source photo": part["leak_source_photo"].map(pct),
            "same near-duplicate group": part["leak_near_duplicate_group"].map(pct),
            "same capture block": part["leak_capture_block"].map(pct),
            "same camera within 10 s": part["leak_same_camera_within_10s"].map(pct)})
        balanced = pd.DataFrame({
            "condition": part["condition"], "balanced accuracy, pooled": part["balanced_accuracy_pooled"].map(pct),
            "inflation vs A (points) [95% range]": ["–" if c == "A" else f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                                                    for c, d, lo, hi in zip(part.condition, part.balanced_accuracy_inflation,
                                                                            part.balanced_accuracy_inflation_low, part.balanced_accuracy_inflation_high)],
            "reading": part["balanced_accuracy_reading"]})
        sections += [f"## {title}", "", "Macro-F1 (percent, test photos of each condition):", "", md_table(shown), "",
                     "Balanced accuracy:", "", md_table(balanced), "",
                     "Leak counter: share of the scored rows (all five rounds) whose photo, group, block or neighbour in time "
                     "is also in training (percent):", "", md_table(leak), ""]
    return "\n".join([
        "# Partition-order experiment", "",
        "Built by `src/partition_order.py` from `runs/`; do not edit by hand. Declared on 2026-11-20 (handbook §24).", "",
        "**What this is.** A general lesson about testing practice. Testing mistakes are common across plant-disease "
        "recognition work (Xu et al., 2024). Here we repeat three of them on our own photos and models and measure how "
        "much each raises the score, compared with the correct protocol. No published study is assessed.", "",
        "**Conditions.** A is the correct protocol: the session-blocked split, augmentation on training photos only (the "
        "five headline runs of each model, reused as they are). B makes eight copies of every photo with the same "
        "augmentation, then splits the copies at random, so copies of one photo land on both sides. C splits photos at "
        "random and ignores near-duplicate groups and capture blocks. D trains on every photo and scores on every photo. "
        "L is a reference row: the locked split, which groups pixel-level copies but not capture sessions, with the "
        "settings selected on it (not A's).", "",
        "**How to read it.** B, C and D are deliberately wrong. Their scores are not estimates of how the models "
        "perform on new photos, and they are used nowhere else. All of them reuse A's selected settings: nothing is "
        "tuned. B takes as many training steps per epoch as A. B is scored on its test copies, which include augmented "
        "copies; `B_clean` keeps only the clean copy of each photo. D stops at A's best epoch and its five repeats use "
        "five seeds on the same photos. Inflation is the pooled score of a condition minus the pooled score of A. Its "
        "range is a bootstrap over source photos (resampled within class; all copies and repeats of a photo stay "
        "together). A condition is called higher (or lower) than A only when the range excludes 0 and at least 4 of 5 "
        "rounds agree. \"Rounds above A\" pairs round r of a condition with round r of A; for D (every repeat scores all "
        "photos) and for L (another split) this is a convention, not a comparison on identical photos. The "
        "same-camera-within-10-s share counts a photo's own training copy, so it is about 100% by construction for B and D.",
        "One training seed per round and condition; the rounds are the repeats.", ""] + sections)


def leak_tables(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The leak counter of A, the locked-split reference L and B, B_clean, C and D, from the split files alone."""
    session = pd.read_csv(REPO_ROOT / config["split_files"]["folds"], dtype=str)
    locked = pd.read_csv(REPO_ROOT / "splits/folds.csv", dtype=str)
    keys = photo_keys(session, locked, read_inventory(config["inventory"]["files"]["primary"]))
    b = read_condition("B")
    return {"A": leak_shares(session, keys), "L": leak_shares(locked, keys), "B": leak_shares(b, keys),
            "B_clean": leak_shares(b, keys, (b["version"] == 0).to_numpy()),
            "C": leak_shares(read_condition("C"), keys), "D": leak_shares(read_condition("D"), keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["report", "leaks"])
    parser.add_argument("--config", default="configs/partition_order.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "leaks":
        for label, shares in leak_tables(config).items():
            print(label, {k: round(v, 4) for k, v in shares.items()}, flush=True)
    else:
        report(config)
    print(f"STAGE9 {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
