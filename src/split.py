"""Stage 3: the locked 5-fold split, and the Paddy Doctor test list.

Primary dataset (methodology "Partitioning"): photos in a label conflict are dropped; the rest are cut into
5 folds that are stratified by class and keep every near-duplicate group whole (StratifiedGroupKFold,
``split_seed``). In round r, fold r is the test set; from the other four folds a stratified, grouped
one-eighth is validation; the rest is training. The table is written once and never regenerated.

Paddy Doctor: the photos of the 4 shared classes with their mapped labels, each byte-identical image
counted once (Stage 1 found 78 extra copies), minus any photo confirmed as overlapping the primary dataset.

    uv run python -m src.split            # writes splits/folds.csv, splits/paddy4.csv and the split report
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import REPO_ROOT, load_config, path_for
from src.inventory import read_inventory
from src.inventory_report import md_table

ROLES = ("train", "val", "test")


def round_column(r: int) -> str:
    return f"round_{r}"


def make_folds(photos: pd.DataFrame, n_folds: int, val_fraction: float, seed: int) -> pd.DataFrame:
    """photos (photo_id, class, group_id) -> the same rows plus round_1..round_n role columns."""
    photos = photos.sort_values("photo_id", ignore_index=True)
    folds = np.empty(len(photos), dtype=int)
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(outer.split(photos, photos["class"], photos["group_id"])):
        folds[test] = k
    table = photos.assign(fold=folds + 1)
    inner_splits = round(1 / val_fraction)
    for r in range(1, n_folds + 1):
        rest = table[table["fold"] != r]
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + r)
        _, val = next(inner.split(rest, rest["class"], rest["group_id"]))
        role = pd.Series("train", index=table.index)
        role[table["fold"] == r] = "test"
        role[rest.index[val]] = "val"
        table[round_column(r)] = role
    return table


def check_folds(table: pd.DataFrame, n_folds: int, max_deviation: float) -> dict[str, Any]:
    """The four declared checks; each value is True when it holds, plus the worst class-share deviation."""
    rounds = [round_column(r) for r in range(1, n_folds + 1)]
    tested_once = bool(((table[rounds] == "test").sum(axis=1) == 1).all())
    groups_whole = all(table.groupby("group_id")[c].nunique().max() == 1 for c in rounds)
    overall = table["class"].value_counts(normalize=True)

    def deviation(members: pd.Series) -> float:
        shares = members.value_counts(normalize=True).reindex(overall.index, fill_value=0)
        return float((shares - overall).abs().max())

    worst = max(deviation(table.loc[table[c] == role, "class"]) for c in rounds for role in ROLES)
    val_test_apart = all(not ((table[c] == "val") & (table[c] == "test")).any() for c in rounds)
    return {"every photo tested exactly once": tested_once,
            "no group in two roles within a round": groups_whole,
            f"class shares within {max_deviation:.0%} points": worst <= max_deviation,
            "validation and test never overlap": val_test_apart, "worst class-share deviation": worst}


def paddy_list(paddy: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    """Paddy Doctor photos of the shared classes, one per byte-identical image (the first photo_id),
    without photos confirmed as overlapping the primary dataset."""
    shared = paddy[paddy["opened_ok"] & paddy["class"].notna()].sort_values("photo_id")
    shared = shared.drop_duplicates("sha256")
    shared = shared[~shared["photo_id"].isin(overlap["photo_b"] if len(overlap) else [])]
    return shared[["photo_id", "class", "sha256"]].reset_index(drop=True)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_folds(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """The locked split table and its SHA-256 (every run record stores the checksum)."""
    path = REPO_ROOT / config["split_files"]["folds"]
    return pd.read_csv(path, dtype=str), sha256(path)


def build(config: dict[str, Any]) -> dict[str, Any]:
    """Write the split files (refusing to replace an existing folds file) and return what the report needs."""
    files, settings = config["split_files"], config["split"]
    folds_path, paddy_path = REPO_ROOT / files["folds"], REPO_ROOT / files["paddy4"]
    if folds_path.exists():
        raise SystemExit(f"{folds_path} exists and is never regenerated (Stage 3)")
    groups = pd.read_csv(REPO_ROOT / config["duplicates"]["groups_file"], dtype=str)
    conflicts = pd.read_csv(REPO_ROOT / config["duplicates"]["conflicts_file"], dtype=str)
    kept = groups[~groups["photo_id"].isin(conflicts["photo_id"])][["photo_id", "class", "group_id"]]
    table = make_folds(kept, settings["n_folds"], settings["val_fraction"], config["split_seed"])
    checks = check_folds(table, settings["n_folds"], settings["max_class_share_deviation"])
    if not all(v for k, v in checks.items() if k != "worst class-share deviation"):
        raise SystemExit(f"split checks failed: {checks}")
    overlap = pd.read_csv(REPO_ROOT / config["duplicates"]["overlap_file"], dtype=str)
    paddy = paddy_list(read_inventory(config["inventory"]["files"]["paddy"]), overlap)
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    table.drop(columns="fold").to_csv(folds_path, index=False)
    paddy.to_csv(paddy_path, index=False)
    return {"table": table, "checks": checks, "excluded": conflicts, "paddy": paddy,
            "folds_sha256": sha256(folds_path), "paddy_sha256": sha256(paddy_path)}


def report(config: dict[str, Any], result: dict[str, Any]) -> str:
    table, settings = result["table"], config["split"]
    counts = []
    for r in range(1, settings["n_folds"] + 1):
        for role in ROLES:
            members = table[table[round_column(r)] == role]
            counts.append({"round": r, "role": role, "photos": len(members),
                           **members["class"].value_counts().reindex(config["classes"],
                                                                     fill_value=0).to_dict()})
    checks = "\n".join(f"- {name}: **{'yes' if ok is True else ('no' if ok is False else f'{ok:.2%}')}**"
                       for name, ok in result["checks"].items())
    paddy = result["paddy"]["class"].value_counts().reindex(config["classes"], fill_value=0)
    text = (
        "# Stage 3 split report\n\nBuilt by `src/split.py`; do not edit by hand.\n\n"
        f"Primary dataset: {len(table)} photos in {table['group_id'].nunique()} near-duplicate groups "
        f"({len(result['excluded'])} photos excluded as label conflicts). StratifiedGroupKFold, "
        f"{settings['n_folds']} folds, seed {config['split_seed']}; validation = a stratified, grouped "
        f"1/{round(1 / settings['val_fraction'])} of the other four folds "
        f"(seed {config['split_seed']} + round).\n\n"
        f"**`{config['split_files']['folds']}` SHA-256:** `{result['folds_sha256']}`\n\n"
        f"Checks:\n\n{checks}\n\n" + md_table(pd.DataFrame(counts)) + "\n\n"
        f"Paddy Doctor test list `{config['split_files']['paddy4']}`: {len(result['paddy'])} photos "
        f"({', '.join(f'{c} {n}' for c, n in paddy.items())}); each byte-identical image once; SHA-256 "
        f"`{result['paddy_sha256']}`.\n")
    target = path_for(config, "reports") / "split_report.md"
    target.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/split.yaml")
    config = load_config(parser.parse_args().config)
    result = build(config)
    report(config, result)
    print(f"folds: {len(result['table'])} photos, sha256 {result['folds_sha256']}; checks {result['checks']}")
    print(f"paddy4: {len(result['paddy'])} photos")


if __name__ == "__main__":
    main()
