"""Stage 9 split files and the leak counter (declared 2026-11-20, handbook §24).

Three split files, built from the photos of the session-blocked split and written once into
``splits/partition_order/``. They are used by Stage 9 only and never mixed with ``splits/folds.csv`` or
``splits/session_blocked/folds.csv``. All three are DELIBERATELY WRONG ways to test; each imitates one mistake.

B  every photo is copied 8 times (version 0 clean, 1-7 augmented) and the copies are split at random
C  photos are split at random: near-duplicate groups and capture blocks are ignored
D  every photo is both a training and a scoring photo (role ``train_and_test``)

Each file has ``photo_id, class, version, sample_id`` and the role columns ``round_1..round_5`` of the main
split files, so the loaders read them the same way. Role ``train_and_test`` is in no loader's vocabulary: D can
never be fed to a standard loader by mistake.

The leak counter that measures each mistake is in ``src.partition_report``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.config import REPO_ROOT
from src.split import round_column

FILENAMES = {"B": "condition_B_augment_then_split.csv", "C": "condition_C_no_grouping.csv",
             "D": "condition_D_train_score_all.csv"}
BOTH = "train_and_test"          # the role of every photo in condition D
SPLIT_DIR = "splits/partition_order"   # the one place that names the folder


def random_roles(units: pd.DataFrame, n_folds: int, val_fraction: float, seed: int) -> pd.DataFrame:
    """Role columns of a stratified random split of the rows (the fold rule of ``src.split``, without groups)."""
    units = units.reset_index(drop=True)
    outer = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for r, (rest, test) in enumerate(outer.split(units, units["class"]), start=1):
        inner = StratifiedShuffleSplit(1, test_size=val_fraction, random_state=seed + r)
        _, val = next(inner.split(rest, units["class"].iloc[rest]))
        role = np.full(len(units), "train", dtype=object)
        role[test], role[rest[val]] = "test", "val"
        units[round_column(r)] = role
    return units


def condition_tables(photos: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """B, C and D from a table with ``photo_id`` and ``class`` (one row per photo)."""
    po, split = config["partition_order"], config["split"]
    photos = photos[["photo_id", "class"]].sort_values("photo_id", ignore_index=True)
    one = photos.assign(version=0, sample_id=photos["photo_id"])
    copies = photos.merge(pd.DataFrame({"version": range(po["versions"])}), how="cross")
    copies["sample_id"] = copies["photo_id"] + "|v" + copies["version"].astype(str)
    seed = {c: config["split_seed"] + po["split_seed_offset"][c] for c in ("B", "C")}
    both = one.assign(**{round_column(r): BOTH for r in range(1, split["n_folds"] + 1)})
    return {"B": random_roles(copies, split["n_folds"], split["val_fraction"], seed["B"]),
            "C": random_roles(one, split["n_folds"], split["val_fraction"], seed["C"]),
            "D": both}


def lock_tables(tables: dict[str, pd.DataFrame], folder: Path) -> dict[str, Path]:
    """Write each file once. An existing file must be byte-identical: a split is never regenerated."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for condition, table in tables.items():
        path, text = folder / FILENAMES[condition], table.to_csv(index=False)
        if path.exists():
            if path.read_text() != text:
                raise ValueError(f"{path} exists with other content; split files are never regenerated")
        else:
            with path.open("x") as out:
                out.write(text)
        paths[condition] = path
    return paths


def read_condition(condition: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / SPLIT_DIR / FILENAMES[condition], dtype=str).astype({"version": int})
