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

The leak counter measures the mistake: for the scored units of a round, the share whose source photo (or
near-duplicate group, or capture block) has a unit in training, and the share with a same-camera training photo
taken within 10 seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.config import REPO_ROOT
from src.session_split import capture_seconds, nearest_in_sorted
from src.split import round_column

FILENAMES = {"B": "condition_B_augment_then_split.csv", "C": "condition_C_no_grouping.csv",
             "D": "condition_D_train_score_all.csv"}
BOTH = "train_and_test"          # the role of every photo in condition D
EXPOSURE_SECONDS = 10


def sample_id(photo_id: str, version: int) -> str:
    return f"{photo_id}|v{version}"


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
    copies["sample_id"] = [sample_id(p, v) for p, v in zip(copies["photo_id"], copies["version"])]
    seed = {c: config["split_seed"] + po["split_seed_offset"][c] for c in ("B", "C")}
    both = one.copy()
    for r in range(1, split["n_folds"] + 1):
        both[round_column(r)] = BOTH
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


def read_condition(condition: str, folder: Path | None = None) -> pd.DataFrame:
    path = (folder or REPO_ROOT / "splits/partition_order") / FILENAMES[condition]
    return pd.read_csv(path, dtype=str).astype({"version": int})


def photo_keys(session_folds: pd.DataFrame, locked_folds: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Per photo (index ``photo_id``): its near-duplicate group (locked split file), its capture block
    (session split file), camera and capture time in seconds (inventory)."""
    info = inventory.set_index("photo_id")
    keys = pd.DataFrame({"group": locked_folds.set_index("photo_id")["group_id"],
                         "block": session_folds.set_index("photo_id")["group_id"],
                         "camera": info["camera"].fillna("unknown"),
                         "seconds": capture_seconds(info["taken_at"])})
    return keys.loc[session_folds["photo_id"]]


def leak_counter(units: pd.DataFrame, round_: int, keys: pd.DataFrame,
                 scored_where: np.ndarray | None = None) -> dict[str, float]:
    """Leak shares of one round. ``units`` has ``photo_id`` and the role columns; ``keys`` is ``photo_keys``.

    A scored unit is a ``test`` unit (condition D: every unit). Each share is the fraction of scored units
    whose source photo / near-duplicate group / capture block has a training unit, and the fraction with a
    same-camera training photo taken within ``EXPOSURE_SECONDS`` (a photo without capture time counts as none).
    ``scored_where`` (a boolean per unit) narrows the scored units, e.g. to the clean copies of condition B.
    """
    role = units[round_column(round_)].to_numpy()
    train, scored = np.isin(role, ("train", BOTH)), np.isin(role, ("test", BOTH))
    if scored_where is not None:
        scored &= scored_where
    info = keys.loc[units["photo_id"]]
    shares = {"scored_units": int(scored.sum())}
    for name, values in (("source_photo", units["photo_id"].to_numpy()),
                         ("near_duplicate_group", info["group"].to_numpy()),
                         ("capture_block", info["block"].to_numpy())):
        shares[name] = float(np.isin(values[scored], values[train]).mean())
    near = np.zeros(len(units), bool)
    camera, seconds = info["camera"].to_numpy(), info["seconds"].to_numpy(float)
    for cam in np.unique(camera[scored]):
        times = np.sort(seconds[train & (camera == cam) & ~np.isnan(seconds)])
        pick = np.flatnonzero(scored & (camera == cam))
        if len(times) and len(pick):
            gap = nearest_in_sorted(times, pd.Series(seconds[pick]))[1]
            near[pick] = np.nan_to_num(gap, nan=np.inf) <= EXPOSURE_SECONDS
    shares[f"same_camera_within_{EXPOSURE_SECONDS}s"] = float(near[scored].mean())
    return shares
