"""Checks for src/split.py: every photo tested once, groups never split, class shares within 2 points,
validation and test apart, reproducible; the Paddy Doctor list counts each image once."""

import numpy as np
import pandas as pd

from src import split


def synthetic(n=1200, seed=0):
    """Imbalanced classes (like the real 44/23/16/17 %) and near-duplicate groups of 1-4 photos."""
    rng = np.random.default_rng(seed)
    classes = rng.choice(["healthy", "leaf_blast", "brown_spot", "hispa"], n, p=[0.44, 0.23, 0.16, 0.17])
    rows, g = [], 0
    i = 0
    while i < n:
        size = int(rng.choice([1, 1, 1, 2, 3, 4]))
        for _ in range(min(size, n - i)):
            rows.append({"photo_id": f"P/{i:05d}.jpg", "class": classes[g % n], "group_id": f"g{g:05d}"})
            i += 1
        g += 1
    return pd.DataFrame(rows)


def test_folds_pass_every_declared_check_and_are_reproducible():
    photos = synthetic()
    table = split.make_folds(photos, n_folds=5, val_fraction=0.125, seed=2026)
    checks = split.check_folds(table, 5, 0.02)
    assert all(v for k, v in checks.items() if k != "worst class-share deviation"), checks
    shares = (table[[split.round_column(r) for r in range(1, 6)]] == "val").mean()
    assert shares.between(0.07, 0.13).all()  # about 10% of all photos are validation in each round
    again = split.make_folds(photos.sample(frac=1, random_state=1), 5, 0.125, 2026)
    assert again.equals(table)  # input order does not matter


def test_checks_catch_a_split_group_and_a_photo_tested_twice():
    table = split.make_folds(synthetic(400), 5, 0.125, 2026)
    broken = table.copy()
    two = broken[broken["group_id"].duplicated(keep=False)].groupby("group_id").head(2).index[:2]
    broken.loc[two[0], "round_1"], broken.loc[two[1], "round_1"] = "test", "train"
    broken.loc[two[0], "round_2"] = "test"
    checks = split.check_folds(broken, 5, 0.02)
    assert not checks["no group in two roles within a round"]
    assert not checks["every photo tested exactly once"]


def test_paddy_list_keeps_shared_classes_once_and_drops_overlap():
    paddy = pd.DataFrame({
        "photo_id": ["blast/1.jpg", "blast/2.jpg", "normal/3.jpg", "tungro/4.jpg", "hispa/5.jpg"],
        "class": ["leaf_blast", "leaf_blast", "healthy", None, "hispa"],
        "sha256": ["x", "x", "y", "z", "w"], "opened_ok": True})
    overlap = pd.DataFrame({"photo_b": ["hispa/5.jpg"]})
    kept = split.paddy_list(paddy, overlap)
    assert kept["photo_id"].tolist() == ["blast/1.jpg", "normal/3.jpg"]
