"""Checks for src/otsu.py and src/sanity_baselines.py on synthetic data."""

import numpy as np
import pandas as pd

from src import otsu
from src import sanity_baselines as sb

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]


def test_otsu_finds_a_dark_leaf_on_white_and_blacks_out_the_background():
    image = np.full((200, 300, 3), 235, np.uint8)
    image[60:140, 40:260] = (40, 120, 40)  # a green "leaf" strip
    mask = otsu.leaf_mask(image)
    assert mask[100, 150] and not mask[10, 10]
    assert abs(mask.mean() - 80 * 220 / (200 * 300)) < 0.02
    assert (otsu.clean(image)[10, 10] == 0).all() and (otsu.clean(image)[100, 150] == (40, 120, 40)).all()


def test_hsv_histogram_is_normalised_and_respects_the_mask():
    image = np.zeros((10, 10, 3), np.uint8)
    image[:, 5:] = (255, 0, 0)
    full = sb.hsv_histogram(image)
    assert full.shape == (512,) and abs(full.sum() - 1) < 1e-6
    left = sb.hsv_histogram(image, np.arange(10)[None, :].repeat(10, 0) < 5)
    assert left.max() == 1.0  # only the black half is counted


def test_tile_count_follows_the_rescale():
    assert sb.tile_count(1881, 1881, 1.0, 224) == 81 and sb.tile_count(734, 734, 2.0, 224) == 49


def folds(n=400, seed=0):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"photo_id": [f"p{i}" for i in range(n)],
                          "class": rng.choice(CLASSES, n, p=[0.44, 0.23, 0.16, 0.17])})
    fold = rng.integers(1, 6, n)
    for r in range(1, 6):
        table[f"round_{r}"] = np.where(fold == r, "test", np.where(rng.random(n) < 0.12, "val", "train"))
    return table


def test_a_perfect_shortcut_is_found_and_the_majority_floor_is_chance():
    table = folds()
    shortcut = table["class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(float)[:, None]
    shortcut = shortcut + np.random.default_rng(1).normal(0, 0.05, shortcut.shape)
    found = sb.predict_round("properties_only", table, shortcut, 1, CLASSES, 0)
    test = found[found["partition"] == "test"]
    assert (test["pred_class"] == test["true_class"]).mean() > 0.95
    floor = sb.predict_round("majority", table, None, 1, CLASSES, 0)
    assert set(floor["pred_class"]) == {"healthy"}
    assert np.allclose(floor[[f"prob_{c}" for c in CLASSES]].sum(axis=1), 1)
    guess = sb.predict_round("random_prior", table, None, 1, CLASSES, 0)
    assert guess.equals(sb.predict_round("random_prior", table, None, 1, CLASSES, 0))  # seeded
