"""Checks for src/leaf_width.py: reproducible photo selection and the pre-declared decision rule."""

import numpy as np
import pandas as pd
import pytest

from src import leaf_width as lw
from src.config import load_config


def fake_inventory(n=300):
    sides = np.random.default_rng(1).integers(700, 3200, n)
    return pd.DataFrame({"photo_id": [f"C/{i:04d}.jpg" for i in range(n)], "class": "healthy",
                         "width": sides, "height": sides, "opened_ok": True})


def test_selection_is_reproducible_and_spans_the_size_range():
    inventory = fake_inventory()
    first = lw.select_photos(inventory, 30, seed=0)
    again = lw.select_photos(inventory.sample(frac=1, random_state=3), 30, seed=0)
    assert first["photo_id"].tolist() == again["photo_id"].tolist()  # input order does not matter
    assert len(first) == 30 and first["photo_id"].is_unique
    sides = inventory["width"]
    assert first["short_side"].min() <= sides.min() * 1.05 and first["short_side"].max() >= sides.max() / 1.05
    assert first["short_side"].is_monotonic_increasing  # one per band, smallest band first


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = load_config("configs/inventory.yaml")
    inventory = fake_inventory()
    inv_file, meas_file = tmp_path / "inventory.csv", tmp_path / "measurements.csv"
    inventory.to_csv(inv_file, index=False)
    config["inventory"]["files"]["primary"] = str(inv_file)  # absolute paths override the repo-relative ones
    config["pixel_scale"]["measurements_file"] = str(meas_file)
    selected = lw.select_photos(inventory, 30, config["seed"])
    return config, selected, meas_file


def write_measurements(path, selected, widths_by_measurer):
    rows = [{"dataset": "primary", "photo_id": pid, "measurer": who, "x1": 0, "y1": 0, "x2": w, "y2": 0,
             "width_px": w, "measured_at": "2026-11-04T00:00:00+00:00"}
            for who, widths in widths_by_measurer.items() for pid, w in zip(selected["photo_id"], widths)]
    pd.DataFrame(rows, columns=lw.COLUMNS).to_csv(path, index=False)


def test_incomplete_until_every_measurer_has_done_every_photo(setup):
    config, selected, meas_file = setup
    assert not lw.decide(config)["complete"]
    write_measurements(meas_file, selected.iloc[:29], {"a": [100] * 29, "b": [100] * 29})
    assert not lw.decide(config)["complete"]


def test_leaf_width_growing_with_photo_size_means_rescale(setup):
    config, selected, meas_file = setup
    widths = selected["short_side"] * 0.2  # resized photos: leaf width scales with the photo
    write_measurements(meas_file, selected, {"a": widths, "b": widths * 1.02})
    result = lw.decide(config)
    assert result["complete"] and result["rescale"] and result["spearman_rho"] > 0.9
    assert result["median_relative_gap"] == pytest.approx(0.02 / 1.01, rel=1e-6)


def test_constant_leaf_width_means_native_resolution(setup):
    config, selected, meas_file = setup
    widths = 400 + np.random.default_rng(2).normal(0, 20, len(selected))  # crops: same zoom everywhere
    write_measurements(meas_file, selected, {"a": widths, "b": widths})
    result = lw.decide(config)
    assert result["complete"] and not result["rescale"]
    assert result["median_leaf_width_px"] == pytest.approx(np.median(widths))


def test_measure_saves_each_click_pair_and_resumes(setup, monkeypatch):
    import matplotlib
    from matplotlib.figure import Figure
    from PIL import Image

    matplotlib.use("Agg")
    config, selected, meas_file = setup
    root = meas_file.parent / "photos"
    for photo_id in selected["photo_id"]:
        (root / photo_id).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(root / photo_id)
    monkeypatch.setattr(lw, "dataset_dir", lambda config, name: root)
    clicks = iter([[(10, 20), (13, 24)]] * 5 + [[]])  # five photos, then the window is closed
    monkeypatch.setattr(Figure, "ginput", lambda self, n, timeout: next(clicks))
    lw.measure(config, "ahsan")
    saved = pd.read_csv(meas_file)
    assert len(saved) == 5 and set(saved["measurer"]) == {"ahsan"}
    assert saved["width_px"].tolist() == [5.0] * 5  # hypot(3, 4)
    clicks = iter([[(0, 0), (6, 8)]] * 25)
    lw.measure(config, "ahsan")  # resumes with the 25 photos left
    saved = pd.read_csv(meas_file)
    assert len(saved) == 30 and saved["photo_id"].is_unique and saved["width_px"].iloc[-1] == 10.0
