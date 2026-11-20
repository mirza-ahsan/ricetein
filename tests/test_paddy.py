"""Checks for src/paddy_embed.py and src/paddy_zero_shot.py (no GPU, no photos, no model files)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import embed, paddy_embed, paddy_zero_shot as pz
from src.mil_models import MILModel

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]


def test_otsu_is_applied_before_the_photo_is_padded_to_a_square(monkeypatch):
    seen = {}
    monkeypatch.setattr(paddy_embed, "load_photo", lambda path, side: np.full((120, 80, 3), 90, np.uint8))

    def clean(rgb):
        seen["shape"] = rgb.shape
        return rgb

    monkeypatch.setattr(paddy_embed.otsu, "clean", clean)
    out = paddy_embed.prepare_photo(("x.jpg", 64, True))
    assert seen["shape"] == (120, 80, 3)        # cleaned as photographed, not as a padded square
    assert out.shape == (64, 64, 3)


@pytest.mark.parametrize("aggregator", ["mean", "max", "attention"])
def test_batched_tile_scoring_equals_scoring_each_photo_alone(tmp_path, monkeypatch, aggregator):
    torch.manual_seed(0)
    dim, lengths = 6, [3, 7, 5]
    folder = tmp_path / "cache" / "tiles"
    rng = np.random.default_rng(0)
    for i, n in enumerate(lengths):
        embed.save(folder / "v0" / f"p{i}.npz", torch.from_numpy(rng.normal(size=(n, dim))).float(),
                   torch.from_numpy(rng.normal(size=(n, dim))).float(), np.zeros((n, 4), np.int32))
    model = MILModel(2 * dim, 8, 0.0, aggregator, 4).eval()
    config = {"paddy": {"caches": {"tiles": {"native": "tiles"}}, "encoder": "E1"}, "encoders": {"E1": {"dim": dim}}}
    monkeypatch.setattr(pz, "cache_folder", lambda name: folder)
    monkeypatch.setattr(pz, "photo_ids", lambda c: [f"p{i}" for i in range(len(lengths))])
    monkeypatch.setattr(pz, "photo_key", lambda photo_id: photo_id)
    monkeypatch.setattr(pz, "load_head", lambda run, d, device: (model, ["cls", "mean"]))
    batched = pz.predict_tile_heads(config, "native", ["run"], torch.device("cpu"), batch=3)["run"]
    for i, n in enumerate(lengths):
        with np.load(folder / "v0" / f"p{i}.npz") as saved:
            x = torch.cat([torch.from_numpy(saved["cls"].astype(np.float32)),
                           torch.from_numpy(saved["mean"].astype(np.float32))], -1)[None]
        alone = torch.softmax(model(x, torch.ones(1, n, dtype=torch.bool))[0], 1).detach().numpy()[0]
        np.testing.assert_allclose(batched[i], alone, atol=1e-5)


def frame(true, pred, probs=None):
    return pd.DataFrame({"photo_id": [f"p{i}" for i in range(len(true))], "true_class": true, "pred_class": pred})


def test_the_committee_picks_the_class_with_the_highest_average_probability():
    f = pd.DataFrame({"photo_id": ["a"], "true_class": ["hispa"], "pred_class": ["hispa"]})
    for c, (p, q) in zip(CLASSES, [(0.9, 0.0), (0.0, 0.0), (0.0, 0.0), (0.1, 0.4)]):
        f[f"prob_{c}"] = p
    g = f.copy()
    for c, p in zip(CLASSES, [0.0, 0.0, 0.0, 1.0]):
        g[f"prob_{c}"] = p
    assert pz.committee([f, g], CLASSES)["pred_class"].iloc[0] == "hispa"    # (0.1 + 1.0) / 2 beats 0.45


def test_a_run_is_scored_once_and_a_second_attempt_is_refused(tmp_path, monkeypatch):
    (tmp_path / "run1").mkdir()
    (tmp_path / "plan.md").write_text("plan")
    monkeypatch.setattr(pz, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(pz, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pz, "PLAN_FILES", ("plan.md",))
    monkeypatch.setattr(pz, "require_clean_code", lambda: "abc1234")
    monkeypatch.setattr(pz, "truth", lambda c: pd.DataFrame({"photo_id": ["a", "b"], "class": ["healthy", "hispa"]}))
    config = {"classes": CLASSES}
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    pz.save_once(config, "run1", "native", 1, probs)
    table = pd.read_csv(tmp_path / "run1" / "paddy_predictions_native.csv")
    assert table["pred_class"].tolist() == ["healthy", "hispa"] and set(table["partition"]) == {"paddy"}
    assert json.loads((tmp_path / "run1" / "paddy_predictions_native_started.json").read_text())["code_version"] == "abc1234"
    with pytest.raises(ValueError, match="no silent retry"):
        pz.save_once(config, "run1", "native", 1, probs)
    (tmp_path / "run1" / "paddy_predictions_midway_started.json").write_text("{}")     # a crashed run keeps its marker
    with pytest.raises(ValueError, match="no silent retry"):
        pz.save_once(config, "run1", "midway", 1, probs)


def paired_config():
    return {"classes": CLASSES, "paddy": {"headline_zoom": "native", "comparison": {"min_round_wins": 4}},
            "evaluation": {"bootstrap": {"resamples": 200, "confidence": 0.95, "stratified": True}}}


def test_attention_beats_another_method_only_with_a_range_above_zero_and_four_round_wins():
    rng = np.random.default_rng(0)
    true = np.array(CLASSES)[rng.integers(0, 4, 400)]
    noisy = lambda p: np.where(rng.random(400) < p, true, np.array(CLASSES)[rng.integers(0, 4, 400)])   # noqa: E731
    good = [frame(true, noisy(0.9)) for _ in range(5)]
    weak = [frame(true, noisy(0.4)) for _ in range(5)]
    twin = [frame(true, f["pred_class"]) for f in good]
    for group in (good, weak, twin):
        for f in group:
            for c in CLASSES:
                f[f"prob_{c}"] = (f["pred_class"] == c).astype(float)
    rows = pz.paired_rows(paired_config(), {("attention", "native"): good, ("mean", "native"): weak,
                                            ("max", "native"): twin, ("mean", "matched"): weak})
    verdicts = {r["B"]: r["verdict"] for r in rows}
    assert verdicts == {"mean": "A beats B", "max": "no clear difference"}      # other zoom levels are not compared
