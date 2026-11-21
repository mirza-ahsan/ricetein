"""Checks for src/paddy_embed.py and src/paddy_zero_shot.py (no GPU, no photos, no model files)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import embed, paddy_embed
from src import paddy_zero_shot as pz
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


def test_a_class_token_only_cache_of_a_timm_encoder_is_described_validated_and_scored(tmp_path, monkeypatch):
    """E2 (ConvNeXt-T) has no patch average: its cache files hold "cls" only, and a head on "cls" scores them."""
    torch.manual_seed(0)
    dim, lengths = 6, [3, 4]
    folder = tmp_path / "cache" / "tiles"
    rng = np.random.default_rng(1)
    for i, n in enumerate(lengths):
        embed.save(folder / "v0" / f"p{i}.npz", torch.from_numpy(rng.normal(size=(n, dim))).float(), None,
                   np.zeros((n, 4), np.int32))
        with np.load(folder / "v0" / f"p{i}.npz") as saved:
            assert set(saved.files) == {"cls", "boxes"}
    model = MILModel(dim, 8, 0.0, "attention", 4).eval()
    config = {"paddy": {"caches": {"tiles": {"native": "tiles"}}, "encoder": "E2"},
              "encoders": {"E2": {"dim": dim, "library": "timm"}}}
    monkeypatch.setattr(pz, "cache_folder", lambda name: folder)
    monkeypatch.setattr(pz, "photo_ids", lambda c: ["p0", "p1"])
    monkeypatch.setattr(pz, "photo_key", lambda photo_id: photo_id)
    monkeypatch.setattr(pz, "load_head", lambda run, d, device: (model, ["cls"]))
    out = pz.predict_tile_heads(config, "native", ["run"], torch.device("cpu"), batch=2)["run"]
    assert out.shape == (2, 4) and np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
    # the cache check asks for "cls" only when the encoder is a timm one
    monkeypatch.setattr(paddy_embed, "cache_folder", lambda name: folder)
    monkeypatch.setattr(paddy_embed, "photo_ids", lambda c: ["p0", "p1"])
    monkeypatch.setattr(paddy_embed, "photo_key", lambda photo_id: photo_id)
    spec = {"format": 1}
    (folder / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(paddy_embed, "cache_spec", lambda c, s: spec)
    monkeypatch.setattr(paddy_embed, "description_caches", lambda c: {"tiles": {"kind": "tiles"}})
    assert paddy_embed.validate(config, "tiles")["descriptor_rows"] == sum(lengths)


def frame(true, pred):
    return pd.DataFrame({"photo_id": [f"p{i}" for i in range(len(true))], "true_class": true, "pred_class": pred})


def test_the_committee_picks_the_class_with_the_highest_average_probability():
    f = pd.DataFrame({"photo_id": ["a"], "true_class": ["hispa"], "pred_class": ["hispa"]})
    for c, (p, q) in zip(CLASSES, [(0.9, 0.0), (0.0, 0.0), (0.0, 0.0), (0.1, 0.4)]):
        f[f"prob_{c}"] = p
    g = f.copy()
    for c, p in zip(CLASSES, [0.0, 0.0, 0.0, 1.0]):
        g[f"prob_{c}"] = p
    assert pz.committee([f, g], CLASSES)["pred_class"].iloc[0] == "hispa"    # (0.1 + 1.0) / 2 beats 0.45


@pytest.fixture
def scoring(tmp_path, monkeypatch):
    (tmp_path / "plan.md").write_text("plan")
    for run in ("run1", "run2"):
        (tmp_path / run).mkdir()
    monkeypatch.setattr(pz, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(pz, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pz, "PLAN_FILES", ("plan.md",))
    monkeypatch.setattr(pz, "require_clean_code", lambda: "abc1234")
    return tmp_path, pd.DataFrame({"photo_id": ["a", "b"], "class": ["healthy", "hispa"]}), {"classes": CLASSES}


PROBS = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])


def test_a_run_is_scored_once_and_written_in_one_step(scoring):
    root, folds, config = scoring
    pz.save_once(config, folds, "run1", "native", 1, PROBS)
    table = pd.read_csv(root / "run1" / "paddy_predictions_native.csv")
    assert table["pred_class"].tolist() == ["healthy", "hispa"] and set(table["partition"]) == {"paddy"}
    assert json.loads((root / "run1" / "paddy_predictions_native_started.json").read_text())["code_version"] == "abc1234"
    assert not list((root / "run1").glob("*.tmp"))
    with pytest.raises(FileExistsError):                  # the marker exists: a second scoring cannot start
        pz.save_once(config, folds, "run1", "native", 1, PROBS)


def test_pending_skips_finished_runs_and_refuses_a_crashed_one(scoring):
    root, folds, config = scoring
    pz.save_once(config, folds, "run1", None, 1, PROBS)
    assert pz.pending(["run1", "run2"], None) == ["run2"]        # a rerun or a later call scores only what is new
    assert pz.pending(["run1", "run2"], "native") == ["run1", "run2"]    # zoom levels are separate
    (root / "run2" / "paddy_predictions_started.json").write_text("{}")    # run2 crashed after its marker
    with pytest.raises(ValueError, match="stop and report"):
        pz.pending(["run1", "run2"], None)


def conditions_config(tmp_path):
    return {"paddy": {"methods": {"session_selection": "runs/session_selection.json",
                                  "condition_selection": "runs/session_conditions_selection.json"}}}


def test_the_conditions_file_must_be_a_dict_of_five_runs_with_new_names(scoring):
    root, _, _ = scoring
    (root / "runs").mkdir()
    five = [f"r{i}" for i in range(5)]
    (root / "runs/session_selection.json").write_text(json.dumps({"methods": {"attention": {"runs": five}}}))
    config = conditions_config(root)
    assert list(pz.methods(config)) == ["attention"]                          # no conditions file yet
    for bad in ([{"name": "x", "runs": five}], {"attention": five}, {"swin_tiny_128": five[:3]}):
        (root / "runs/session_conditions_selection.json").write_text(json.dumps({"configurations": bad}))
        with pytest.raises(ValueError, match="expected"):
            pz.methods(config)
    (root / "runs/session_conditions_selection.json").write_text(json.dumps({"configurations": {"swin_tiny_128": five}}))
    assert list(pz.methods(config)) == ["attention", "swin_tiny_128"]


def test_an_unfinished_training_run_is_refused(scoring):
    root, _, _ = scoring
    (root / "run1" / "record.json").write_text(json.dumps({"status": "failed", "config": {}}))
    with pytest.raises(ValueError, match="did not complete"):
        pz.record("run1")


def scored_frames(rounds):
    out = []
    for r in rounds:
        f = pd.DataFrame({"round": r, "photo_id": ["a", "b"], "true_class": ["healthy", "hispa"],
                          "pred_class": ["healthy", "hispa"]})
        for c in CLASSES:
            f[f"prob_{c}"] = 0.25
        out.append(f)
    return out


def test_round_tables_are_ordered_by_round_and_methods_that_are_not_fully_scored_are_listed(scoring, monkeypatch):
    root, _, _ = scoring
    config = {"paddy": {"zoom_factors": {"native": 1.0}, "inputs": {"tile_methods": ["attention"]}}}
    monkeypatch.setattr(pz, "photo_ids", lambda c: ["a", "b"])
    monkeypatch.setattr(pz, "methods", lambda c: {"attention": [f"run{i}" for i in range(1, 6)], "max": ["x1", "x2", "x3", "x4", "x5"]})
    for i, r in zip(range(1, 6), [3, 1, 5, 2, 4]):                       # run names and rounds in a different order
        (root / f"run{i}").mkdir(exist_ok=True)
        scored_frames([r])[0].to_csv(root / f"run{i}" / "paddy_predictions_native.csv", index=False)
    frames, missing = pz.load_frames(config)
    assert [int(f["round"].iloc[0]) for f in frames[("attention", "native")]] == [1, 2, 3, 4, 5]
    assert missing == ["max"]


def test_h3_is_not_evaluable_until_every_fine_tuned_model_is_compared():
    config = {"paddy": {"headline_zoom": "native", "inputs": {"whole_image_models": ["a", "b"]}}}
    row = lambda b, v: {"B": b, "difference": 0.1, "low": 0.05, "high": 0.2, "round_wins": 5, "verdict": v}   # noqa: E731
    assert "not evaluable" in pz.h3_sentence(config, pd.DataFrame([row("a", "A beats B")]))
    assert "H3 is supported" in pz.h3_sentence(config, pd.DataFrame([row("a", "A beats B"), row("b", "A beats B")]))
    assert "not shown" in pz.h3_sentence(config, pd.DataFrame([row("a", "A beats B"), row("b", "no clear difference")]))


def paired_config():
    return {"classes": CLASSES, "paddy": {"headline_zoom": "native",
                                          "comparison": {"min_round_wins": 4, "ci_excludes_zero": True}},
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
