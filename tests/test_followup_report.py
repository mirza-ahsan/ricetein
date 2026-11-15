"""Checks for src/followup_report.py on synthetic predictions: block resampling and the two verdicts."""

import numpy as np
import pandas as pd
import pytest

from src import followup_report as report
from src import stats

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
SETTINGS = {"resamples": 200, "confidence": 0.95, "stratified": True}
COMPARISON = {"ci_excludes_zero": True, "min_round_wins": 4}
CONFIG = {"classes": CLASSES, "seed": 0,
          "evaluation": {"bootstrap": SETTINGS, "comparison": COMPARISON, "calibration_bins": 15}}


def test_block_samples_keep_every_drawn_block_whole():
    blocks = np.array(["a"] * 3 + ["b"] * 2 + ["c"] * 4)
    sizes = {"a": 3, "b": 2, "c": 4}
    for sample in report.block_samples(blocks, 50, seed=1):
        values, counts = np.unique(blocks[sample], return_counts=True)
        assert all(count % sizes[v] == 0 for v, count in zip(values, counts))  # whole blocks only
        for v, count in zip(values, counts):
            assert sorted(np.flatnonzero(blocks == v)) == sorted(set(sample[blocks[sample] == v]))


def test_perfect_prediction_with_one_photo_per_block_has_a_point_range():
    true = np.array(CLASSES * 5)
    out = report.block_interval(true, true, np.arange(len(true)).astype(str), stats.macro_f1(CLASSES), SETTINGS)
    assert out["value"] == out["low"] == out["high"] == 1.0


def test_a_single_block_gives_no_spread():
    rng = np.random.default_rng(0)
    true = rng.choice(CLASSES, 60)
    pred = np.where(rng.random(60) < 0.7, true, rng.choice(CLASSES, 60))
    out = report.block_interval(true, pred, np.zeros(60), stats.macro_f1(CLASSES), SETTINGS)
    assert out["low"] == out["high"] == pytest.approx(out["value"])


def test_a_difference_against_itself_is_exactly_zero():
    rng = np.random.default_rng(1)
    true = rng.choice(CLASSES, 80)
    pred = np.where(rng.random(80) < 0.6, true, rng.choice(CLASSES, 80))
    out = report.block_interval(true, pred, np.repeat(np.arange(16), 5), stats.macro_f1(CLASSES), SETTINGS,
                                other=pred)
    assert out == {"value": 0.0, "low": 0.0, "high": 0.0}


def predictions(accuracy: float, seed: int, photos: int = 200) -> pd.DataFrame:
    """Five rounds, each testing every fifth photo, with validation rows too; one prob_ column per class."""
    rng = np.random.default_rng(seed)
    true = np.array(CLASSES * (photos // 4))
    rows = []
    for r in range(1, 6):
        for partition, index in (("test", np.arange(r - 1, photos, 5)), ("val", np.arange(r % 5, photos, 5)[:20])):
            t = true[index]
            pred = np.where(rng.random(len(t)) < accuracy, t, rng.choice(CLASSES, len(t)))
            frame = pd.DataFrame({"round": r, "partition": partition, "photo_id": [f"p{i:03d}" for i in index],
                                  "true_class": t, "pred_class": pred})
            for c in CLASSES:
                frame[f"prob_{c}"] = (pred == c).astype(float)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def blocks_for(photos: int = 200) -> pd.Series:
    return pd.Series([f"b{i // 10}" for i in range(photos)], index=[f"p{i:03d}" for i in range(photos)])


def test_method_scores_report_both_ranges_around_the_pooled_value():
    scores = report.method_scores(predictions(0.9, 0), blocks_for(), CONFIG)
    for metric in report.METRICS:
        assert scores[f"{metric}_photo_low"] <= scores[f"{metric}_pooled"] <= scores[f"{metric}_photo_high"]
        assert scores[f"{metric}_block_low"] <= scores[f"{metric}_pooled"] <= scores[f"{metric}_block_high"]
    assert 0 <= scores["ece"] <= 1


def test_a_clearly_better_method_beats_a_worse_one_and_a_copy_does_not():
    good, bad = predictions(0.95, 1), predictions(0.4, 2)
    rows = report.paired_rows(good, {"bad": bad, "copy": good}, blocks_for(), CONFIG).set_index(["B", "metric"])
    assert rows.loc[("bad", "macro_f1"), "verdict_photo"] == "A beats B"
    assert rows.loc[("bad", "macro_f1"), "verdict_block"] == "A beats B"
    assert rows.loc[("bad", "macro_f1"), "round_wins"] >= 4
    assert rows.loc[("copy", "macro_f1"), "difference"] == 0
    assert rows.loc[("copy", "macro_f1"), "verdict_block"] == "no clear difference"


def test_the_block_verdict_needs_enough_round_wins_as_well_as_a_positive_range():
    assert report.verdict(0.01, 4, COMPARISON) == "A beats B"
    assert report.verdict(0.01, 3, COMPARISON) == "no clear difference"
    assert report.verdict(-0.01, 5, COMPARISON) == "no clear difference"
    assert report.verdict(0.0, 5, COMPARISON) == "no clear difference"


def test_report_refuses_to_run_before_every_planned_run_has_test_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "get_default_runs_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="evaluation_plan"):
        report.require_tested()
    (tmp_path / "followup_evaluation_plan.json").write_text('{"runs": {"run_a": {}, "run_b": {}}}')
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "test_predictions.csv").write_text("")
    with pytest.raises(RuntimeError, match="run_b"):
        report.require_tested()
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_b" / "test_predictions.csv").write_text("")
    report.require_tested()


def test_changed_verdicts_names_the_rows_that_differ():
    same = pd.DataFrame({"B": ["x"], "metric": ["macro_f1"], "verdict_photo": ["A beats B"], "verdict_block": ["A beats B"]})
    assert "same under both" in report.changed_verdicts(same)
    different = same.assign(verdict_block="no clear difference")
    assert "x (macro_f1)" in report.changed_verdicts(different)


def test_build_report_formats_every_section_in_percent_with_one_decimal():
    blocks, good, bad = blocks_for(), predictions(0.95, 1), predictions(0.6, 2)
    locked, session = report.method_scores(good, blocks, CONFIG), report.method_scores(bad, blocks, CONFIG)
    sensitivity = pd.DataFrame([{"method": "attention", **{f"locked_{k}": v for k, v in locked.items()},
                                 **{f"session_{k}": v for k, v in session.items()}}])
    sensitivity["drop_points"] = 100 * (sensitivity["locked_macro_f1_mean"] - sensitivity["session_macro_f1_mean"])
    paired = report.paired_rows(good, {"bad": bad}, blocks, CONFIG)
    shortcut_row = {f"{s}_{m}_{k}": 0.25 for s in ("locked", "session") for m in report.METRICS for k in ("mean", "sd")}
    tables = {
        "shortcuts": pd.DataFrame([{"baseline": "majority", **shortcut_row}]),
        "split_sensitivity": sensitivity,
        "search": pd.DataFrame([
            {"method": "mean", "value": "declared", "mean_val_macro_f1": 0.8, "declared": True, "selected": True,
             "selection_changed": False},
            {"method": "mean", "value": "wide", "mean_val_macro_f1": 0.7, "declared": False, "selected": False,
             "selection_changed": False}]),
        "final": pd.DataFrame([{"method": "attention", **locked}]),
        "paired": paired, "paired_session": paired,
        "stability": pd.DataFrame([{"aggregator": "mean", "seed": 0, "round": 1, "val_macro_f1": 0.8},
                                   {"aggregator": "mean", "seed": "all", "round": "mean ± SD over seeds",
                                    "val_macro_f1": 0.8, "sd": 0.01}])}
    text = report.build_report(tables)
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
        assert heading in text
    assert "25.0 ± 25.0" in text and "80.0 ± 1.0" in text and "**after** the core test results were seen" in text
    assert "NaN" not in text and "| nan |" not in text
