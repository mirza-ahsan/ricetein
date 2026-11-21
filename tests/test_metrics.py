"""Known-answer checks for the scoring toolkit (src/metrics.py, src/stats.py). Must pass before any model is
scored (Stage 4a)."""

import numpy as np
import pandas as pd
import pytest

from src import metrics, stats

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
BOOT = {"resamples": 400, "confidence": 0.95, "stratified": True}
RULE = {"ci_excludes_zero": True, "min_round_wins": 4}


def labels(counts=(44, 23, 16, 17)):
    return np.repeat(CLASSES, counts)


def test_perfect_predictions_score_100_percent():
    true = labels()
    s = metrics.scores(true, true, CLASSES)
    assert s["macro_f1"] == s["balanced_accuracy"] == s["accuracy"] == 1.0
    assert np.trace(s["confusion"]) == len(true)


def test_always_healthy_gives_25_percent_balanced_accuracy():
    true = labels()
    s = metrics.scores(true, np.full(len(true), "healthy"), CLASSES)
    assert s["balanced_accuracy"] == pytest.approx(0.25)
    assert s["accuracy"] == pytest.approx(0.44)
    # F1(healthy) = 2*0.44/(1+0.44); the other three are 0: the methodology's 15.3% for 1,488/3,355 healthy
    assert s["macro_f1"] == pytest.approx(2 * 0.44 / 1.44 / 4)
    assert s["per_class"]["hispa"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 17}


def test_confusion_rows_are_true_classes():
    s = metrics.scores(["hispa", "hispa"], ["healthy", "hispa"], CLASSES)
    assert s["confusion"][3] == [1, 0, 0, 1] and s["per_class"]["hispa"]["recall"] == 0.5


def predictions(pred_fn, rounds=5, per_round=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, rounds + 1):
        true = rng.choice(CLASSES, per_round, p=[0.44, 0.23, 0.16, 0.17])
        for i, t in enumerate(true):
            rows.append({"photo_id": f"{r}-{i}", "round": r, "partition": "test", "true_class": t,
                         "pred_class": pred_fn(t, rng)})
    return pd.DataFrame(rows)


def test_per_round_and_pooled_scores():
    good = predictions(lambda t, rng: t if rng.random() < 0.9 else "healthy")
    result = metrics.per_round(good, CLASSES)
    assert len(result["rounds"]) == 5 and 0.85 < result["summary"]["accuracy"]["mean"] < 0.95
    assert metrics.pooled(good, CLASSES)["n"] == 1000
    with pytest.raises(ValueError):
        metrics.pooled(pd.concat([good, good.head(1)]), CLASSES)


def test_bootstrap_interval_contains_the_value_and_narrows_with_more_photos():
    true = labels((440, 230, 160, 170))
    pred = np.where(np.random.default_rng(1).random(len(true)) < 0.8, true, "healthy")
    small = stats.bootstrap(true[::10], pred[::10], stats.macro_f1(CLASSES), BOOT)
    large = stats.bootstrap(true, pred, stats.macro_f1(CLASSES), BOOT)
    assert small["low"] <= small["value"] <= small["high"] and large["low"] <= large["value"] <= large["high"]
    assert large["high"] - large["low"] < small["high"] - small["low"]


def test_paired_comparison_rule_needs_interval_above_zero_and_four_round_wins():
    strong = predictions(lambda t, rng: t if rng.random() < 0.95 else "healthy", seed=3)
    weak = strong.assign(pred_class=[t if np.random.default_rng(i).random() < 0.6 else "healthy"
                                     for i, t in enumerate(strong["true_class"])])
    result = stats.compare(strong, weak, stats.macro_f1(CLASSES), BOOT, RULE)
    assert result["verdict"] == "A beats B" and result["round_wins"] == 5 and result["low"] > 0
    same = stats.compare(strong, strong, stats.macro_f1(CLASSES), BOOT, RULE)
    assert same["verdict"] == "no clear difference" and same["difference"] == 0


def test_ece_is_zero_when_confidence_matches_accuracy_and_large_when_overconfident():
    rng = np.random.default_rng(0)
    n = 4000
    true_index = rng.integers(0, 4, n)
    probs = np.full((n, 4), 0.1)
    correct = rng.random(n) < 0.7  # 70% right at confidence 0.7: perfectly calibrated
    top = np.where(correct, true_index, (true_index + 1) % 4)
    probs[np.arange(n), top] = 0.7
    assert stats.ece(probs, true_index, 15) == pytest.approx(0, abs=0.03)
    overconfident = np.full((n, 4), 0.0)
    overconfident[np.arange(n), top] = 1.0
    assert stats.ece(overconfident, true_index, 15) == pytest.approx(0.3, abs=0.03)
    assert stats.reliability(overconfident, true_index, 15)["photos"].sum() == n
