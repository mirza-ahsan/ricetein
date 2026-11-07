"""Stage 4a: confidence ranges, paired comparisons and calibration (methodology "Reporting", "Calibration").

- 95% percentile bootstrap intervals from 2,000 class-stratified resamples (base.yaml ``evaluation``).
- Paired comparison of two methods on the same photos: the bootstrap interval of the difference, plus the
  number of rounds each wins. "A beats B" only if the interval is entirely above 0 AND A wins at least
  4 of the 5 rounds (decision rule 4); otherwise "no clear difference".
- Expected calibration error over 15 equal-width confidence bins, and the reliability table behind it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

Metric = Callable[[np.ndarray, np.ndarray], float]


def macro_f1(classes: list[str]) -> Metric:
    return lambda true, pred: float(f1_score(true, pred, labels=classes, average="macro", zero_division=0))


def balanced_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(true, pred))


def resample_indices(true: np.ndarray, resamples: int, stratified: bool, seed: int) -> list[np.ndarray]:
    """Bootstrap index sets; stratified ones keep every class's count (resampled within each class)."""
    rng = np.random.default_rng(seed)
    if not stratified:
        return [rng.integers(0, len(true), len(true)) for _ in range(resamples)]
    groups = [np.flatnonzero(true == c) for c in np.unique(true)]
    return [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(resamples)]


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def bootstrap(true: Any, pred: Any, metric: Metric, settings: dict[str, Any],
              seed: int = 0) -> dict[str, float]:
    """The metric, and its percentile bootstrap interval (``settings`` = base.yaml evaluation.bootstrap)."""
    true, pred = np.asarray(true), np.asarray(pred)
    values = np.array([metric(true[i], pred[i])
                       for i in resample_indices(true, settings["resamples"], settings["stratified"], seed)])
    low, high = interval(values, settings["confidence"])
    return {"value": metric(true, pred), "low": low, "high": high}


def paired(true: Any, pred_a: Any, pred_b: Any, metric: Metric, settings: dict[str, Any],
           seed: int = 0) -> dict[str, float]:
    """Difference metric(A) - metric(B) on the same photos, with its paired bootstrap interval."""
    true, pred_a, pred_b = np.asarray(true), np.asarray(pred_a), np.asarray(pred_b)
    diffs = np.array([metric(true[i], pred_a[i]) - metric(true[i], pred_b[i])
                      for i in resample_indices(true, settings["resamples"], settings["stratified"], seed)])
    low, high = interval(diffs, settings["confidence"])
    return {"difference": metric(true, pred_a) - metric(true, pred_b), "low": low, "high": high}


def compare(a: pd.DataFrame, b: pd.DataFrame, metric: Metric, settings: dict[str, Any],
            comparison: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Pooled out-of-fold test predictions of two methods (same photos) -> difference, interval, round wins
    and the verdict under the declared rule."""
    test_a = a[a["partition"] == "test"].set_index("photo_id").sort_index()
    test_b = b[b["partition"] == "test"].set_index("photo_id").sort_index()
    if not test_a.index.equals(test_b.index) or not (test_a["round"] == test_b["round"]).all():
        raise ValueError("the two methods must be tested on the same photos in the same rounds")
    result = paired(test_a["true_class"], test_a["pred_class"], test_b["pred_class"], metric, settings, seed)
    wins = sum(metric(g["true_class"].to_numpy(), g["pred_class"].to_numpy())
               > metric(g["true_class"].to_numpy(), test_b.loc[g.index, "pred_class"].to_numpy())
               for _, g in test_a.groupby("round"))
    ci_ok = result["low"] > 0 if comparison["ci_excludes_zero"] else True
    better = bool(ci_ok and wins >= comparison["min_round_wins"])
    return {**result, "round_wins": int(wins), "rounds": int(test_a["round"].nunique()),
            "verdict": "A beats B" if better else "no clear difference"}


def reliability(probabilities: np.ndarray, true_index: np.ndarray, bins: int) -> pd.DataFrame:
    """Per confidence bin: photos, mean confidence and accuracy of the top prediction."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == true_index
    which = np.minimum((confidence * bins).astype(int), bins - 1)  # confidence 1.0 goes to the last bin
    table = pd.DataFrame({"bin": which, "confidence": confidence, "correct": correct})
    out = table.groupby("bin").agg(photos=("correct", "size"), confidence=("confidence", "mean"),
                                   accuracy=("correct", "mean"))
    return out.reindex(range(bins)).fillna({"photos": 0}).astype({"photos": int})


def ece(probabilities: np.ndarray, true_index: np.ndarray, bins: int) -> float:
    """Expected calibration error: sum over bins of (share of photos) x |accuracy - mean confidence|."""
    table = reliability(probabilities, true_index, bins).dropna()
    weight = table["photos"] / table["photos"].sum()
    return float((weight * (table["accuracy"] - table["confidence"]).abs()).sum())
