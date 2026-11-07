"""Stage 4a: every score we report, from per-photo predictions (methodology "Evaluation").

Predictions are the run-record format: one row per photo with ``true_class``, ``pred_class`` and one
``prob_<class>`` column per class (class order fixed in base.yaml), plus ``round`` and ``partition`` when a
run covers several rounds. Main scores: macro-F1 and balanced accuracy; also accuracy, per-class precision,
recall and F1, and the confusion matrix (rows = true class, columns = predicted class).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def scores(true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray, classes: list[str]) -> dict[str, Any]:
    """All scores for one set of predictions. Classes never predicted get precision 0 (and F1 0)."""
    true, pred = np.asarray(true), np.asarray(pred)
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=classes,
                                                                     zero_division=0)
    return {
        "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "per_class": {c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for c, p, r, f, s in zip(classes, precision, recall, f1, support)},
        "confusion": confusion_matrix(true, pred, labels=classes).tolist(),
        "n": int(len(true)),
    }


def probabilities(predictions: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """(n, k) array of the ``prob_<class>`` columns, in class order."""
    return predictions[[f"prob_{c}" for c in classes]].to_numpy(float)


def per_round(predictions: pd.DataFrame, classes: list[str], partition: str = "test") -> dict[str, Any]:
    """Scores of each round on ``partition``, and the mean and standard deviation of the main scores."""
    rows = predictions[predictions["partition"] == partition]
    rounds = {int(r): scores(g["true_class"], g["pred_class"], classes) for r, g in rows.groupby("round")}
    summary = {}
    for key in ("macro_f1", "balanced_accuracy", "accuracy"):
        values = np.array([s[key] for s in rounds.values()])
        spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[key] = {"mean": float(values.mean()), "std": spread}
    return {"rounds": rounds, "summary": summary}


def pooled(predictions: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    """Scores of the pooled out-of-fold test predictions: each photo once, from the round that tested it."""
    test = predictions[predictions["partition"] == "test"]
    if test["photo_id"].duplicated().any():
        raise ValueError("a photo appears in more than one test fold")
    return scores(test["true_class"], test["pred_class"], classes)
