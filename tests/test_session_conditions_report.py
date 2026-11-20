"""Checks for src/session_conditions_report.py: condition names, and the report text from small tables (no GPU, no data)."""

import pandas as pd
import pytest

from src import followup, session_conditions, session_conditions_report as report


def test_labels_cover_the_224_baseline_and_every_declared_condition():
    assert set(report.LABELS) == {"224", *session_conditions.CONDITIONS, session_conditions.BEST_CONDITION[0]}


@pytest.mark.parametrize("model", followup.MODELS)
@pytest.mark.parametrize("condition", list(report.LABELS))
def test_split_key_returns_the_model_and_condition_for_every_run_key(model, condition):
    assert report.split_key(f"{model}_{condition}") == (model, condition)


def test_split_key_rejects_an_unknown_condition():
    with pytest.raises(ValueError):
        report.split_key("swin_tiny_300")


def _score_row(model, condition):
    row = {"model": model, "condition": condition, "ece": 0.1}
    for metric in report.fr.METRICS:
        row |= {f"{metric}_mean": 0.7, f"{metric}_sd": 0.02, f"{metric}_pooled": 0.7, f"{metric}_photo_low": 0.65,
                f"{metric}_photo_high": 0.75, f"{metric}_block_low": 0.6, f"{metric}_block_high": 0.8}
    return row


def _paired_row(direction, verdict):
    return {"model": "swin_tiny", "condition": "128", "direction": direction, "metric": "macro_f1",
            "difference": 0.01, "photo_low": 0.0, "photo_high": 0.02, "block_low": -0.01, "block_high": 0.03,
            "round_wins": 3, "rounds": 5, "verdict_photo": verdict, "verdict_block": "no clear difference"}


def test_report_lists_both_directions_names_the_448_model_and_flags_changed_verdicts():
    tables = {"scores": pd.DataFrame([_score_row("swin_tiny", "224"), _score_row("swin_tiny", "128")]),
              "paired": pd.DataFrame([_paired_row("condition beats 224 px", "A beats B"),
                                      _paired_row("224 px beats condition", "no clear difference")])}
    text = report.build_report(tables, "convnext_tiny")
    assert "swin_tiny, 128 px" in text and "swin_tiny, 224 px" in text
    assert "(convnext_tiny) is also trained at 448 px" in text
    assert "Does the condition beat" in text and "Does 224 px beat" in text
    assert "photo range 'A beats B', block range 'no clear difference'" in text
    assert "The verdict is the same under both ranges in every row." in text
