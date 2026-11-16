"""Checks for src/session_report.py on made-up tables (no runs, no bootstrap)."""

import json

import pandas as pd
import pytest

from src import session_report as report

SCORE = {f"{metric}_{part}": 0.7 for metric in ("macro_f1", "balanced_accuracy")
         for part in ("mean", "pooled", "photo_low", "photo_high", "block_low", "block_high")}
SCORE |= {"macro_f1_sd": 0.02, "balanced_accuracy_sd": 0.03, "ece": 0.05}


def final_table() -> pd.DataFrame:
    return pd.DataFrame([{"method": "attention", **SCORE}, {"method": "whole512", **SCORE, "macro_f1_mean": 0.6}])


def side_tables():
    locked = pd.DataFrame({"method": ["attention", "whole512"], "macro_f1_mean": [0.828, 0.818],
                           "macro_f1_sd": [0.015, 0.018]})
    refit = pd.DataFrame({"method": ["attention"], "session_macro_f1_mean": [0.757], "session_macro_f1_sd": [0.03]})
    return locked, refit


def test_side_by_side_takes_each_column_from_its_own_table_and_leaves_gaps_empty():
    sides = report.side_by_side(final_table(), *side_tables()).set_index("method")
    assert sides.loc["attention", "locked_mean"] == 0.828 and sides.loc["attention", "refit_mean"] == 0.757
    assert sides.loc["attention", "tuned_mean"] == 0.7 and sides.loc["whole512", "tuned_mean"] == 0.6
    assert pd.isna(sides.loc["whole512", "refit_mean"])   # the whole-photo controls were never refitted


def test_build_report_has_every_section_and_no_missing_values():
    final = final_table()
    tables = {
        "search": pd.DataFrame([
            {"method": "attention", "value": "declared", "mean_val_macro_f1": 0.8, "declared": True, "selected": True,
             "selection_changed": True},
            {"method": "swin_tiny_224", "value": "1e-05", "mean_val_macro_f1": 0.7, "declared": False,
             "selected": False, "selection_changed": False}]),
        "final": final,
        "paired": pd.DataFrame([{"B": "whole512", "metric": "macro_f1", "difference": 0.1, "photo_low": 0.05,
                                 "photo_high": 0.15, "block_low": -0.01, "block_high": 0.2, "round_wins": 5,
                                 "rounds": 5, "verdict_photo": "A beats B", "verdict_block": "no clear difference"}]),
        "side_by_side": report.side_by_side(final, *side_tables()),
        "shortcuts": pd.DataFrame([{"baseline": "time_neighbour", "session_macro_f1_mean": 0.304,
                                    "session_macro_f1_sd": 0.108, "session_balanced_accuracy_mean": 0.31,
                                    "session_balanced_accuracy_sd": 0.1}]),
        "stability": pd.DataFrame([{"aggregator": "attention", "seed": 0, "round": 1, "val_macro_f1": 0.8},
                                   {"aggregator": "attention", "seed": "all", "round": "mean ± SD over seeds",
                                    "val_macro_f1": 0.8, "sd": 0.01}])}
    text = report.build_report(tables)
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert heading in text
    assert "70.0 ± 2.0" in text and "30.4 ± 10.8" in text and "82.8 ± 1.5" in text and "75.7 ± 3.0" in text
    assert "| — |" in text                                  # whole512 has no refitted column
    assert "whole512 (macro_f1)" in text                     # the row whose two verdicts differ is named
    assert "scored once" in text and "validation scores only" in text
    assert "NaN" not in text and "| nan |" not in text


def test_the_report_refuses_a_selection_that_does_not_match_the_frozen_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(report.fr, "get_default_runs_dir", lambda: tmp_path)
    (tmp_path / report.PLAN).write_text(json.dumps({"runs": {"run_a": {}}}))
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "test_predictions.csv").write_text("")
    (tmp_path / "session_selection.json").write_text(json.dumps({"methods": {"attention": {"runs": ["run_b"]}}}))
    with pytest.raises(RuntimeError, match="name different runs"):
        report.main()
