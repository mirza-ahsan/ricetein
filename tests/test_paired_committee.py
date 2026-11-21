"""Checks for src/paired_committee.py: the run check, the photo-set check, the verdict direction and the report text
(made-up predictions; no GPU, no test files)."""

import json

import pandas as pd
import pytest

from src import paired_committee as pcm

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
CONFIG = {"classes": CLASSES, "seed": 0,
          "evaluation": {"bootstrap": {"resamples": 50, "confidence": 0.95, "stratified": True},
                         "comparison": {"ci_excludes_zero": True, "min_round_wins": 4}}}


def predictions(correct_share, seed, rounds=5, photos=40):
    """``rounds`` test rounds (each photo is a test photo in one round only); ``correct_share`` of them right, the rest the next class."""
    rows = []
    for r in range(1, rounds + 1):
        for i in range(photos):
            wrong = (i * 7 + seed + r) % 10 >= correct_share * 10
            rows.append({"photo_id": f"p{r}_{i}", "round": r, "partition": "test", "true_class": CLASSES[i % 4],
                         "pred_class": CLASSES[(i % 4 + wrong) % 4]})
    return pd.DataFrame(rows)


def test_require_started_names_the_run_without_a_marker(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "test_started.json").write_text("{}")
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(pcm, "get_default_runs_dir", lambda: tmp_path)
    pcm.require_started(["a"], ["a", "b"], "plan.json")
    with pytest.raises(RuntimeError, match="b has no test_started.json"):
        pcm.require_started(["b"], ["a", "b"], "plan.json")
    with pytest.raises(RuntimeError, match="c is not in runs/plan.json"):
        pcm.require_started(["c"], ["a", "b"], "plan.json")


def test_single_runs_must_match_the_tested_condition_runs(tmp_path, monkeypatch):
    runs = [f"r{i}" for i in range(5)]
    (tmp_path / "committee_selection.json").write_text(json.dumps({"member_runs": {pcm.SINGLE: runs}}))
    (tmp_path / "session_conditions_selection.json").write_text(json.dumps({"configurations": {pcm.SINGLE: runs}}))
    monkeypatch.setattr(pcm, "get_default_runs_dir", lambda: tmp_path)
    assert pcm.single_runs() == runs
    (tmp_path / "session_conditions_selection.json").write_text(
        json.dumps({"configurations": {pcm.SINGLE: runs[::-1]}}))
    with pytest.raises(RuntimeError, match="differ"):
        pcm.single_runs()


def test_the_photo_sets_must_be_identical_in_every_round():
    a = predictions(0.9, 0)
    pcm.require_same_photos(a, predictions(0.5, 1))
    with pytest.raises(RuntimeError, match="round 3"):
        pcm.require_same_photos(a, a.drop(a[(a["round"] == 3) & (a["photo_id"] == "p3_0")].index))
    renamed = a.copy()
    renamed.loc[(renamed["round"] == 2) & (renamed["photo_id"] == "p2_5"), "photo_id"] = "other"
    with pytest.raises(RuntimeError, match="round 2"):
        pcm.require_same_photos(a, renamed)
    with pytest.raises(RuntimeError, match="different test rounds"):
        pcm.require_same_photos(a, predictions(0.9, 0, rounds=4))


def test_the_verdict_reads_in_both_directions():
    blocks = pd.Series({f"p{r}_{i}": f"b{i % 8}" for r in range(1, 6) for i in range(40)})
    better, worse = predictions(0.95, 0), predictions(0.5, 1)
    ahead = pcm.cm.paired_with_two_way(better, worse, "A", "B", blocks, CONFIG)
    row = ahead[ahead["metric"] == "macro_f1"].iloc[0]
    assert row["difference"] > 0 and row["round_wins"] == 5 and row["round_losses"] == 0
    assert row["verdict_photo"] == "A beats B"
    behind = pcm.cm.paired_with_two_way(worse, better, "A", "B", blocks, CONFIG)
    row = behind[behind["metric"] == "macro_f1"].iloc[0]
    assert row["difference"] < 0 and row["round_losses"] == 5
    assert row["verdict_photo"] == "B beats A"


def test_report_names_both_models_and_states_each_verdict():
    row = {"A": "row2_plain", "B": pcm.SINGLE, "difference": 0.024, "photo_low": 0.01, "photo_high": 0.04,
           "block_low": -0.01, "block_high": 0.06, "round_wins": 4, "round_losses": 1, "rounds": 5,
           "verdict_photo": "A beats B", "verdict_block": "no clear difference"}
    paired = pd.DataFrame([{**row, "metric": "macro_f1"}, {**row, "metric": "balanced_accuracy"}])
    scores = pd.DataFrame([{"method": name, "ece": 0.1} | {
        f"{m}_{k}": v for m in pcm.fr.METRICS for k, v in (
            ("mean", 0.8), ("sd", 0.03), ("photo_low", 0.7), ("photo_high", 0.85), ("block_low", 0.6),
            ("block_high", 0.9))} for name in pcm.NAMES.values()])
    text = pcm.build_report(paired, scores)
    assert "ConvNeXt-T 448 px alone" in text and "post-hoc" in text and "one of the two members" in text
    assert "macro_f1, photo range: committee beats ConvNeXt-T." in text
    assert "balanced_accuracy, block range: no clear difference." in text
