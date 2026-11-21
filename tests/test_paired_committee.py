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


BLOCKS = pd.Series({f"p{r}_{i}": f"b{i % 8}" for r in range(1, 6) for i in range(40)})


def predictions(correct_share, seed, rounds=5, photos=40):
    """``rounds`` test rounds (each photo is a test photo in one round only); ``correct_share`` of them right, the rest the next class."""
    rows = []
    for r in range(1, rounds + 1):
        for i in range(photos):
            wrong = (i * 7 + seed + r) % 10 >= correct_share * 10
            rows.append({"photo_id": f"p{r}_{i}", "round": r, "partition": "test", "true_class": CLASSES[i % 4],
                         "pred_class": CLASSES[(i % 4 + wrong) % 4]})
    return pd.DataFrame(rows)


def test_require_started_checks_the_marker_and_the_plan_hash(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    digest = pcm.sha256(plan)
    for name, marker in (("a", {"plan_sha256": digest}), ("old", {"plan_sha256": "0" * 64}), ("b", None)):
        (tmp_path / name).mkdir()
        if marker is not None:
            (tmp_path / name / "test_started.json").write_text(json.dumps(marker))
    monkeypatch.setattr(pcm, "get_default_runs_dir", lambda: tmp_path)
    pcm.require_started(["a"], "plan.json", ["a", "b"])
    with pytest.raises(RuntimeError, match="b has no test_started.json"):
        pcm.require_started(["b"], "plan.json", ["a", "b"])
    with pytest.raises(RuntimeError, match="old: test_started.json was written under another version"):
        pcm.require_started(["old"], "plan.json", ["old"])
    with pytest.raises(RuntimeError, match="c is not in runs/plan.json"):
        pcm.require_started(["c"], "plan.json", ["a", "b"])


def test_the_committee_folder_must_hold_the_declared_committee(tmp_path, monkeypatch):
    declared = {"members": ["m", "n"], "weights": [1, 1], "runs": {"m": ["r1"], "n": ["r2"]}}
    (tmp_path / f"committee_{pcm.COMMITTEE}").mkdir()
    folder_file = tmp_path / f"committee_{pcm.COMMITTEE}" / "committee.json"
    monkeypatch.setattr(pcm.cm, "get_default_runs_dir", lambda: tmp_path)
    folder_file.write_text(json.dumps({"id": "x", **declared}))
    pcm.require_declared_committee({"committees": {pcm.COMMITTEE: declared}})
    folder_file.write_text(json.dumps({"id": "x", **declared, "weights": [1, 2]}))
    with pytest.raises(RuntimeError, match="differs"):
        pcm.require_declared_committee({"committees": {pcm.COMMITTEE: declared}})


def test_the_single_models_test_files_must_match_the_frozen_hashes(tmp_path, monkeypatch):
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "test_predictions.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(pcm, "get_default_runs_dir", lambda: tmp_path)
    good = pcm.sha256(tmp_path / "r1" / "test_predictions.csv")
    pcm.require_unchanged(["r1"], {"tested_members_test_sha256": {"r1": good}})
    with pytest.raises(RuntimeError, match="r1: test predictions changed"):
        pcm.require_unchanged(["r1"], {"tested_members_test_sha256": {"r1": "0" * 64}})


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
    better, worse = predictions(0.95, 0), predictions(0.5, 1)
    ahead = pcm.cm.paired_with_two_way(better, worse, "A", "B", BLOCKS, CONFIG)
    row = ahead[ahead["metric"] == "macro_f1"].iloc[0]
    assert row["difference"] > 0 and row["round_wins"] == 5 and row["round_losses"] == 0
    assert row["verdict_photo"] == "A beats B"
    behind = pcm.cm.paired_with_two_way(worse, better, "A", "B", BLOCKS, CONFIG)
    row = behind[behind["metric"] == "macro_f1"].iloc[0]
    assert row["difference"] < 0 and row["round_losses"] == 5
    assert row["verdict_photo"] == "B beats A"


def test_a_gain_with_too_few_round_wins_is_no_clear_difference():
    """The range is above 0 but the committee wins only 3 rounds (two rounds tied): neither side wins."""
    single = predictions(0.7, 1)
    better = single.copy()
    for r in (1, 2, 3):                       # the committee fixes five photos in rounds 1 to 3 only
        fixed = better[(better["round"] == r) & (better["pred_class"] != better["true_class"])].head(5).index
        better.loc[fixed, "pred_class"] = better.loc[fixed, "true_class"]
    paired = pcm.cm.paired_with_two_way(better, single, "A", "B", BLOCKS, CONFIG)
    row = paired[paired["metric"] == "macro_f1"].iloc[0]
    assert row["round_wins"] == 3 and row["round_losses"] == 0
    assert row["verdict_photo"] == "no clear difference" and row["verdict_block"] == "no clear difference"


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
