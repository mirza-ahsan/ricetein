"""Checks for src/paired_conditions.py: verdict rule, run order, the run check and the report text (no GPU, no data)."""

import json

import pandas as pd
import pytest

from src import paired_conditions as pc


@pytest.mark.parametrize("ahead, behind, expected", [
    ("A beats B", "no clear difference", pc.ATTENTION),
    ("no clear difference", "A beats B", pc.OTHER),
    ("no clear difference", "no clear difference", pc.NONE)])
def test_overall_verdict_follows_the_direction_that_passes(ahead, behind, expected):
    assert pc.overall(ahead, behind) == expected


def test_the_448_condition_comes_first_and_the_rest_keep_their_order():
    keys = ["swin_tiny_128", "convnext_tiny_448", "inceptionv3_224_otsu"]
    assert pc.ordered(keys) == ["convnext_tiny_448", "swin_tiny_128", "inceptionv3_224_otsu"]


def test_condition_names_read_like_the_session_conditions_report():
    assert pc.condition_name("convnext_tiny_448") == "ConvNeXt-T, 448 px"
    assert pc.condition_name("efficientnetv2s_224_otsu") == "EfficientNetV2-S, 224 px + Otsu"


def test_require_started_names_the_run_without_a_test_started_file(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"runs": ["a", "b"]}))
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "test_started.json").write_text("{}")
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(pc, "get_default_runs_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="b has no test_started.json"):
        pc.require_started("plan.json")


def _predictions(correct_share, seed):
    """Five rounds of 40 photos; ``correct_share`` of them predicted right, the rest as the next class."""
    rows = []
    for round_ in range(1, 6):
        for i in range(40):
            true = i % 4
            wrong = (i * 7 + seed + round_) % 10 >= correct_share * 10
            rows.append({"photo_id": f"p{round_}_{i}", "round": round_, "partition": "test",
                         "true_class": ["healthy", "leaf_blast", "brown_spot", "hispa"][true],
                         "pred_class": ["healthy", "leaf_blast", "brown_spot", "hispa"][(true + wrong) % 4]})
    return pd.DataFrame(rows)


def test_a_clearly_better_attention_is_reported_in_both_directions():
    config = {"classes": ["healthy", "leaf_blast", "brown_spot", "hispa"], "seed": 0,
              "evaluation": {"bootstrap": {"resamples": 50, "confidence": 0.95, "stratified": True},
                             "comparison": {"ci_excludes_zero": True, "min_round_wins": 4}}}
    blocks = pd.Series({f"p{r}_{i}": f"b{i % 8}" for r in range(1, 6) for i in range(40)})
    paired = pc.paired_conditions(_predictions(0.95, 0), {"swin_tiny_128": _predictions(0.5, 1)}, blocks, config)
    assert set(paired["metric"]) == {"macro_f1", "balanced_accuracy"}
    row = paired[paired["metric"] == "macro_f1"].iloc[0]
    assert row["difference"] > 0 and row["attention_round_wins"] == 5 and row["condition_round_wins"] == 0
    assert row["verdict_photo"] == pc.ATTENTION
    swapped = pc.paired_conditions(_predictions(0.5, 1), {"swin_tiny_128": _predictions(0.95, 0)}, blocks, config)
    assert swapped[swapped["metric"] == "macro_f1"].iloc[0]["verdict_photo"] == pc.OTHER


def test_report_names_every_row_and_counts_the_verdicts():
    row = {"condition": "convnext_tiny_448", "difference": -0.012, "photo_low": -0.03, "photo_high": 0.01,
           "block_low": -0.05, "block_high": 0.03, "attention_round_wins": 2, "condition_round_wins": 3, "rounds": 5,
           "verdict_photo": pc.NONE, "verdict_block": pc.NONE}
    paired = pd.DataFrame([{**row, "metric": "macro_f1"}, {**row, "metric": "balanced_accuracy"}])
    means = pd.DataFrame([{"method": "attention", "macro_f1_mean": 0.779, "macro_f1_sd": 0.031,
                           "balanced_accuracy_mean": 0.78, "balanced_accuracy_sd": 0.03}])
    text = pc.build_report(paired, means)
    assert "ConvNeXt-T, 448 px" in text and "post-hoc" in text
    assert "macro_f1, photo range: 0 attention beats it, 1 no clear difference, 0 it beats attention" in text
