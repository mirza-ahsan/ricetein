"""Checks for src/committee.py on made-up predictions, and on the tracked validation predictions (no GPU, no test files)."""

import json

import numpy as np
import pandas as pd
import pytest

from src import committee
from src.config import load_config

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
CHOICE = {"min_members": 2, "max_members": 5, "min_gain": 0.005, "tie": 1e-9, "weights": [1, 2, 3]}


def table(round_: int, partition: str, probs: np.ndarray, truth: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"round": round_, "partition": partition,
                        "photo_id": [f"{partition}{round_}_{i:03d}" for i in range(len(truth))],
                        "true_class": truth, "pred_class": np.array(CLASSES)[probs.argmax(axis=1)]})
    for k, c in enumerate(CLASSES):
        out[f"prob_{c}"] = probs[:, k]
    return out


def one_hot(labels: list[int], confidence: float = 0.7) -> np.ndarray:
    probs = np.full((len(labels), 4), (1 - confidence) / 3)
    probs[np.arange(len(labels)), labels] = confidence
    return probs


TRUTH = [0, 1, 2, 3] * 5


def validation(guesses: dict[str, list[int]]) -> committee.Validation:
    """Members that always answer with the given class indices, for two rounds of the same 20 photos."""
    names = [CLASSES[i] for i in TRUTH]
    return committee.Validation({m: {r: table(r, "val", one_hot(g), names) for r in (1, 2)} for m, g in guesses.items()},
                                CLASSES)


def test_average_is_the_plain_mean_and_weights_move_it():
    a, b = table(1, "test", one_hot([0, 1]), ["healthy", "leaf_blast"]), table(1, "test", one_hot([1, 1]), ["healthy", "leaf_blast"])
    plain = committee.average([a, b], None, CLASSES)
    assert np.allclose(plain[["prob_healthy", "prob_leaf_blast"]].to_numpy()[0], [0.4, 0.4])
    assert plain["pred_class"].tolist()[1] == "leaf_blast"
    heavy = committee.average([a, b], [3, 1], CLASSES)
    assert heavy["pred_class"][0] == "healthy"                          # member a carries the weight
    assert np.allclose(heavy[[f"prob_{c}" for c in CLASSES]].sum(axis=1), 1.0)
    assert list(plain.columns) == [*committee.TABLE_COLUMNS, *[f"prob_{c}" for c in CLASSES]]


def test_average_refuses_members_that_predict_other_photos():
    a = table(1, "test", one_hot([0, 1]), ["healthy", "leaf_blast"])
    b = a.assign(photo_id=["x", "y"])
    with pytest.raises(ValueError, match="same photos"):
        committee.average([a, b], None, CLASSES)


def test_score_is_the_mean_of_round_macro_f1():
    val = validation({"perfect": TRUTH, "wrong": [(i + 1) % 4 for i in TRUTH]})
    assert val.score(["perfect"]) == 1.0 and val.score(["wrong"]) == 0.0


def test_greedy_takes_the_best_first_and_always_a_second_member():
    val = validation({"best": TRUTH, "other": [0] * 20, "bad": [1] * 20})
    found = committee.greedy(val, ["bad", "other", "best"], CHOICE)
    assert found["members"][0] == "best" and len(found["members"]) == 2   # the second is added although it cannot help


def test_greedy_stops_when_a_member_adds_less_than_the_declared_gain():
    val = validation({"a": TRUTH, "b": TRUTH, "c": TRUTH})
    assert committee.greedy(val, ["a", "b", "c"], CHOICE)["members"] == ["a", "b"]   # b is a tie won by order; c adds 0


def test_greedy_adds_a_third_member_that_outvotes_the_mistakes_of_two():
    def wrong_on(photos: set[int]) -> list[int]:
        return [0 if i in photos else t for i, t in enumerate(TRUTH)]      # a wrong answer is always class 0

    val = validation({"one": wrong_on({1, 2, 3, 5}), "two": wrong_on({6, 7, 9, 10}), "three": wrong_on({11, 13, 14, 15})})
    found = committee.greedy(val, ["one", "two", "three"], CHOICE)
    assert found["members"] == ["one", "two", "three"] and found["steps"][-1]["gain"] >= CHOICE["min_gain"]


def test_greedy_with_a_forced_first_member_starts_there():
    val = validation({"best": TRUTH, "weak": [0] * 20, "forced": [1] * 20})
    found = committee.greedy(val, ["best", "weak", "forced"], CHOICE, first="forced")
    assert found["members"][0] == "forced" and found["steps"][0]["added"] != "forced"


def test_ties_go_to_the_member_listed_first():
    assert committee.first_best([(0.5, "late"), (0.5 + 1e-12, "later")], 1e-9) == (0.5, "late")
    assert committee.first_best([(0.5, "a"), (0.6, "b")], 1e-9) == (0.6, "b")


def test_weights_stay_plain_without_a_declared_gain():
    val = validation({"a": TRUTH, "b": TRUTH})
    result = committee.weighted(val, ["a", "b"], CHOICE)
    assert result["same_as_plain"] and result["weights"] == [1, 1]


def test_weights_are_used_when_they_help():
    names = [CLASSES[i] for i in TRUTH]
    careful = one_hot(TRUTH, 0.7)                                              # right on every photo
    sure_but_wrong_early = one_hot([0] * 10 + TRUTH[10:], 0.9)                 # confidently wrong on the first ten
    val = committee.Validation({m: {r: table(r, "val", p, names) for r in (1, 2)}
                                for m, p in (("careful", careful), ("sure", sure_but_wrong_early))}, CLASSES)
    result = committee.weighted(val, ["careful", "sure"], CHOICE)
    assert not result["same_as_plain"] and result["weights"][0] > result["weights"][1]
    assert result["score"] >= val.score(["careful", "sure"]) + CHOICE["min_gain"]


def test_choice_is_deterministic_and_reads_validation_files_only(monkeypatch):
    config, seen = load_config("configs/committee.yaml"), []
    original = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: (seen.append(str(path)), original(path, *a, **k))[1])
    first, second = committee.choose(config), committee.choose(config)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert seen and all(p.endswith("/predictions.csv") for p in seen)         # no test, Paddy or TTA file was opened
    assert first["committees"]["row2_plain"]["members"] and set(first["committees"]["row3_plain"]["members"]) <= set(
        config["committee"]["pools"]["B"])


def test_pool_a_choice_without_a_tile_head_falls_back_to_the_forced_member(monkeypatch):
    config = load_config("configs/committee.yaml")
    only_cnn = dict(config["committee"]["pools"], A=config["committee"]["pools"]["B"] + ["attention_e1"])
    config["committee"]["pools"] = only_cnn
    config["committee"]["forced_first"] = "attention_e1"
    chosen = committee.choose(config)["committees"]["row2_plain"]
    assert any(config["committee"]["members"][m]["tile"] for m in chosen["members"])


def test_selection_never_changes_once_written(tmp_path, monkeypatch):
    config = {"classes": CLASSES, "committee": {"files": {"selection": str(tmp_path / "s.json"),
                                                          "disagreement": str(tmp_path / "d.csv")}}}
    monkeypatch.setattr(committee, "require_clean_code", lambda: "abc1234")
    result = {"member_runs": {}, "committees": {}, "tta_member": "x", "disagreement": [], "partition": "val"}
    monkeypatch.setattr(committee, "choose", lambda _: result)
    committee.select(config)
    monkeypatch.setattr(committee, "choose", lambda _: {**result, "tta_member": "y"})
    with pytest.raises(ValueError, match="never replaced"):
        committee.select(config)


# ------------------------------------------------------------------------------------------- the run-once marker
def make_runs(root, members: dict[str, list[int]], rounds=(1, 2)) -> dict[str, list[str]]:
    names = [CLASSES[i] for i in TRUTH]
    runs = {}
    for m, guess in members.items():
        runs[m] = []
        for r in rounds:
            folder = root / f"{m}_r{r}"
            folder.mkdir()
            table(r, "val", one_hot(guess), names).to_csv(folder / "predictions.csv", index=False)
            table(r, "test", one_hot(guess), names).to_csv(folder / "test_predictions.csv", index=False)
            runs[m].append(folder.name)
    return runs


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(committee, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(committee, "require_clean_code", lambda: "abc1234")
    runs = make_runs(tmp_path, {"a": TRUTH, "b": [0] * 20})
    config = {"classes": CLASSES, "committee": {"rounds": [1, 2]}}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"tested_members_test_sha256": {
        run: committee.sha256(tmp_path / run / "test_predictions.csv") for rs in runs.values() for run in rs}}))
    spec = {"members": ["a", "b"], "weights": [2, 1], "runs": runs}
    return config, spec, plan, tmp_path


def test_a_committee_is_tested_once_and_a_crash_after_the_marker_is_refused(setup):
    config, spec, plan, root = setup
    committee.test_committee(config, "row2_plain", spec, plan)
    folder = root / "committee_row2_plain"
    assert {"test_started.json", "test_predictions.csv", "predictions.csv", "test_metrics.json", "committee.json"} <= {
        p.name for p in folder.iterdir()}
    done = (folder / "test_predictions.csv").read_bytes()
    committee.test_committee(config, "row2_plain", spec, plan)              # a finished committee is left alone
    assert (folder / "test_predictions.csv").read_bytes() == done
    (folder / "test_predictions.csv").unlink()                                # a crash after the marker
    with pytest.raises(ValueError, match="stop and report"):
        committee.test_committee(config, "row2_plain", spec, plan)
    assert (folder / "test_started.json").exists()


def test_a_member_whose_test_file_changed_after_the_plan_is_refused(setup):
    config, spec, plan, root = setup
    path = root / "a_r1" / "test_predictions.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after the plan"):
        committee.test_committee(config, "row2_plain", spec, plan)


def test_committee_test_predictions_are_the_round_wise_weighted_average(setup):
    config, spec, plan, root = setup
    committee.test_committee(config, "row2_plain", spec, plan)
    out = pd.read_csv(root / "committee_row2_plain" / "test_predictions.csv")
    assert sorted(out["round"].unique()) == [1, 2] and len(out) == 40
    assert list(out.columns) == [*committee.TABLE_COLUMNS, *[f"prob_{c}" for c in CLASSES]]
    expected = (2 * one_hot(TRUTH) + 1 * one_hot([0] * 20)) / 3
    assert np.allclose(out[[f"prob_{c}" for c in CLASSES]].to_numpy()[:20], expected)


def test_plan_is_frozen_once_and_refuses_a_changed_selection(setup, monkeypatch):
    config, spec, _, root = setup
    selection = {"member_runs": spec["runs"], "tta_member": "a",
                 "committees": {"row2_plain": {"members": ["a", "b"], "weights": [1, 1]}}}
    (root / "selection.json").write_text(json.dumps(selection))
    config["committee"]["files"] = {"selection": str(root / "selection.json"), "plan": str(root / "committee_plan.json")}
    committee.plan(config)
    frozen = json.loads((root / "committee_plan.json").read_text())
    assert frozen["runs"] == {} and set(frozen["tested_members_test_sha256"]) == {f"{m}_r{r}" for m in "ab" for r in (1, 2)}
    committee.plan(config)                                                   # again: nothing changes
    (root / "selection.json").write_text(json.dumps({**selection, "tta_member": "b"}))
    with pytest.raises(ValueError, match="another selection"):
        committee.plan(config)


# ------------------------------------------------------------------------------------------- rules and views
def test_the_rule_is_read_in_both_directions():
    rule = {"min_round_wins": 4, "ci_excludes_zero": True}
    assert committee.two_way(0.01, 0.05, 4, 5, rule) == "A beats B"
    assert committee.two_way(0.01, 0.05, 3, 5, rule) == "no clear difference"
    assert committee.two_way(-0.05, -0.01, 1, 5, rule) == "B beats A"
    assert committee.two_way(-0.05, -0.01, 2, 5, rule) == "no clear difference"
    assert committee.two_way(-0.01, 0.05, 5, 5, rule) == "no clear difference"


def test_eight_views_are_distinct_and_start_with_the_photo():
    import torch

    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    views = committee.views(x)
    assert len(views) == 8 and torch.equal(views[0], x)
    assert len({v.numpy().tobytes() for v in views}) == 8
    assert all(v.shape == x.shape for v in views)


# ------------------------------------------------------------------------------------------- the column contract
def test_results_columns_match_what_make_tables_expects(tmp_path):
    """make_tables.passthrough turns float columns whose names hold f1/accuracy/recall/precision into percentages
    and prints other floats with 3 significant digits, so score columns must be fractions named that way and every
    other column text."""
    scores = pd.DataFrame({"name": ["attention_e1", "row2_plain"], **{
        f"{m}_{k}": [0.78, 0.8] for m in ("macro_f1", "balanced_accuracy") for k in ("mean", "sd")}})
    paired = pd.DataFrame([{"A": "row2_plain", "B": "attention_e1", "metric": "macro_f1", "difference": 0.02, "photo_low": 0.0,
                            "photo_high": 0.04, "block_low": -0.01, "block_high": 0.05, "round_wins": 4, "rounds": 5,
                            "verdict_photo": "A beats B", "verdict_block": "no clear difference"}])
    paddy = pd.DataFrame({"committee": ["attention_e1", "row2_plain"], **{
        f"{m}_{k}": [0.26, 0.3] for m in ("macro_f1", "balanced_accuracy") for k in ("mean", "sd")}})
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"committees": {"row2_plain": {"members": ["a", "b"], "weights": [1, 2]}}}))
    config = {"committee": {"best_single": "attention_e1", "cost": "pending: Stage 13", "files": {"plan": str(plan)}}}
    out = committee.results_rows(config, {"scores": scores, "paired": paired, "paddy": paddy})
    assert list(out["row"]) == ["1 best single model", "2 committee with tile models"]
    assert out.loc[1, "members"] == "a + b x2" and out.loc[1, "rounds_won"] == "4/5"
    assert out.loc[1, "verdict_photo"] == "committee beats attention" and out.loc[0, "verdict_photo"] == "–"
    for col in out.columns:
        is_float = pd.api.types.is_float_dtype(out[col])
        assert is_float == col.startswith(("test_", "paddy_")), col           # only the score columns are numbers
        if is_float:
            assert "f1" in col or "accuracy" in col                           # and they are shown as percentages
    assert (out["cost"] == "pending: Stage 13").all()


def test_paddy_scores_compare_committee_rounds_with_the_single_model(monkeypatch):
    names = [CLASSES[i] for i in TRUTH] * 3
    truth = TRUTH * 3   # class index of each of the 60 photos

    def rounds(guess: list[int]) -> dict[int, pd.DataFrame]:
        return {r: table(r, "paddy", one_hot(guess), names).assign(photo_id=[f"pdd{i}" for i in range(60)])
                for r in (1, 2, 3, 4, 5)}                                    # the same Paddy photos in every round

    config = {"classes": CLASSES, "seed": 0, "committee": {"rounds": [1, 2, 3, 4, 5]},
              "evaluation": {"bootstrap": {"resamples": 50, "confidence": 0.95, "stratified": True},
                             "comparison": {"min_round_wins": 4, "ci_excludes_zero": True}}}
    better = committee.paddy_scores(config, rounds(truth), rounds([0] * 60))
    assert better["macro_f1_round_wins"] == 5 and better["macro_f1_verdict"] == "A beats B"
    assert better["macro_f1_mean"] == 1.0 and better["macro_f1_five_round"] == 1.0
    worse = committee.paddy_scores(config, rounds([0] * 60), rounds(truth))
    assert worse["macro_f1_verdict"] == "B beats A" and worse["macro_f1_round_wins"] == 0
    same = committee.paddy_scores(config, rounds(truth), rounds(truth))
    assert same["macro_f1_verdict"] == "no clear difference"
