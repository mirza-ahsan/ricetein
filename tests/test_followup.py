"""Checks for src/followup.py: the edge rule, tie-breaking and the validation-only selection (no GPU, no data)."""

import json

import pytest

from src import followup

CNN_DECLARED = [3e-5, 1e-4, 3e-4]


def test_edge_rule_extends_the_top_and_the_bottom():
    top = {3e-5: 0.7, 1e-4: 0.8, 3e-4: 0.9}
    assert followup.next_value(top, followup.LR_LADDER, CNN_DECLARED) == 1e-3
    bottom = {3e-5: 0.9, 1e-4: 0.8, 3e-4: 0.7}
    assert followup.next_value(bottom, followup.LR_LADDER, CNN_DECLARED) == 1e-5


def test_edge_rule_stops_in_the_middle_at_the_budget_and_at_the_ladder_end():
    middle = {3e-5: 0.7, 1e-4: 0.9, 3e-4: 0.8}
    assert followup.next_value(middle, followup.LR_LADDER, CNN_DECLARED) is None
    three_added = {3e-5: 0.1, 1e-4: 0.2, 3e-4: 0.3, 1e-3: 0.4, 3e-3: 0.5, 1e-2: 0.6}
    assert followup.next_value(three_added, followup.LR_LADDER, CNN_DECLARED) is None  # budget of 3, also the ladder end
    four_declared = [1e-6, 3e-6, 1e-5, 3e-5]
    at_budget = {1e-6: 0.1, 3e-6: 0.2, 1e-5: 0.3, 3e-5: 0.4, 1e-4: 0.5, 3e-4: 0.6, 1e-3: 0.7}
    assert followup.next_value(at_budget, followup.LR_LADDER, four_declared) is None
    last = {1e-3: 0.1, 3e-3: 0.2, 1e-2: 0.9}
    assert followup.next_value(last, followup.LR_LADDER, [1e-3, 3e-3, 1e-2]) is None


def test_the_smaller_value_wins_a_tie():
    assert followup.best_value({1.0: 0.5, 10.0: 0.5}) == 1.0
    assert followup.best_value({10.0: 0.5, 1.0: 0.5}) == 1.0


CORE_RUNS = {
    "inceptionv3_224": "stage5_inceptionv3_224_lr0.0003_r{}",
    "efficientnetv2s_224": "stage5_efficientnetv2s_224_lr0.0003_r{}",
    "convnext_tiny_224": "stage5_convnext_tiny_224_lr0.0003_r{}",
    "swin_tiny_224": "stage5_swin_tiny_224_lr3e-05_r{}",
    "probe_224": "stage7a_probe_e1_224_l2100_r{}",
    "probe_512": "stage7a_probe_e1_512_l2100_r{}",
    "mean": "stage7_mean_r{}",
    "max": "stage7_max_r{}",
    "attention": "stage7_attention_r{}",
}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(followup, "core_methods",
                        lambda: {m: [pattern.format(r) for r in range(1, 6)] for m, pattern in CORE_RUNS.items()})
    monkeypatch.setattr(followup, "core_config",
                        lambda m: {"lr": 3e-5 if m.startswith("swin") else 3e-4, "l2_strength": 100.0})
    monkeypatch.setattr(followup, "completed", lambda name, checkpoint: True)
    monkeypatch.setattr(followup, "require_clean_code", lambda: "abc1234")

    def write(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    # inceptionv3 moves to the extended 1e-3; the other three keep their core choice
    write("followup_extend_cnn.json", {
        "inceptionv3": {"0.0001": 0.80, "0.0003": 0.85, "0.001": 0.90, "3e-05": 0.70},
        "efficientnetv2s": {"0.0001": 0.80, "0.0003": 0.85, "3e-05": 0.70},
        "convnext_tiny": {"0.0001": 0.80, "0.0003": 0.85, "3e-05": 0.70},
        "swin_tiny": {"0.0001": 0.80, "0.0003": 0.85, "3e-05": 0.90, "1e-05": 0.70}})
    write("followup_extend_probe_224.json", {"10": 0.80, "100": 0.85, "1000": 0.90, "10000": 0.80})
    write("followup_extend_probe_512.json", {"10": 0.80, "100": 0.85, "1000": 0.80})
    for name, best in (("mean", 0.80), ("max", 0.78), ("attention", 0.79)):
        for suffix, score in (("", best), ("_wide", best + (0.01 if name == "mean" else -0.01))):
            folder = tmp_path / f"stage7_tune_{name}{suffix}"
            folder.mkdir()
            (folder / "metrics.json").write_text(json.dumps(
                {"best_mean_val_macro_f1": score, "best_params": {"lr": 1e-3 if suffix else 5e-4}}))
    return tmp_path


def test_select_flags_only_the_changed_methods_and_names_their_runs(runs):
    selection = followup.select()
    methods = selection["methods"]
    assert selection["partition"] == "val" and selection["code_version"] == "abc1234"
    assert methods["inceptionv3_224"]["changed"] and methods["inceptionv3_224"]["selected"] == "0.001"
    assert methods["inceptionv3_224"]["runs"][0] == "stage5_inceptionv3_224_lr0.001_r1"
    assert not methods["efficientnetv2s_224"]["changed"]
    assert methods["efficientnetv2s_224"]["runs"][0] == "stage5_efficientnetv2s_224_lr0.0003_r1"
    assert not methods["swin_tiny_224"]["changed"]  # 3e-5 is still the best; 1e-5 was worse
    assert methods["probe_224"]["changed"]
    assert methods["probe_224"]["runs"][0] == "stage7a_probe_e1_224_ext_l21000_r1"
    assert not methods["probe_512"]["changed"]
    assert methods["probe_512"]["runs"][0] == "stage7a_probe_e1_512_l2100_r1"
    assert methods["mean"]["changed"] and methods["mean"]["selected"] == "wide"
    assert methods["mean"]["runs"][0] == "stage7_mean_wide_r1"
    assert methods["mean"]["params"] == {"lr": 1e-3}
    for name in ("max", "attention"):
        assert not methods[name]["changed"] and methods[name]["selected"] == "declared"
        assert methods[name]["runs"][0] == f"stage7_{name}_r1"
    assert methods["whole224"]["runs"][4] == "stage7_whole224_r5" and methods["whole512"]["changed"]
    assert json.loads((runs / "followup_selection.json").read_text())["methods"] == methods


def test_session_runs_are_listed_for_all_nine_methods_with_the_core_settings(runs):
    session = followup.select()["session"]
    assert len(session) == 9 and all(len(names) == 5 for names in session.values())
    assert all("_session" in name for names in session.values() for name in names)
    assert session["swin_tiny_224"][0] == "stage5_swin_tiny_session_224_lr3e-05_r1"
    assert session["probe_512"][0] == "stage7a_probe_e1_512_session_l2100_r1"
    assert session["attention"][4] == "stage7_attention_session_r5"


def test_a_wide_study_that_only_ties_loses(runs):
    (runs / "stage7_tune_mean_wide" / "metrics.json").write_text(json.dumps(
        {"best_mean_val_macro_f1": 0.80, "best_params": {"lr": 1e-3}}))
    methods = followup.select()["methods"]
    assert methods["mean"]["selected"] == "declared" and not methods["mean"]["changed"]


def test_select_refuses_while_any_run_is_unfinished(runs, monkeypatch):
    monkeypatch.setattr(followup, "completed", lambda name, checkpoint: False)
    with pytest.raises(RuntimeError, match="not complete"):
        followup.select()
    assert not (runs / "followup_selection.json").exists()


def test_probe_runs_are_checked_for_a_pickle_and_the_rest_for_a_torch_file(runs, monkeypatch):
    seen = {}
    monkeypatch.setattr(followup, "completed", lambda name, checkpoint: seen.setdefault(name, checkpoint) or True)
    followup.select()
    assert seen["stage7a_probe_e1_224_ext_l21000_r1"] == "model.pkl"
    assert seen["stage7_mean_wide_r1"] == "model.pt" and seen["stage5_inceptionv3_224_lr0.001_r1"] == "model.pt"


def test_evaluate_tests_session_and_changed_runs_once_from_one_frozen_plan(runs, monkeypatch):
    followup.select()
    frozen, tested = {}, []
    monkeypatch.setattr(followup.evaluate, "freeze_plan", lambda names, path: frozen.update(names=list(names), path=path))
    monkeypatch.setattr(followup.evaluate, "evaluate", lambda name, plan: tested.append((name, plan)))
    followup.evaluate_new()
    names = frozen["names"]
    assert len(names) == len(set(names)) == 45 + 10 + 5 + 5 + 5  # session, whole controls, inceptionv3, probe_224, mean
    assert "stage7_max_r1" not in names and "stage7_mean_wide_r1" in names
    assert [n for n, _ in tested] == names and {p for _, p in tested} == {runs / "followup_evaluation_plan.json"}


def test_evaluate_skips_runs_that_already_have_both_test_files(runs, monkeypatch):
    followup.select()
    done = runs / "stage7_whole224_r1"
    done.mkdir()
    (done / "test_metrics.json").write_text("{}")
    (done / "test_predictions.csv").write_text("")
    tested = []
    monkeypatch.setattr(followup.evaluate, "freeze_plan", lambda names, path: None)
    monkeypatch.setattr(followup.evaluate, "evaluate", lambda name, plan: tested.append(name))
    followup.evaluate_new()
    assert "stage7_whole224_r1" not in tested and "stage7_whole224_r2" in tested


def test_completed_needs_a_finished_record_and_the_metrics_predictions_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("src.records.get_default_runs_dir", lambda: tmp_path)
    folder = tmp_path / "stage7_mean_r1"
    folder.mkdir()
    (folder / "record.json").write_text(json.dumps({"status": "completed"}))
    for name in ("metrics.json", "predictions.csv"):
        (folder / name).write_text("")
    assert not followup.completed("stage7_mean_r1", "model.pt")          # no checkpoint yet
    (folder / "model.pt").write_text("")
    assert followup.completed("stage7_mean_r1", "model.pt")
    (folder / "record.json").write_text(json.dumps({"status": "running"}))
    assert not followup.completed("stage7_mean_r1", "model.pt")          # not finished
    assert not followup.completed("stage7_missing_r1", "model.pt")
