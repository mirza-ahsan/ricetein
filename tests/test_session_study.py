"""Checks for src/session_study.py: the session-split searches, the validation-only selection and the
once-only test plan (no GPU, no data)."""

import json

import pytest

from src import followup, session_study
from src.config import load_config

D11 = {  # the locked-split settings refitted on the session split and tested on 2026-11-16
    "inceptionv3_224": "stage5_inceptionv3_session_224_lr0.0003_r{}",
    "efficientnetv2s_224": "stage5_efficientnetv2s_session_224_lr0.0003_r{}",
    "convnext_tiny_224": "stage5_convnext_tiny_session_224_lr0.0003_r{}",
    "swin_tiny_224": "stage5_swin_tiny_session_224_lr3e-05_r{}",
    "probe_224": "stage7a_probe_e1_224_session_l2100_r{}",
    "probe_512": "stage7a_probe_e1_512_session_l2100_r{}",
    "mean": "stage7_mean_session_r{}",
    "max": "stage7_max_session_r{}",
    "attention": "stage7_attention_session_r{}",
}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(session_study, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(followup, "completed", lambda name, checkpoint: True)
    monkeypatch.setattr(session_study, "completed", lambda name, checkpoint: True)   # not the checkpoints on disk
    monkeypatch.setattr(session_study, "require_clean_code", lambda: "abc1234")

    def write(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    write("followup_selection.json", {"session": {m: [p.format(r) for r in range(1, 6)] for m, p in D11.items()}})
    # inceptionv3 moves to the extended 1e-3; the other three keep the locked-split choice
    write("session_search_cnn.json", {
        "inceptionv3": {"3e-05": 0.60, "0.0001": 0.65, "0.0003": 0.70, "0.001": 0.75, "0.003": 0.72},
        "efficientnetv2s": {"3e-05": 0.60, "0.0001": 0.65, "0.0003": 0.70, "0.001": 0.68},
        "convnext_tiny": {"3e-05": 0.60, "0.0001": 0.65, "0.0003": 0.70, "0.001": 0.68},
        "swin_tiny": {"1e-05": 0.60, "3e-05": 0.70, "0.0001": 0.65, "0.0003": 0.62}})
    write("session_search_probe_224.json", {"0.001": 0.5, "0.01": 0.5, "0.1": 0.5, "1": 0.6, "10": 0.7, "100": 0.6})
    write("session_search_probe_512.json", {"0.001": 0.5, "0.01": 0.5, "0.1": 0.5, "1": 0.6, "10": 0.6, "100": 0.7,
                                             "1000": 0.65})
    for head in session_study.HEADS:
        for suffix, score in (("", 0.70), ("_wide", 0.71 if head in ("mean", "whole512") else 0.69)):
            folder = tmp_path / f"stage7_tune_session_{head}{suffix}"
            folder.mkdir()
            (folder / "metrics.json").write_text(json.dumps(
                {"best_mean_val_macro_f1": score, "best_params": {"lr": 2e-3 if suffix else 5e-4}}))
    return tmp_path


def test_every_session_study_config_changes_only_the_split_the_names_and_the_wide_range():
    session_folds = "splits/session_blocked/folds.csv"
    for head in session_study.HEADS:
        locked = load_config(f"configs/mil/{head}.yaml")
        for suffix in ("", "_wide"):
            config = load_config(f"configs/mil/session_{head}{suffix}.yaml")
            stem = f"session_{head}{suffix}"
            assert config["split_files"]["folds"] == session_folds
            assert config["mil"] == {**locked["mil"], "name": stem, "tuned_from": stem}
            lr = [1e-4, 1e-2] if suffix else locked["search"]["lr"]
            assert config["search"] == {**locked["search"], "lr": lr}
            assert config.get("whole_bag") == locked.get("whole_bag")
            different = {k for k in locked.keys() | config.keys() if locked.get(k) != config.get(k)}
            assert different <= {"split_files", "mil", "search", "config_file"}


def test_the_ten_studies_are_the_five_heads_with_and_without_the_wide_range():
    assert len(session_study.STUDIES) == 10
    assert {"attention", "attention_wide", "whole224", "whole512_wide"} <= set(session_study.STUDIES)


def test_cnn_search_reads_the_session_configs_and_applies_the_edge_rule(runs, monkeypatch):
    seen = []

    def run_one(config, lr, r):
        assert config["split_files"]["folds"] == "splits/session_blocked/folds.csv"
        seen.append((config["wholeimage"]["model"], lr, r))
        return "name", {3e-5: 0.6, 1e-4: 0.7, 3e-4: 0.8, 1e-3: 0.75}[lr]   # best at the top, then worse

    monkeypatch.setattr(followup.stage5, "run_one", run_one)
    session_study.cnn()
    searched = json.loads((runs / "session_search_cnn.json").read_text())
    assert list(searched["swin_tiny"]) == ["3e-05", "0.0001", "0.0003", "0.001"]   # one value added, then stop
    assert ("swin_tiny_session", 1e-3, 5) in seen and len(seen) == 4 * 4 * 5


def test_probe_search_fits_one_strength_per_call_and_extends_at_the_edge(runs, monkeypatch):
    calls = []

    def run(config):
        strengths = config["probe"]["l2_strengths"]
        assert len(strengths) == 1 and config["probe"]["name"].endswith("_session")
        calls.append((config["probe"]["size"], strengths[0]))
        return {"trials": [{"mean_val_macro_f1": {100.0: 0.8, 1000.0: 0.7}.get(strengths[0], 0.5)}]}

    monkeypatch.setattr(session_study.linear_probe, "run", run)
    session_study.probes()
    searched = json.loads((runs / "session_search_probe_224.json").read_text())
    assert list(searched) == ["0.001", "0.01", "0.1", "1", "10", "100", "1000"]   # 100 was the edge; 1000 is worse
    assert calls.count((512, 100.0)) == 1 and (512, 1000.0) in calls


def test_select_names_the_runs_and_marks_what_differs_from_the_locked_split_settings(runs):
    selection = session_study.select()
    methods = selection["methods"]
    assert selection["partition"] == "val" and selection["split"] == "splits/session_blocked/folds.csv"
    assert methods["inceptionv3_224"]["selected"] == "0.001" and methods["inceptionv3_224"]["changed"]
    assert methods["inceptionv3_224"]["runs"][0] == "stage5_inceptionv3_session_224_lr0.001_r1"
    assert methods["swin_tiny_224"]["runs"][4] == "stage5_swin_tiny_session_224_lr3e-05_r5"
    assert not methods["swin_tiny_224"]["changed"] and not methods["convnext_tiny_224"]["changed"]
    assert methods["probe_224"]["changed"] and methods["probe_224"]["runs"][0] == "stage7a_probe_e1_224_session_l210_r1"
    assert not methods["probe_512"]["changed"]
    assert methods["mean"]["selected"] == "wide" and methods["mean"]["study"] == "session_mean_wide"
    assert methods["mean"]["runs"][0] == "stage7_session_mean_wide_r1" and methods["mean"]["params"] == {"lr": 2e-3}
    assert methods["attention"]["selected"] == "declared"
    assert methods["attention"]["runs"][0] == "stage7_session_attention_r1"
    assert methods["whole512"]["runs"][0] == "stage7_session_whole512_wide_r1"
    assert all(methods[head]["changed"] for head in session_study.HEADS)   # none equals a refitted run
    assert len(methods) == 11
    assert json.loads((runs / "session_selection.json").read_text()) == selection


def test_select_refuses_while_any_selected_run_is_unfinished(runs, monkeypatch):
    monkeypatch.setattr(followup, "completed", lambda name, checkpoint: False)
    with pytest.raises(RuntimeError, match="not complete"):
        session_study.select()
    assert not (runs / "session_selection.json").exists()


def test_seeds_refit_only_the_selected_tile_studies(runs, monkeypatch):
    session_study.select()
    fitted = []
    monkeypatch.setattr(session_study.train_mil, "final",
                        lambda config, seeds: fitted.append((config["mil"]["name"], seeds)))
    session_study.seeds()
    assert fitted == [("session_mean_wide", [0, 1, 2]), ("session_max", [0, 1, 2]), ("session_attention", [0, 1, 2])]


def test_evaluate_freezes_one_plan_and_skips_runs_that_were_already_tested(runs, monkeypatch):
    session_study.select()
    tested_before = runs / "stage5_swin_tiny_session_224_lr3e-05_r1"
    tested_before.mkdir()
    (tested_before / "test_metrics.json").write_text("{}")
    (tested_before / "test_predictions.csv").write_text("")
    frozen, tested = {}, []
    monkeypatch.setattr(followup.evaluate, "freeze_plan",
                        lambda names, path: frozen.update(names=list(names), path=path))
    monkeypatch.setattr(followup.evaluate, "evaluate", lambda name, plan: tested.append(name))
    session_study.evaluate_selected()
    assert len(frozen["names"]) == len(set(frozen["names"])) == 55
    assert frozen["path"] == runs / "session_evaluation_plan.json"
    assert tested_before.name in frozen["names"] and tested_before.name not in tested
    assert len(tested) == 54 and "stage7_session_attention_r1" in tested


def test_select_cannot_change_once_the_test_plan_is_frozen(runs):
    selection = session_study.select()
    names = [run for entry in selection["methods"].values() for run in entry["runs"]]
    (runs / session_study.PLAN).write_text(json.dumps({"runs": {name: {} for name in names}}))
    assert session_study.select() == selection                      # the same choice is still fine
    (runs / "session_search_probe_224.json").write_text(json.dumps({"10": 0.6, "100": 0.7}))
    with pytest.raises(RuntimeError, match="frozen with other runs"):
        session_study.select()
    assert json.loads((runs / "session_selection.json").read_text()) == selection


def test_a_tested_run_without_its_training_files_stops_every_action(runs, monkeypatch):
    tested = runs / "stage5_swin_tiny_session_224_lr3e-05_r1"
    tested.mkdir()
    (tested / "test_started.json").write_text("{}")
    session_study.require_intact()                                   # complete (the fixture says so): fine
    monkeypatch.setattr(session_study, "completed", lambda name, checkpoint: False)
    with pytest.raises(RuntimeError, match="stage5_swin_tiny_session_224_lr3e-05_r1: tested"):
        session_study.require_intact()
    monkeypatch.setattr("sys.argv", ["session_study", "cnn"])
    with pytest.raises(RuntimeError, match="tested"):
        session_study.main()


def test_main_sends_a_study_name_to_its_session_config_and_rejects_stray_names(runs, monkeypatch):
    started = []
    monkeypatch.setattr(session_study, "study", started.append)
    monkeypatch.setattr("sys.argv", ["session_study", "study", "attention_wide"])
    session_study.main()
    assert started == ["session_attention_wide"]
    for argv in (["session_study", "study", "attention_session"], ["session_study", "select", "extra"]):
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            session_study.main()
