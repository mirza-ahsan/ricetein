"""Checks for src/session_conditions.py: the validation-only choice of the 448 px model, the run names of the
session condition configs and the once-only test plan (no GPU, no data)."""

import json

import pytest

from src import followup, session_conditions
from src.config import load_config

SELECTION = {"methods": {
    "inceptionv3_224": {"selected": "0.001", "searched": {"0.0003": 0.728, "0.001": 0.731}},
    "efficientnetv2s_224": {"selected": "0.001", "searched": {"0.001": 0.759, "0.003": 0.751}},
    "convnext_tiny_224": {"selected": "0.0003", "searched": {"0.0003": 0.772, "0.001": 0.765}},
    "swin_tiny_224": {"selected": "0.0001", "searched": {"3e-05": 0.760, "0.0001": 0.771}},
}}


def test_selected_learning_rate_and_validation_score_are_read_from_the_selection():
    chosen = session_conditions.selected_at_224(SELECTION)
    assert chosen["inceptionv3"] == (0.001, 0.731)
    assert chosen["swin_tiny"] == (0.0001, 0.771)


def test_448_model_is_the_best_validation_score_and_ties_go_to_the_first_listed():
    chosen = session_conditions.selected_at_224(SELECTION)
    assert session_conditions.best_model(chosen) == "convnext_tiny"
    tied = {model: (1e-4, 0.7) for model in followup.MODELS}
    assert session_conditions.best_model(tied) == followup.MODELS[0]


@pytest.mark.parametrize("model", followup.MODELS)
@pytest.mark.parametrize("condition, size, otsu", [("128", 128, False), ("224_otsu", 224, True), ("448", 448, False)])
def test_condition_configs_use_the_session_split_and_keep_the_run_names_apart(model, condition, size, otsu):
    config = session_conditions.condition_config(model, condition)
    w = config["wholeimage"]
    assert config["split_files"]["folds"] == "splits/session_blocked/folds.csv"
    assert (w["model"], w["size"], w["otsu"]) == (f"{model}_session", size, otsu)
    assert w["models"][w["model"]]["revision"] == load_config("configs/wholeimage/recipe.yaml")[
        "wholeimage"]["models"][model]["revision"]
    assert w["batch_size"][size] == (16 if size == 448 else 32)


def test_swin_knows_every_input_size():
    sizes = load_config("configs/session/swin_tiny_224.yaml")["wholeimage"]["models"]["swin_tiny_session"]["size_kwargs"]
    assert sizes[128] == {"img_size": 128, "window_size": 4}
    assert {224, 448} <= set(sizes)


def test_planned_runs_are_45_and_448_is_only_for_the_best_model():
    pairs = session_conditions.planned("convnext_tiny")
    assert len(pairs) * len(followup.ROUNDS) == 45    # the 45 runs declared in handbook §24
    assert [m for m, c in pairs if c == "448"] == ["convnext_tiny"]


def test_a_selection_that_is_not_the_declared_runs_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(session_conditions, "get_default_runs_dir", lambda: tmp_path)
    (tmp_path / "session_selection.json").write_text(json.dumps(SELECTION))
    full = {f"{m}_{c}": [f"run{r}" for r in followup.ROUNDS] for m, c in session_conditions.planned("convnext_tiny")}
    seen = {}
    monkeypatch.setattr(session_conditions, "evaluate_once", lambda names, plan: seen.update(names=names, plan=plan))

    def store(configurations):
        (tmp_path / session_conditions.SELECTION).write_text(json.dumps({"configurations": configurations}))

    for broken in ({**full, "swin_tiny_128": ["run1"]}, {k: v for k, v in full.items() if k != "swin_tiny_128"},
                   {**full, "swin_tiny_448": full["swin_tiny_128"]}):
        store(broken)
        with pytest.raises(RuntimeError, match="declared runs"):
            session_conditions.evaluate_selected()
    assert not seen
    store(full)
    session_conditions.evaluate_selected()
    assert sorted(seen["names"]) == sorted(run for names in full.values() for run in names)
    assert seen["plan"] == "session_conditions_plan.json"


def test_every_condition_has_one_photo_cache_and_the_cache_does_not_depend_on_the_model(monkeypatch):
    built = []
    monkeypatch.setattr(session_conditions, "photo_cache", lambda config, size, otsu, workers: built.append((size, otsu)))
    session_conditions.caches()
    assert built == [(128, False), (224, True), (448, False)]
