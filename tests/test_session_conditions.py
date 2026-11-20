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
    config = load_config(f"configs/session/{model}_{condition}.yaml")
    w = config["wholeimage"]
    assert config["split_files"]["folds"] == "splits/session_blocked/folds.csv"
    assert (w["model"], w["size"], w["otsu"]) == (f"{model}_session", size, otsu)
    assert w["model"] in w["models"] and w["models"][w["model"]]["revision"]
    assert w["models"][w["model"]]["revision"] == load_config("configs/wholeimage/recipe.yaml")[
        "wholeimage"]["models"][model]["revision"]


def test_swin_knows_every_input_size():
    sizes = load_config("configs/session/swin_tiny_128.yaml")["wholeimage"]["models"]["swin_tiny_session"]["size_kwargs"]
    assert sizes[128] == {"img_size": 128, "window_size": 4}
    assert {224, 448} <= set(sizes)


def test_only_runs_named_in_the_selection_are_tested(tmp_path, monkeypatch):
    monkeypatch.setattr(session_conditions, "get_default_runs_dir", lambda: tmp_path)
    runs = [f"stage5_x_session_128_lr0.001_r{r}" for r in range(1, 6)]
    (tmp_path / session_conditions.SELECTION).write_text(json.dumps({"configurations": {"x_128": runs}}))
    seen = {}
    monkeypatch.setattr(session_conditions, "evaluate_once", lambda names, plan: seen.update(names=names, plan=plan))
    session_conditions.evaluate_selected()
    assert seen == {"names": runs, "plan": "session_conditions_plan.json"}
