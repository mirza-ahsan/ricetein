"""Frozen selection and one-time test protections, without touching research data or a GPU."""

import json

import pandas as pd
import pytest

from src import evaluate as ev
from src import records


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "get_code_version", lambda: "abc123")
    monkeypatch.setattr(ev, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(records, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(ev, "read_folds", lambda config: (None, "split-sha"))
    folder = tmp_path / "stage7_attention_r1"
    folder.mkdir()
    record = {
        "status": "completed",
        "code_version": "abc123",
        "split_checksum": "split-sha",
        "config": {"classes": ["a", "b"]},
    }
    (folder / "record.json").write_text(json.dumps(record))
    (folder / "model.pt").write_bytes(b"checkpoint")
    (folder / "metrics.json").write_text("{}")
    (folder / "predictions.csv").write_text("true_class,pred_class\na,a\n")
    prediction = pd.DataFrame({"true_class": ["a", "b"], "pred_class": ["a", "b"]})
    calls = []
    monkeypatch.setitem(ev.EVALUATORS, "stage7", lambda *args: calls.append(True) or prediction)
    return folder, tmp_path / "plan.json", calls


def test_unselected_or_changed_models_cannot_read_test(setup):
    folder, plan, calls = setup
    ev.freeze_plan([folder.name], plan)
    (folder / "model.pt").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="changed after selection"):
        ev.evaluate(folder.name, plan)
    assert not calls
    assert not (folder / "test_started.json").exists()


def test_test_is_evaluated_once_and_frozen_plan_cannot_change(setup):
    folder, plan, calls = setup
    ev.freeze_plan([folder.name], plan)
    assert ev.evaluate(folder.name, plan)["macro_f1"] == 1
    assert calls == [True]
    with pytest.raises(ValueError, match="already evaluated or started"):
        ev.evaluate(folder.name, plan)
    with pytest.raises(ValueError, match="cannot be changed"):
        ev.freeze_plan([], plan)
    assert calls == [True]


def test_dirty_training_run_cannot_be_selected(setup):
    folder, plan, calls = setup
    record = json.loads((folder / "record.json").read_text())
    record["code_version"] = "old-dirty"
    (folder / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="untraceable"):
        ev.freeze_plan([folder.name], plan)
    assert not calls


def test_recorded_stage5_config_gets_its_integer_size_keys_back():
    """record.json stores size tables with string keys; evaluation must look them up by integer size."""
    from src.config import load_config

    config = load_config("configs/wholeimage/swin_tiny_128.yaml")
    recorded = json.loads(json.dumps({**config, "lr": 3e-5, "round": 1}))
    assert "224" in recorded["wholeimage"]["batch_size"]  # the failure of the first deadline evaluation
    restored = ev.restore_size_keys(recorded)
    whole = restored["wholeimage"]
    assert whole["batch_size"] == config["wholeimage"]["batch_size"]
    assert whole["models"]["swin_tiny"]["size_kwargs"][128] == {"img_size": 128, "window_size": 4}
    assert whole["models"] == config["wholeimage"]["models"]
    assert "224" in recorded["wholeimage"]["batch_size"]  # the record itself is not modified
    assert ev.restore_size_keys(restored) == restored
