"""An interrupted Optuna search resumes the sampled parameters and already scored validation folds."""

import collections
import copy

import pandas as pd
import pytest

from src import records
from src import train_mil as tm
from src.config import load_config


def test_interrupted_search_reuses_fold_scores_and_rejects_changed_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "get_code_version", lambda: "abc123")
    monkeypatch.setattr(records, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(tm, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(tm, "completed_cache", lambda *args: {"spec_sha256": "features"})
    monkeypatch.setattr(tm, "read_folds", lambda c: (pd.DataFrame(), "split-sha"))
    monkeypatch.setattr(tm, "store_for", lambda *args: object())
    config = load_config("configs/mil/attention.yaml")
    config["search"]["trials"] = 2
    calls = []

    def fit(store, folds, r, params, config, seed):
        calls.append((params["lr"], r))
        if len(calls) == 2:
            raise RuntimeError("simulated interruption before a fold finishes")
        return {"f1": 0.5 + r * 0.01, "epoch": 1}

    monkeypatch.setattr(tm, "fit", fit)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        tm.tune(config)
    result = tm.tune(config)
    assert result["trials"] == 2
    assert len(calls) == 11  # ten completed folds and the one interrupted attempt
    counts = collections.Counter(calls)
    assert counts[calls[0]] == 1
    assert counts[calls[1]] == 2
    assert tm.tune(config) == result
    assert len(calls) == 11
    changed = copy.deepcopy(config)
    changed["search"]["trials"] = 3
    with pytest.raises(ValueError, match="different code, settings or split"):
        tm.tune(changed)
