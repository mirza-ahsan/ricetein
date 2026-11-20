"""Checks for src/session_ablations.py: each variant changes only what it names, no variant gets tuning, the packed
files are built one kind at a time, a queue never ends as DONE with variants skipped, only the headline variants
are tested (and only when all are trained), and the validation table is computed from run metrics (no GPU, no data)."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src import session_ablations as sa
from src.config import load_config


def test_every_variant_changes_only_what_it_names_and_reuses_the_main_settings():
    main = load_config(sa.MAIN)["mil"]
    for name, spec in sa.VARIANTS.items():
        mil = sa.variant_config(name)["mil"]
        assert mil["name"] == f"session_ablate_{name}"
        assert mil["tuned_from"] == main["tuned_from"] == "session_attention"   # train_mil.final reads this study's parameters
        changed = {k for k in {*main, *mil} if main.get(k) != mil.get(k)} - {"name"}
        assert changed == set(spec["mil"]), (name, changed)


def test_the_main_run_names_follow_what_train_mil_writes():
    assert sa.runs_of("otsu_on")[0] == "stage7_session_ablate_otsu_on_r1" and len(sa.runs_of("otsu_on")) == 5
    assert sa.HEADLINE == ["otsu_on", "native_scale"]


@pytest.fixture
def tmp_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(sa, "cache_folder", lambda name, write=False: tmp_path / "cache" / name)
    return tmp_path


def test_the_main_configuration_must_be_the_declared_study(tmp_runs):
    (tmp_runs / "session_selection.json").write_text(
        json.dumps({"methods": {"attention": {"selected": "wide", "runs": ["x"]}}}))
    with pytest.raises(ValueError, match="wide study"):
        sa.main_runs()


def test_packed_files_are_built_one_kind_at_a_time_for_training_and_for_test(tmp_runs, monkeypatch):
    (tmp_runs / "stage7_tune_session_attention").mkdir()
    (tmp_runs / "stage7_tune_session_attention" / "metrics.json").write_text(json.dumps({"best_params": {"descriptor": "cm"}}))
    calls = []
    monkeypatch.setattr(sa, "read_folds", lambda config: (SimpleNamespace(photo_id=["b", "a"]), "sum"))
    monkeypatch.setattr(sa, "completed_cache", lambda config, name: {"spec_sha256": "abc"})
    monkeypatch.setattr(sa.train_mil, "pack", lambda folder, target, key, ids, versions, fp: calls.append((key, versions, ids)))
    sa.pack_one_key_at_a_time(sa.variant_config("otsu_on"))                      # description cm, 8 versions
    assert sorted((k, v) for k, v, _ in calls) == [("cls", 1), ("cls", 8), ("mean", 1), ("mean", 8)]
    assert all(ids == ["a", "b"] for _, _, ids in calls)
    calls.clear()
    sa.pack_one_key_at_a_time(sa.variant_config("description_c"))                # class token only: one kind
    assert {k for k, _, _ in calls} == {"cls"}
    calls.clear()
    sa.pack_one_key_at_a_time(sa.variant_config("single_version"))               # one version for training and test
    assert {v for _, v, _ in calls} == {1}


def test_fit_trains_the_ready_variants_and_ends_with_an_error_for_the_waiting_ones(tmp_runs, monkeypatch):
    trained = []
    for ready in ("e1_dinov3_vitb16",):
        (tmp_runs / "cache" / ready).mkdir(parents=True)
        (tmp_runs / "cache" / ready / "complete.json").write_text("{}")
    monkeypatch.setattr(sa, "pack_one_key_at_a_time", lambda config: None)
    monkeypatch.setattr(sa.train_mil, "final", lambda config, seeds: trained.append(config["mil"]["name"]))
    with pytest.raises(SystemExit, match="NOT DONE"):
        sa.fit()
    assert "session_ablate_unweighted_loss" in trained and "session_ablate_otsu_on" not in trained
    trained.clear()
    sa.fit(["single_version", "description_m"])                                  # a shard whose caches are all ready
    assert trained == ["session_ablate_single_version", "session_ablate_description_m"]


def test_the_headline_variants_are_tested_only_when_all_of_them_are_trained(monkeypatch):
    seen = {}
    monkeypatch.setattr(sa, "evaluate_once", lambda runs, plan: seen.update(runs=runs, plan=plan))
    monkeypatch.setattr(sa, "trained", lambda: ["otsu_on"])
    with pytest.raises(RuntimeError, match="train every headline variant"):
        sa.evaluate_headline()
    monkeypatch.setattr(sa, "trained", lambda: list(sa.VARIANTS))
    sa.evaluate_headline()
    assert sorted(seen["runs"]) == sorted(sa.runs_of("otsu_on") + sa.runs_of("native_scale"))
    assert seen["plan"] == "session_ablations_plan.json"


def test_validation_table_has_the_noise_floor_and_the_rounds_better_than_main(tmp_runs, monkeypatch):
    main = [f"main_r{r}" for r in range(1, 6)]
    monkeypatch.setattr(sa, "main_runs", lambda: main)
    monkeypatch.setattr(sa, "completed_run", lambda run, files: (tmp_runs / run / "metrics.json").exists())

    def write(run, score):
        (tmp_runs / run).mkdir()
        (tmp_runs / run / "metrics.json").write_text(json.dumps({"val_macro_f1": score}))

    for r in range(5):
        write(main[r], 0.80 + 0.01 * r)
        write(f"main_s1_r{r + 1}", 0.79 + 0.01 * r)
        write(f"main_s2_r{r + 1}", 0.81 + 0.01 * r)
        write(sa.runs_of("otsu_on")[r], 0.79 + 0.01 * r)           # one point below main in every round
        write(sa.runs_of("tile448")[r], 0.81 + 0.01 * r)           # one point above main in every round
    rows = {r["variant"]: r for r in sa.validation_rows()}
    assert list(rows) == ["main", "main, seeds 1 and 2", "otsu_on", "tile448"]
    assert rows["main"]["val_macro_f1_mean"] == pytest.approx(0.82) and rows["main"]["rounds_better_than_main"] == "–"
    assert rows["main, seeds 1 and 2"]["difference_vs_main"] == pytest.approx(0.0)
    assert rows["otsu_on"]["difference_vs_main"] == pytest.approx(-0.01) and rows["otsu_on"]["rounds_better_than_main"] == 0
    assert rows["tile448"]["difference_vs_main"] == pytest.approx(0.01) and rows["tile448"]["rounds_better_than_main"] == 5


def test_the_reading_says_who_is_better_from_the_range_and_the_round_wins():
    row = lambda low, high, wins: pd.DataFrame([{"photo_low": low, "photo_high": high, "round_wins": wins}]).itertuples().__next__()  # noqa: E731
    assert sa.reading(row(0.01, 0.05, 5)) == "main beats variant"
    assert sa.reading(row(-0.06, -0.01, 0)) == "variant beats main"        # a variant clearly better than main
    assert sa.reading(row(0.01, 0.05, 3)) == "no clear difference"        # range above 0 but too few round wins
    assert sa.reading(row(-0.02, 0.03, 2)) == "no clear difference"
