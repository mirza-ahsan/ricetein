"""Checks for src/cost.py (Stage 13): the timing harness (warm-up discarded, median), parameter counts, the declared
rule for the final model, the one-time test marker, the column contract of the cost table, and the run lists against
the frozen selections (fake models and fake runs; no GPU, no photos)."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import compress, cost
from src.config import load_config
from src.mil_models import MILModel

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
# what the cost chart of src/make_figures.py (branch stage14/tooling) reads from results/tables/cost_results.csv
COST_COLUMNS = ("method", "seconds_per_photo", "macro_f1")


# ---------------------------------------------------------------------------------------------- timing harness
def test_warm_up_photos_are_discarded_and_the_median_is_over_the_timed_photos():
    now = {"t": 0.0}
    seconds = {"w1": 1000.0, "w2": 1000.0, "a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 100.0}
    order, syncs = [], []

    def run(path, laps):
        order.append(path)
        now["t"] += seconds[path] / 2
        laps.mark("prepare")
        now["t"] += seconds[path] / 2
        laps.mark("decide")

    rows = cost.time_photos(run, ["a", "b", "c", "d", "e"], ["w1", "w2"], lambda: syncs.append(1), clock=lambda: now["t"])
    assert order == ["w1", "w2", "a", "b", "c", "d", "e"] and len(rows) == 5
    assert [r["total"] for r in rows] == [1.0, 2.0, 3.0, 4.0, 100.0]
    summary = cost.summarise(rows)
    assert summary["median"] == 3.0 and summary["q1"] == 2.0 and summary["q3"] == 4.0 and summary["n"] == 5
    assert summary["median_prepare"] == 1.5 and summary["median_decide"] == 1.5
    assert len(syncs) >= 2 * 7 * 2 // 2    # the GPU is synchronised at every reading


def test_the_100_timing_photos_are_a_seeded_draw_of_training_photos_with_5_distinct_warm_ups(monkeypatch):
    folds = pd.DataFrame({"photo_id": [f"p{k:04d}" for k in range(500)], "class": ["healthy"] * 500,
                          "round_1": ["train"] * 300 + ["val"] * 100 + ["test"] * 100})
    monkeypatch.setattr(cost, "read_folds", lambda config: (folds, "sum"))
    config = load_config("configs/cost.yaml")
    timed, warm = cost.timing_photos(config)
    assert len(timed) == 100 and len(warm) == 5 and len(set(timed + warm)) == 105
    assert set(timed + warm) <= set(folds["photo_id"][:300])
    assert (timed, warm) == cost.timing_photos(config)


def test_a_busy_machine_is_refused():
    quiet = {"max_process_cpu_percent": 15, "max_total_cpu_percent": 20, "max_gpu_mib": 1500}
    calm = {"total_cpu_percent": 3.0, "busy_processes": [{"pid": 1, "name": "hyprland", "cpu_percent": 2.0}], "gpu_mib_used": 400}
    cost.require_quiet(calm, quiet, "gpu")
    for change in ({"gpu_mib_used": 4000}, {"total_cpu_percent": 60.0},
                   {"busy_processes": [{"pid": 7, "name": "python", "cpu_percent": 90.0}]}):
        with pytest.raises(RuntimeError, match="not quiet"):
            cost.require_quiet({**calm, **change}, quiet, "cpu")


# ---------------------------------------------------------------------------------------------- sizes
def test_parameter_count():
    assert cost.count(torch.nn.Linear(3, 2).parameters()) == 8
    assert cost.count(MILModel(8, 4, 0.0, "attention", 4).state_dict().values()) == (
        8 * 4 + 4 + 2 * (4 * 128 + 128) + 128 + 4 * 4 + 4)


@pytest.fixture
def fake_runs(tmp_path, monkeypatch):
    for module in (cost, compress):
        monkeypatch.setattr(module, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr("src.records.get_default_runs_dir", lambda: tmp_path)
    return tmp_path


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_sizes_of_a_tile_method_and_of_a_probe(fake_runs, monkeypatch):
    head = MILModel(8, 4, 0.0, "attention", 4)
    write(fake_runs / "session_selection.json", {"methods": {k: {"runs": [f"{k}_r{r}" for r in range(1, 6)]}
                                                              for k in ("attention", "probe_224")}})
    (fake_runs / "attention_r1").mkdir()
    torch.save(head.state_dict(), fake_runs / "attention_r1" / "model.pt")
    x = np.random.default_rng(0).standard_normal((40, 6))
    probe = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, np.arange(40) % 4)
    (fake_runs / "probe_224_r1").mkdir()
    (fake_runs / "probe_224_r1" / "model.pkl").write_bytes(pickle.dumps(probe))
    cache = fake_runs / "cache"
    cache.mkdir()
    (cache / "v0.npz").write_bytes(b"x" * 1000)
    monkeypatch.setattr(cost, "load_encoder", lambda config, code, device: torch.nn.Linear(10, 10))
    monkeypatch.setattr(cost, "encoder_file_bytes", lambda config, code: 5000)
    monkeypatch.setattr(cost, "cache_folder", lambda name, write=False: cache)
    config = load_config("configs/cost.yaml")
    row = cost.size_row("attention", config)
    assert row["params_encoder"] == 110 and row["params_decider"] == cost.count(head.state_dict().values())
    assert row["disk_bytes"] == 5000 + (fake_runs / "attention_r1" / "model.pt").stat().st_size and row["cache_bytes"] == 1000
    probe_row = cost.size_row("probe_224", config)
    assert probe_row["params_decider"] == 6 + 6 + 4 * 6 + 4      # scaler mean and scale, coefficients, intercepts


# ---------------------------------------------------------------------------------------------- the final model
ORDER = compress.CANDIDATES


def test_final_model_rule():
    val = {"e1_top16": 0.76, "e4": 0.74, "e4_top16": 0.73, "distilled_e4": 0.755, "distilled_e4_top16": 0.74}
    seconds = {"e1_top16": 1.0, "e4": 0.8, "e4_top16": 0.2, "distilled_e4": 0.8, "distilled_e4_top16": 0.2}
    # reference 0.77, margin 2.0 points: eligible are 0.76 and 0.755 (0.74 is 3 points below): the faster of those wins
    assert cost.choose_final(val, seconds, 0.77, 5.0, 2.0, ORDER)[0] == "distilled_e4"
    # equal time: the higher validation macro-F1 wins; equal on both: the first in the declared order
    equal_time = {**seconds, "e1_top16": 0.8}
    assert cost.choose_final(val, equal_time, 0.77, 5.0, 2.0, ORDER)[0] == "e1_top16"
    assert cost.choose_final({**val, "distilled_e4": 0.76}, equal_time, 0.77, 5.0, 2.0, ORDER)[0] == "e1_top16"
    # none within the margin: the best validation macro-F1 among those faster than the reference
    chosen, why = cost.choose_final(val, {**seconds, "e1_top16": 9.0}, 0.90, 5.0, 2.0, ORDER)
    assert chosen == "distilled_e4" and "none within the margin" in why
    with pytest.raises(RuntimeError, match="faster"):
        cost.choose_final(val, {n: 9.0 for n in ORDER}, 0.90, 5.0, 2.0, ORDER)


def make_final_fixture(fake_runs, monkeypatch, selected="e4"):
    config = load_config("configs/cost.yaml")
    write(fake_runs / "cost_selection.json", {"partition": "val", "selected": selected, "reason": "test"})
    for run in compress.head_runs(compress.VARIANTS[selected][0]):
        write(fake_runs / run / "record.json", {"status": "completed"})
        (fake_runs / run / "model.pt").write_bytes(run.encode())
    folds = pd.DataFrame({"photo_id": [str(k) for k in range(8)], "class": CLASSES * 2,
                          **{f"round_{r}": ["train"] * 4 + ["val"] * 2 + ["test"] * 2 for r in range(1, 6)}})
    val = pd.DataFrame({"round": 1, "partition": "val", "photo_id": ["4", "5"], "true_class": CLASSES[:2], "pred_class": CLASSES[:2]})
    (fake_runs / "cost_fewer_tiles").mkdir()
    val.to_csv(fake_runs / "cost_fewer_tiles" / f"predictions_{selected}.csv", index=False)
    seen = {}

    def scores(config_, names, folds_, role, selection, rows):
        seen.update(names=names, role=role, rows={r: folds_["photo_id"].to_numpy()[v].tolist() for r, v in rows.items()})
        table = pd.DataFrame({"round": 1, "partition": role, "photo_id": ["6", "7"], "true_class": CLASSES[2:], "pred_class": CLASSES[2:]})
        return {names[0]: {"scores": [1.0] * 5, "predictions": table, "tiles_used": 79.0}}

    monkeypatch.setattr(cost, "require_clean_code", lambda: "abc1234")
    monkeypatch.setattr(cost, "read_folds", lambda config_: (folds, "checksum"))
    monkeypatch.setattr("src.records.get_code_version", lambda: "abc1234")
    monkeypatch.setattr(compress, "variant_scores", scores)
    monkeypatch.setattr(compress, "tile_selection", lambda config_, photos, workers=8: pytest.fail("not a top16 model"))
    return config, seen


def test_the_final_model_is_tested_once_and_only_on_test_photos(fake_runs, monkeypatch):
    config, seen = make_final_fixture(fake_runs, monkeypatch)
    cost.plan(config)
    cost.plan(config)                                              # repeating the plan changes nothing
    assert not (fake_runs / "cost_final" / "test_metrics.json").exists() and (fake_runs / "cost_final" / "predictions.csv").exists()
    cost.evaluate_final(config)
    assert seen["role"] == "test" and set(sum(seen["rows"].values(), [])) == {"6", "7"}
    assert (fake_runs / "cost_final" / "test_started.json").exists() and (fake_runs / "cost_final" / "test_predictions.csv").exists()
    first = (fake_runs / "cost_final" / "test_metrics.json").read_text()
    with pytest.raises(RuntimeError, match="no retry"):
        cost.evaluate_final(config)
    assert (fake_runs / "cost_final" / "test_metrics.json").read_text() == first


def test_a_round_model_changed_after_the_plan_stops_the_test_without_using_it_up(fake_runs, monkeypatch):
    config, _ = make_final_fixture(fake_runs, monkeypatch)
    cost.plan(config)
    (fake_runs / "stage7_session_ablate_encoder_e4_r3" / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after the plan"):
        cost.evaluate_final(config)
    assert not (fake_runs / "cost_final" / "test_started.json").exists()


def test_the_test_needs_the_frozen_plan_and_the_plan_cannot_change(fake_runs, monkeypatch):
    config, _ = make_final_fixture(fake_runs, monkeypatch)
    with pytest.raises(FileNotFoundError):
        cost.evaluate_final(config)
    cost.plan(config)
    write(fake_runs / "cost_selection.json", {"partition": "val", "selected": "e4", "reason": "other"})
    with pytest.raises(ValueError, match="cannot be changed"):
        cost.plan(config)


# ---------------------------------------------------------------------------------------------- the cost table
def test_cost_table_has_the_columns_of_the_chart_and_the_committee_is_a_sum(fake_runs, monkeypatch):
    config = load_config("configs/cost.yaml")
    config["cost"]["committee"] = {"members": ["attention", "mean"], "test_macro_f1": 0.8}
    write(fake_runs / "session_selection.json", {"methods": {m: {"runs": [f"{m}_r{r}" for r in range(1, 6)]}
                                                              for m in cost.METHODS if m != "convnext_tiny_448"}})
    write(fake_runs / "session_conditions_selection.json",
          {"configurations": {"convnext_tiny_448": [f"c448_r{r}" for r in range(1, 6)]}})
    for m in cost.METHODS:
        for run in compress.selected_runs(m):
            write(fake_runs / run / "test_metrics.json", {"macro_f1": 0.7})
    names = [*cost.METHODS, "e1_top16", "e4", "e4_top16", "distilled_e4", "attention_int8"]
    write(fake_runs / "cost_sizes" / "metrics.json", {"rows": [
        {"method": m, "params_encoder": 86e6, "params_decider": 2.6e5, "disk_bytes": 3e8, "cache_bytes": 6e9} for m in names]})
    timed = list(cost.TIMED)
    for device, base in (("gpu", 0.1), ("cpu", 2.0)):
        write(fake_runs / f"cost_time_{device}" / "metrics.json",
              {"median_seconds_per_photo": {m: base * (k + 1) for k, m in enumerate(timed)},
               "peak_gpu_mib": {m: 900.0 for m in timed}, "peak_rss_mib": {m: 3000.0 for m in timed}})
    write(fake_runs / "cost_plan.json", {"selected": "e4_top16"})
    write(fake_runs / "cost_final" / "test_metrics.json", {"macro_f1_mean": 0.74})
    table = cost.method_rows(config)
    assert set(COST_COLUMNS) <= set(table.columns)
    assert table["macro_f1"].between(0, 1).all() and table["seconds_per_photo"].gt(0).all()
    committee = table[table["method"] == "committee"].iloc[0]
    parts = table[table["method"].isin(["attention", "mean"])]
    assert committee["seconds_per_photo"] == pytest.approx(parts["seconds_per_photo"].sum())
    assert committee["peak_gpu_mb"] == parts["peak_gpu_mb"].max() and committee["macro_f1"] == 0.8
    assert table["method"].iloc[-1].startswith("final_small_model (e4_top16)") and table["macro_f1"].iloc[-1] == 0.74


# ---------------------------------------------------------------------------------------------- the run lists
def test_every_main_method_resolves_to_five_recorded_runs_with_test_scores():
    for method in cost.METHODS:
        runs = compress.selected_runs(method)
        assert len(runs) == 5 and [r[-3:] for r in runs] == [f"_r{k}" for k in range(1, 6)], method
        for run in runs:
            assert (cost.get_default_runs_dir() / run / "record.json").exists(), run
            assert 0 < json.loads((cost.get_default_runs_dir() / run / "test_metrics.json").read_text())["macro_f1"] < 1
    assert compress.selected_runs("attention")[0] == "stage7_session_attention_r1"
    assert compress.selected_runs("convnext_tiny_448")[0] == "stage5_convnext_tiny_session_448_lr0.0003_r1"


def test_the_e4_runs_exist_and_use_the_main_settings():
    main = json.loads((cost.get_default_runs_dir() / compress.selected_runs("attention")[0] / "record.json").read_text())["config"]
    for run in compress.head_runs("e4"):
        record = json.loads((cost.get_default_runs_dir() / run / "record.json").read_text())["config"]
        assert record["params"] == main["params"] and record["mil"]["aggregator"] == "attention"
        assert record["mil"]["cache"] == "e4_dinov3_vits16" and record["split_files"] == main["split_files"]


def test_the_distilled_head_is_timed_as_the_e4_head():
    assert cost.TIMING_KEY["distilled_e4"] == "e4" and cost.TIMING_KEY["distilled_e4_top16"] == "e4_top16"
    assert set(cost.TIMING_KEY) == set(compress.CANDIDATES)
