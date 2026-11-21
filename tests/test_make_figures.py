"""Stage 14 figures: built from the same records, pending (skipped) when a source is missing."""

import json

import pytest
from final_fixture import add_photos, make_fixture

from src import make_figures as mf
from src import make_tables as mt


def status(root):
    return {n: f["status"] for n, f in json.loads((root / "results/figures/final_figures.json").read_text())["figures"].items()}


def test_the_session_figures_are_built_and_the_rest_are_pending(tmp_path):
    root = make_fixture(tmp_path)
    mf.build_all(root)
    assert status(root) == {"samples": "pending", "confusion": "present", "partition_order": "present", "ablations": "present",
                            "paddy_zoom": "pending", "deletion": "pending", "heatmaps": "pending", "cost": "pending"}
    for stem in ("final_confusion", "final_partition_order", "final_ablations"):
        for ext in ("png", "pdf"):
            assert (root / f"results/figures/{stem}.{ext}").stat().st_size > 1000
    for stem in ("final_samples", "final_paddy_zoom", "final_deletion", "final_heatmaps", "final_cost"):
        assert not (root / f"results/figures/{stem}.png").exists()     # skipped, not an empty file
    reasons = json.loads((root / "results/figures/final_figures.json").read_text())["figures"]
    assert "photos exist only in the main checkout" in reasons["samples"]["reason"]


def test_later_stage_figures_appear_when_their_outputs_exist(tmp_path):
    root = make_fixture(tmp_path, paddy=True, explain=True, cost=True)
    add_photos(root)
    mf.build_all(root)
    assert set(status(root).values()) == {"present"}


def test_a_cost_table_without_the_needed_columns_is_pending_with_the_reason(tmp_path):
    root = make_fixture(tmp_path)
    (root / "results/tables/cost_results.csv").write_text("method,params\nattention,260000\n")
    mf.build_all(root)
    figures = json.loads((root / "results/figures/final_figures.json").read_text())["figures"]
    assert figures["cost"]["status"] == "pending" and "seconds_per_photo" in figures["cost"]["reason"]


def test_figures_follow_their_sources(tmp_path):
    root = make_fixture(tmp_path)
    mf.build_all(root)
    first = json.loads((root / "results/figures/final_figures.json").read_text())["figures"]["partition_order"]["sources"]
    path = root / "results/tables/partition_order_inflation.csv"
    path.write_text(path.read_text().replace("0.75", "0.74"))
    mf.build_all(root)
    second = json.loads((root / "results/figures/final_figures.json").read_text())["figures"]["partition_order"]["sources"]
    assert first != second and list(first) == list(second)           # the recorded checksum changed with the data


def test_a_run_outside_its_frozen_plan_stops_the_confusion_figure(tmp_path):
    root = make_fixture(tmp_path)
    plan = root / "runs/session_evaluation_plan.json"
    data = json.loads(plan.read_text())
    data["runs"].pop("stage7_session_attention_r1")
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="must not be read"):
        mf.build_all(root)


def test_numbers_helper_reads_signed_cells():
    assert mf.numbers("[3.8, 6.7]") == [3.8, 6.7] and mf.numbers("-12.0") == [-12.0] and mf.numbers("+0.5 [0.0, 1.0]") == [0.5, 0.0, 1.0]


def test_the_figure_list_matches_the_promised_list():
    assert [s.name for s in mf.SPECS] == list(mt.PROMISED_FIGURES)


def test_the_real_session_figures_build(tmp_path):
    out = tmp_path / "figures"
    report = mf.build_all(mt.REPO_ROOT, out)["figures"]
    assert report["confusion"]["status"] == report["partition_order"]["status"] == report["ablations"]["status"] == "present"
    assert (out / "final_confusion.pdf").exists()


def test_an_older_figure_is_removed_when_its_source_goes_missing(tmp_path):
    root = make_fixture(tmp_path)
    mf.build_all(root)
    assert (root / "results/figures/final_confusion.png").exists()
    (root / "runs/session_conditions_selection.json").write_text(json.dumps({"configurations": {}}))
    (root / "runs/session_selection.json").unlink()
    mf.build_all(root)
    assert not (root / "results/figures/final_confusion.png").exists() and status(root)["confusion"] == "pending"
