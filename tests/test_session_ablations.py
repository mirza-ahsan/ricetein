"""Checks for src/session_ablations.py: each variant changes exactly one thing, no variant gets tuning, only the
headline variants are tested, and the validation table is computed from run metrics (no GPU, no data)."""

import json

import pytest

from src import session_ablations as sa
from src.config import load_config


def test_every_variant_changes_only_what_it_names():
    main = load_config(sa.MAIN)["mil"]
    for name, spec in sa.VARIANTS.items():
        mil = sa.variant_config(name)["mil"]
        assert mil["name"] == f"session_ablate_{name}" and mil["tuned_from"] == main["tuned_from"]
        changed = {k for k in {*main, *mil} if main.get(k) != mil.get(k)} - {"name"}
        assert changed <= {"cache", "override", "class_weighting", "versions"}, (name, changed)
        assert mil.get("cache", main["cache"]) == spec.get("cache", main["cache"])


def test_the_selected_settings_are_reused_not_searched_again():
    for name in sa.VARIANTS:
        config = sa.variant_config(name)
        assert config["mil"]["tuned_from"] == "session_attention"     # train_mil.final reads this study's best parameters
    assert sa.variant_config("description_m")["mil"]["override"] == {"descriptor": "m"}
    assert sa.variant_config("unweighted_loss")["mil"]["class_weighting"] is False
    assert sa.variant_config("single_version")["mil"]["versions"] == 1
    assert sa.variant_config("encoder_e2")["mil"]["override"] == {"descriptor": "c"}   # E2 has no patch average


def test_a_variant_whose_cache_is_not_complete_is_skipped_not_trained(monkeypatch):
    trained = []
    monkeypatch.setattr(sa.train_mil, "final", lambda config, seeds: trained.append(config["mil"]["name"]))

    def completed(config, name):
        if name in ("e1_otsu", "e3_dinov2_vitb14"):
            raise FileNotFoundError(name)

    monkeypatch.setattr(sa, "completed_cache", completed)
    sa.fit()
    assert "session_ablate_otsu_on" not in trained and "session_ablate_encoder_e3" not in trained
    assert "session_ablate_unweighted_loss" in trained and len(trained) == len(sa.VARIANTS) - 2


def test_only_the_headline_variants_are_sent_to_test(monkeypatch):
    seen = {}
    monkeypatch.setattr(sa, "evaluate_once", lambda runs, plan: seen.update(runs=runs, plan=plan))
    sa.evaluate_headline()
    assert sorted(seen["runs"]) == sorted(sa.runs_of("otsu_on") + sa.runs_of("native_scale"))
    assert seen["plan"] == "session_ablations_plan.json"


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "get_default_runs_dir", lambda: tmp_path)
    main = [f"main_r{r}" for r in range(1, 6)]
    monkeypatch.setattr(sa, "main_runs", lambda: main)

    def write(run, scores):
        (tmp_path / run).mkdir()
        (tmp_path / run / "metrics.json").write_text(json.dumps({"val_macro_f1": scores}))

    for r, run in enumerate(main):
        write(run, 0.80 + 0.01 * r)
    for r, run in enumerate(sa.runs_of("otsu_on")):
        write(run, 0.79 + 0.01 * r)         # one point below main in every round
    for r, run in enumerate(sa.runs_of("tile448")):
        write(run, 0.81 + 0.01 * r)         # one point above main in every round
    return tmp_path


def test_validation_table_reports_the_difference_and_the_rounds_better_than_main(runs):
    rows = {r["variant"]: r for r in sa.validation_rows(["otsu_on", "tile448"])}
    assert rows["otsu_on"]["difference_vs_main"] == pytest.approx(-0.01) and rows["otsu_on"]["rounds_better_than_main"] == 0
    assert rows["tile448"]["difference_vs_main"] == pytest.approx(0.01) and rows["tile448"]["rounds_better_than_main"] == 5
    assert rows["main"]["val_macro_f1_mean"] == pytest.approx(0.82)
    assert sa.trained() == ["otsu_on", "tile448"]
