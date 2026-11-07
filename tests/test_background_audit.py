"""Checks for src/background_audit.py and src/label_check.py: frozen samples, the labelling loops,
settling, kappa and the suspects list (fake answers; no screen needed)."""

import pandas as pd
import pytest
from PIL import Image

from src import background_audit as bg
from src import label_check as lc
from src import review_tool

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]


def inventory(per_class=30):
    rows = [{"photo_id": f"{c}/{i:03d}.jpg", "class": c, "opened_ok": True}
            for c in CLASSES for i in range(per_class)]
    return pd.DataFrame(rows)


def test_sample_is_stratified_reproducible_and_ignores_row_order():
    first = bg.draw_sample(inventory(), 5, seed=0)
    again = bg.draw_sample(inventory().sample(frac=1, random_state=1), 5, seed=0)
    assert first.equals(again) and len(first) == 20 and first["photo_id"].is_unique
    assert first["class"].value_counts().to_dict() == {c: 5 for c in CLASSES}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for module in (bg, lc):
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(module, "read_inventory", lambda path: inventory())
    monkeypatch.setattr(bg, "dataset_dir", lambda config, name: tmp_path / "photos")
    (tmp_path / "photos").mkdir()
    for photo_id in inventory()["photo_id"]:
        (tmp_path / "photos" / photo_id).parent.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(tmp_path / "photos" / photo_id)
    config = {"seed": 0, "classes": CLASSES, "reviewers": ["ahsan", "harris", "umer"],
              "inventory": {"files": {"primary": "unused"}},
              "background": {"per_class": 3, "sample_file": "sample.csv", "labels_dir": "bg",
                             "audit_file": "audit.csv"},
              "label_check": {"per_class": 2, "checks_dir": "checks", "suspects_file": "suspects.csv"}}
    return config, tmp_path


def fake(answers):
    shown = []

    def ask(images, title, choices, toggles=None):
        shown.append(title)
        return next(answers)
    return ask, shown


def test_labelling_saves_resumes_and_results_need_settling(setup):
    config, tmp = setup
    ask, _ = fake(iter([("plain", {"hand"}), (review_tool.STOP, set())]))
    bg.label(config, "ahsan", ask=ask)
    saved = pd.read_csv(tmp / "bg" / "ahsan.csv")
    assert len(saved) == 1 and bool(saved["hand"].iloc[0]) and not bool(saved["strong_shadow"].iloc[0])

    bg.label(config, "ahsan", ask=fake(iter([("plain", set())] * 11))[0])  # resumes: 11 left of 12
    labels = iter([("cluttered", {"multiple_leaves"})] + [("plain", set())] * 11)
    bg.label(config, "umer", ask=fake(labels)[0])
    result = bg.results(config)
    assert result["complete"] is False and result["unsettled"] == 1

    ask, shown = fake(iter([("cluttered", set())]))
    bg.settle(config, ask=ask)
    assert len(shown) == 1  # only the one disagreement
    result = bg.results(config)
    assert result["complete"] and result["n_settled"] == 1 and result["plain"]["n"] == 12
    assert result["plain"]["share"] == pytest.approx(11 / 12)
    assert result["plain"]["low"] < 11 / 12 < result["plain"]["high"]
    assert sum(v["n"] for v in result["plain_by_class"].values()) == 12
    assert "background_ahsan" in result["table"].columns and not (tmp / "audit.csv").exists()  # reads only


def test_a_third_rater_is_stopped_before_starting(setup):
    config, tmp = setup
    for who in ("ahsan", "harris"):
        bg.label(config, who, ask=fake(iter([("plain", set()), (review_tool.STOP, set())]))[0])
    with pytest.raises(SystemExit, match="already"):
        bg.label(config, "umer", ask=fake(iter([]))[0])
    pd.DataFrame([{"photo_id": "x", "rater": "bob", "background": "plain"}]).to_csv(tmp / "bg" / "bob.csv")
    with pytest.raises(ValueError, match="raters"):  # an unknown name in the folder is refused too
        bg.results(config)


def test_label_check_lists_only_answers_naming_another_class(setup):
    config, tmp = setup
    keys = lc.choices(CLASSES)
    assert keys["1"] == "looks_healthy" and keys["k"] == "looks_right"
    for who in ("ahsan", "harris"):
        lc.check(config, who, ask=fake(iter([("looks_right", set())] * 8))[0])
    assert lc.results(config)["complete"] is False  # umer has not checked yet
    lc.check(config, "umer", ask=fake(iter([("looks_hispa", set())] * 8))[0])
    result = lc.results(config)
    assert result["complete"] and result["checked"] == 24
    assert len(result["suspects"]) == 6  # "looks like hispa" on hispa photos is not a suspect
    assert set(result["suspects"]["looks_like"]) == {"hispa"} and not (tmp / "suspects.csv").exists()
