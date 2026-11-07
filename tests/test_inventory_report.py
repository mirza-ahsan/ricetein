"""Checks for src/inventory_report.py: helpers, and every section on small or awkward inventories."""

import pandas as pd

from src import inventory_report as rep
from src.config import load_config

CONFIG = load_config("configs/inventory.yaml")


def test_md_table_renders_rows_and_shows_missing_values_as_a_dash():
    text = rep.md_table(pd.DataFrame({"a": [1, None], "b": ["x", None]}))
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1.0 | x |", "| – | – |"]


def test_duplicate_groups_counts_files_and_folders():
    frame = pd.DataFrame({"photo_id": ["A/1", "B/2", "A/3", "A/4"], "folder": ["A", "B", "A", "A"],
                          "sha256": ["s1", "s1", "s2", None]})
    assert rep.duplicate_groups(frame, "sha256").to_dict("index") == {"s1": {"files": 2, "folders": "A + B"}}


def test_duration_is_readable():
    assert rep.duration(pd.Timedelta(minutes=3, seconds=48)) == "3 min"
    assert rep.duration(pd.Timedelta(days=1, hours=2, minutes=14)) == "1 d 2 h 14 min"
    assert rep.duration(pd.Timedelta(seconds=20)) == "under a minute"


def inventory(n_per_class=3, **overrides):
    rows = []
    for label, folder in CONFIG["datasets"]["primary"]["class_folders"].items():
        for i in range(n_per_class):
            rows.append({"photo_id": f"{folder}/{i}.jpg", "folder": folder, "class": label,
                         "file_bytes": 1000 + i, "sha256": f"{folder}{i}",
                         "width": 900 + i, "height": 900 + i,
                         "format": "JPEG", "mode": "RGB", "exif_orientation": 1, "camera": None,
                         "taken_at": None, "jpeg_quality": 95, "jpeg_quality_exact": True,
                         "phash": f"{folder}{i}", "opened_ok": True, "error": None})
    frame = pd.DataFrame(rows).assign(**overrides)
    frame["short_side"] = frame[["width", "height"]].min(axis=1)
    return frame


def test_sections_handle_an_inventory_with_nothing_unusual():
    primary = inventory()
    assert "Every file decodes fully" in rep.integrity_section(primary, primary)
    text = rep.duplicates_section(CONFIG, primary, primary.assign(sha256=primary["sha256"] + "p"))
    assert "0 duplicate groups" in text and "Stage 3" not in text
    camera = rep.camera_section(CONFIG, primary)
    assert "no photo carries camera EXIF" in camera and "no photo carries a capture time" in camera
    assert "Different from the rest" not in camera and "practically uniform" in camera
    assert "0 of them contain more than one class" in rep.phash_section(CONFIG, primary)
    assert "Photos are square" in rep.sizes_section(CONFIG, primary)


def test_sections_report_problems_instead_of_claiming_all_is_well():
    primary = inventory()
    primary.loc[0, ["opened_ok", "error"]] = [False, "OSError: truncated"]
    primary.loc[1, "exif_orientation"] = 6
    text = rep.integrity_section(primary, inventory())
    assert "1 file(s) fail to decode" in text and "OSError: truncated" in text and "rotation tag" in text
    counts = rep.counts_section(CONFIG, primary, copies=(5, 4, 4))
    assert "Warning" in counts and "missing from the inventory" in counts


def test_pixel_scale_section_renders_pending_complete_and_not_applied(tmp_path):
    import copy

    import numpy as np

    from src import leaf_width as lw

    config = copy.deepcopy(CONFIG)
    settings = config["pixel_scale"]
    settings.update(selection_file=str(tmp_path / "selection.csv"), measurements_dir=str(tmp_path / "lw"),
                    measurements_file=str(tmp_path / "all.csv"),
                    decision_file=str(tmp_path / "decision.json"))
    sides = np.random.default_rng(1).integers(700, 3200, 300)
    fake = pd.DataFrame({"photo_id": [f"C/{i}.jpg" for i in range(300)], "class": "healthy",
                         "width": sides, "height": sides, "opened_ok": True})
    selected = lw.select_photos(fake, 30, config["seed"])
    selected.to_csv(tmp_path / "selection.csv", index=False)
    assert "Pending" in rep.pixel_scale_section(config)

    (tmp_path / "lw").mkdir()
    for who in ("ahsan", "harris"):
        rows = pd.DataFrame({"dataset": "primary", "photo_id": selected["photo_id"], "measurer": who,
                             "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.0,
                             "width_px": selected["short_side"] * 0.2,
                             "status": "accept", "measured_at": "2026-11-05T00:00:00+00:00"})[lw.COLUMNS]
        if who == "harris":
            rows.loc[:2, ["x1", "y1", "x2", "y2", "width_px"]] = np.nan  # three photos unmeasurable
            rows.loc[:2, "status"] = "unmeasurable"
        rows.to_csv(tmp_path / "lw" / f"{who}.csv", index=False)
    text = rep.pixel_scale_section(config)
    assert "rescale every photo" in text and "3 photos marked unmeasurable" in text and "| – |" in text
    assert "rho from each measurer alone: ahsan 1.000, harris 1.000" in text and "Model rater" not in text
    assert "0 of 27 single removals would flip" in text
    assert not (tmp_path / "decision.json").exists()  # the report only reads; `decide` writes

    harris = pd.read_csv(tmp_path / "lw" / "harris.csv")
    harris.loc[:14, "status"] = "unmeasurable"
    harris.to_csv(tmp_path / "lw" / "harris.csv", index=False)
    assert "Rule not applied" in rep.pixel_scale_section(config)
