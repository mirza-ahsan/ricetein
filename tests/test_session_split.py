"""Capture blocks and the session-blocked split: whole blocks stay on one side."""

import numpy as np
import pandas as pd

from src import session_split as ss
from src.split import check_folds, make_folds, round_column


def photos(rows):
    return pd.DataFrame(rows, columns=["photo_id", "camera", "taken_at", "group_id"])


def test_blocks_break_on_long_pauses_camera_changes_and_missing_times():
    table = photos([
        ("a", "redmi", "2019:04:19 10:00:00", "g1"),
        ("b", "redmi", "2019:04:19 10:00:59", "g2"),   # 59 s later: same block
        ("c", "redmi", "2019:04:19 10:02:00", "g3"),   # 61 s later: new block
        ("d", "iphone", "2019:04:19 10:02:01", "g4"),  # another camera: new block
        ("e", "redmi", None, "g5"),                    # no capture time: its own block
        ("f", "redmi", None, "g6"),
    ])
    blocks = ss.capture_blocks(table, 60)
    assert blocks[0] == blocks[1]
    assert len({blocks[0], blocks[2], blocks[3], blocks[4], blocks[5]}) == 5


def test_a_duplicate_group_merges_the_blocks_it_spans():
    table = photos([
        ("a", "redmi", "2019:04:19 10:00:00", "g1"),
        ("b", "redmi", "2019:04:19 12:00:00", "g1"),   # hours later, but a confirmed copy of a
        ("c", "redmi", "2019:04:19 12:00:05", "g2"),
        ("d", "redmi", "2019:04:19 15:00:00", "g3"),
    ])
    blocks = ss.capture_blocks(table, 60)
    assert blocks[0] == blocks[1] == blocks[2] != blocks[3]


def test_block_ids_do_not_depend_on_row_order():
    table = photos([(f"p{i:02d}", "redmi", f"2019:04:19 10:{i:02d}:00", f"g{i}") for i in range(12)])
    shuffled = table.sample(frac=1, random_state=0)
    assert ss.capture_blocks(table, 150).to_dict() == ss.capture_blocks(shuffled, 150).sort_index().to_dict()


def blocked_table():
    rng = np.random.default_rng(0)
    rows = []
    for block in range(60):
        for shot in range(int(rng.integers(3, 12))):
            rows.append((f"b{block:02d}_{shot:02d}", "redmi" if block % 3 else "iphone",
                         f"2019:04:{19 + block // 30} {block % 24:02d}:{shot // 6:02d}:{shot % 6 * 9:02d}",
                         f"g{block}_{shot}", ["healthy", "leaf_blast", "brown_spot", "hispa"][block % 4]))
    table = pd.DataFrame(rows, columns=["photo_id", "camera", "taken_at", "group_id", "class"])
    table["group_id"] = ss.capture_blocks(table, 60)
    return table


def test_no_test_photo_has_a_training_photo_within_the_gap():
    table = blocked_table()
    folds = make_folds(table[["photo_id", "class", "group_id"]], 5, 0.125, 2026)
    assert check_folds(folds, 5, 1.0)["no group in two roles within a round"]
    capture = table.set_index("photo_id").loc[folds["photo_id"]].reset_index()
    folds = folds.assign(camera=capture["camera"], seconds=ss.capture_seconds(capture["taken_at"]))
    for r in range(1, 6):
        assert (ss.nearest_training_gap(folds, r) > 60).all()
        assert (folds[round_column(r)] == "test").sum() > 0
    assert ss.exposure(folds, 5) == {5: 0.0, 10: 0.0, 60: 0.0}


def test_nearest_training_gap_known_answer():
    folds = pd.DataFrame({"photo_id": list("abcd"), "camera": ["x", "x", "x", "y"],
                          "seconds": [0.0, 7.0, 100.0, 8.0], "round_1": ["train", "test", "train", "test"]})
    gaps = ss.nearest_training_gap(folds, 1)
    assert gaps.tolist() == [7.0, np.inf]  # d is from a camera with no training photo
