"""Checks for the small helpers in src/inventory_report.py (the full report runs on real data)."""

import pandas as pd

from src.inventory_report import duplicate_groups, md_table


def test_md_table_renders_header_separator_and_rows():
    text = md_table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |", "| 2 | y |"]


def test_duplicate_groups_counts_files_and_folders():
    frame = pd.DataFrame({"photo_id": ["A/1", "B/2", "A/3", "A/4"], "folder": ["A", "B", "A", "A"],
                          "sha256": ["s1", "s1", "s2", "s3"]})
    groups = duplicate_groups(frame, "sha256")
    assert groups.to_dict("index") == {"s1": {"files": 2, "folders": "A + B"}}
