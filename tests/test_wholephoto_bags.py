"""Checks for src/wholephoto_bags.py: whole-photo rows re-filed as one-tile bags (no real cache is touched)."""

import json

import numpy as np
import pytest

from src import wholephoto_bags

SOURCE_SPEC = {"format": 2, "embed": {"name": "src_cache", "whole_sizes": [224, 512], "versions": 2}}
CONFIG = {"mil": {"cache": "bag512"}, "whole_bag": {"source": "src_cache", "size": 512}}
FILES = ("A__x.npz", "B__y.npz")


def make_source(root):
    source = root / "src_cache"
    source.mkdir()
    (source / "spec.json").write_text(json.dumps(SOURCE_SPEC))
    for version in range(2):
        for index, name in enumerate(FILES):
            base = 100 * version + 10 * index
            cls = (base + np.arange(8).reshape(2, 4)).astype(np.float16)
            (source / f"v{version}").mkdir(exist_ok=True)
            np.savez(source / f"v{version}" / name, cls=cls, mean=cls + 0.5, boxes=np.zeros((2, 4), np.int32))
    return source


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(wholephoto_bags, "cache_folder", lambda name, write=False: tmp_path / name)
    monkeypatch.setattr(wholephoto_bags, "completed_cache", lambda config, name: {})
    monkeypatch.setattr(wholephoto_bags, "require_clean_code", lambda: "abc1234")
    return make_source(tmp_path)


def test_output_files_are_row_one_of_their_source(source, tmp_path):
    wholephoto_bags.build(CONFIG)
    for version in range(2):
        for name in FILES:
            with np.load(source / f"v{version}" / name) as old, np.load(tmp_path / "bag512" / f"v{version}" / name) as new:
                assert new["cls"].shape == (1, 4) and new["cls"].dtype == np.float16
                assert np.array_equal(new["cls"], old["cls"][1:2])
                assert np.array_equal(new["mean"], old["mean"][1:2])
                assert new["boxes"].shape == (1, 4)


def test_completion_record_and_spec(source, tmp_path):
    record = wholephoto_bags.build(CONFIG)
    spec_text = (tmp_path / "bag512" / "spec.json").read_text()
    spec = json.loads(spec_text)
    assert record["name"] == "bag512" and record["photos"] == 2 and record["versions"] == 2
    assert record["files"] == 4 and record["descriptor_rows"] == 4
    assert record["spec_sha256"] == wholephoto_bags.fingerprint(spec)
    assert spec["embed"]["name"] == "bag512" and spec["derived"]["whole_size"] == 512
    assert spec["derived"]["from_spec_sha256"] == wholephoto_bags.fingerprint(SOURCE_SPEC)


def test_second_build_returns_the_same_record_and_rewrites_nothing(source, tmp_path):
    first = wholephoto_bags.build(CONFIG)
    files = sorted((tmp_path / "bag512").rglob("*"))
    stamps = [path.stat().st_mtime_ns for path in files]
    assert wholephoto_bags.build(CONFIG) == first
    assert [path.stat().st_mtime_ns for path in files] == stamps


def test_changed_source_spec_is_refused(source, tmp_path):
    wholephoto_bags.build(CONFIG)
    changed = {**SOURCE_SPEC, "embed": {**SOURCE_SPEC["embed"], "versions": 3}}
    (source / "spec.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different source"):
        wholephoto_bags.build(CONFIG)
