"""Checks for scripts/import_kaggle_cache.py: a cache from Kaggle is brought in only when it is complete, was
built with the settings and code checked out here, and does not replace an existing cache (no GPU, no data)."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("import_kaggle_cache", Path(__file__).parents[1] / "scripts/import_kaggle_cache.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)

EXPECTED = {"expected": True}


@pytest.fixture
def folders(tmp_path):
    source, target = tmp_path / "outputs" / "x", tmp_path / "cache" / "x"
    source.mkdir(parents=True)
    target.parent.mkdir()
    (source / "spec.json").write_text(json.dumps(EXPECTED))
    (source / "complete.json").write_text("{}")
    return source, target


def test_a_complete_cache_with_the_expected_settings_is_accepted(folders):
    importer.check(folders[0], EXPECTED, folders[1])


def test_an_unfinished_cache_is_refused(folders):
    (folders[0] / "complete.json").unlink()
    with pytest.raises(ValueError, match="did not finish"):
        importer.check(folders[0], EXPECTED, folders[1])


def test_leftover_unfinished_files_are_refused(folders):
    (folders[0] / "v0").mkdir()
    (folders[0] / "v0" / "p.tmp.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="unfinished files"):
        importer.check(folders[0], EXPECTED, folders[1])


def test_a_cache_built_with_other_settings_or_code_is_refused(folders):
    with pytest.raises(ValueError, match="other settings, code or inputs"):
        importer.check(folders[0], {"expected": False}, folders[1])


def test_an_existing_cache_is_never_overwritten(folders):
    folders[1].mkdir()
    with pytest.raises(ValueError, match="never overwritten"):
        importer.check(folders[0], EXPECTED, folders[1])


def test_a_paddy_cache_is_checked_against_the_paddy_spec_and_a_stage_6_cache_against_its_own(monkeypatch):
    seen = []
    monkeypatch.setattr(importer.paddy_embed, "description_caches", lambda c: {"paddy_e1_native": {"kind": "tiles"}})
    monkeypatch.setattr(importer.paddy_embed, "cache_spec", lambda c, s: seen.append("paddy") or {"p": 1})
    monkeypatch.setattr(importer.embed, "cache_spec", lambda c: seen.append("stage6") or {"s": 1})
    assert importer.plan("paddy_e1_native")[0] == {"p": 1}
    assert importer.plan("e2_convnext_tiny")[0] == {"s": 1}
    assert seen == ["paddy", "stage6"]


def bring(monkeypatch, tmp_path, validate):
    monkeypatch.setattr(importer, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(importer, "cache_folder", lambda name, write=False: tmp_path / "cache" / name)
    monkeypatch.setattr(importer, "plan", lambda name: (EXPECTED, validate))
    source = tmp_path / "kaggle" / "outputs" / "k" / "cache" / "x"
    source.mkdir(parents=True)
    (source / "spec.json").write_text(json.dumps(EXPECTED))
    (source / "complete.json").write_text(json.dumps({"spec_sha256": "abc"}))
    return importer.bring_in("k", "x")


def test_a_cache_that_validates_is_moved_in_one_rename(monkeypatch, tmp_path):
    manifest = {"spec_sha256": "abc", "photos": 1, "versions": 1, "descriptor_rows": 1, "bytes": 1}
    assert bring(monkeypatch, tmp_path, lambda: manifest) == manifest
    assert (tmp_path / "cache" / "x" / "complete.json").exists()
    assert not (tmp_path / "kaggle" / "outputs" / "k" / "cache" / "x").exists()


def test_a_cache_that_fails_validation_is_renamed_not_left_in_place(monkeypatch, tmp_path):
    (tmp_path / "cache" / "x_rejected").mkdir(parents=True)          # an earlier rejection must not block this one

    def broken():
        raise ValueError("invalid descriptor arrays")

    with pytest.raises(ValueError, match="invalid descriptor arrays"):
        bring(monkeypatch, tmp_path, broken)
    assert not (tmp_path / "cache" / "x").exists() and (tmp_path / "cache" / "x_rejected1").exists()
