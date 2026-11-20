"""Checks for scripts/import_kaggle_cache.py: a cache from Kaggle is brought in only when it is complete, was
built with the settings and code checked out here, and does not replace an existing cache (no GPU, no data)."""

import importlib.util
import json
from pathlib import Path

import pytest

from src import embed

spec = importlib.util.spec_from_file_location("import_kaggle_cache", Path(__file__).parents[1] / "scripts/import_kaggle_cache.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)

CONFIG = {"embed": {"name": "x"}}


@pytest.fixture
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "cache_spec", lambda config: {"expected": True})
    source, target = tmp_path / "from_kaggle", tmp_path / "cache" / "x"
    source.mkdir()
    (source / "spec.json").write_text(json.dumps({"expected": True}))
    (source / "complete.json").write_text("{}")
    return source, target


def test_a_complete_cache_with_the_expected_settings_is_accepted(folders):
    importer.check(folders[0], CONFIG, folders[1])


def test_an_unfinished_cache_is_refused(folders):
    (folders[0] / "complete.json").unlink()
    with pytest.raises(ValueError, match="did not finish"):
        importer.check(folders[0], CONFIG, folders[1])


def test_a_cache_built_with_other_settings_or_code_is_refused(folders):
    (folders[0] / "spec.json").write_text(json.dumps({"expected": False}))
    with pytest.raises(ValueError, match="other settings, code or inputs"):
        importer.check(folders[0], CONFIG, folders[1])


def test_an_existing_cache_is_never_overwritten(folders):
    folders[1].mkdir(parents=True)
    with pytest.raises(ValueError, match="never overwritten"):
        importer.check(folders[0], CONFIG, folders[1])
