"""A completed run is reusable while the code it executes is unchanged (handbook §24, 2026-11-14)."""

import json

import pytest

from src import records


def test_source_files_follow_imports_inside_functions_too():
    files = records.source_files("src.train_wholeimage")
    assert {"src/train_wholeimage.py", "src/photo_dataset.py", "src/augment.py", "src/otsu.py",
            "src/metrics.py", "src/split.py", "src/seeding.py", "src/config.py",
            "src/kaggle_bootstrap.py"} <= set(files)  # kaggle_bootstrap is imported inside build_model
    assert "src/evaluate.py" not in files and "src/train_mil.py" not in files
    assert "src/train_wholeimage.py" in records.source_files("src.train_mil")


def test_code_paths_by_run_kind():
    paths = records.code_paths("stage7_tune_attention")
    assert "src/train_mil.py" in paths and "src/mil_models.py" in paths
    assert "src/records.py" not in paths and "configs" not in paths
    assert {"pyproject.toml", "uv.lock"} <= set(paths)
    assert "src/linear_probe.py" in records.code_paths("stage7a_probe_e1_224_l2100_r1")
    assert "src/linear_probe.py" not in records.code_paths("stage7_attention_r1")
    assert records.code_paths("stage99_anything") == records.CODE_PATHS  # unknown kinds stay strict


@pytest.fixture
def old_run(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(records, "get_code_version", lambda: "new0000")
    folder = tmp_path / "stage5_swin_tiny_224_lr3e-05_r1"
    folder.mkdir()
    (folder / "record.json").write_text(json.dumps(
        {"status": "completed", "code_version": "old0000", "split_checksum": "sha", "config": {"a": 1}}))
    (folder / "metrics.json").write_text("{}")
    (folder / "predictions.csv").write_text("x\n")
    return folder.name


def test_reuse_asks_git_only_about_the_files_the_run_depends_on(old_run, monkeypatch):
    asked = []
    monkeypatch.setattr(records, "_git", lambda *args: asked.append(args) or "")
    assert records.reusable_run(old_run, {"a": 1}, "sha")
    command = asked[0]
    assert command[:4] == ("diff", "old0000", "new0000", "--")
    assert "src/train_wholeimage.py" in command and "src" not in command and "configs" not in command


def test_reuse_refuses_when_that_code_changed(old_run, monkeypatch):
    monkeypatch.setattr(records, "_git", lambda *args: "diff --git a/src/augment.py b/src/augment.py")
    with pytest.raises(ValueError, match="source code changed"):
        records.reusable_run(old_run, {"a": 1}, "sha")
