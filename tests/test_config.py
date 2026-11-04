"""Known-answer checks for src/config.py: the extends chain, merging and paths per machine."""

from pathlib import Path

import pytest

from src import config as cfgmod
from src.config import REPO_ROOT, dataset_dir, deep_merge, load_config, path_for


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def base_config():
    return load_config("configs/base.yaml")


@pytest.fixture
def kaggle(monkeypatch, tmp_path):
    """Pretend to be on Kaggle, with outputs under tmp_path/working."""
    monkeypatch.setattr(cfgmod, "on_kaggle", lambda: True)
    monkeypatch.setattr(cfgmod, "KAGGLE_WORKING", tmp_path / "working")
    return tmp_path


def test_deep_merge_merges_dicts_and_replaces_everything_else():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "items": [1, 2]}
    merged = deep_merge(base, {"nested": {"y": 3, "z": 4}, "items": [9]})
    assert merged == {"a": 1, "nested": {"x": 1, "y": 3, "z": 4}, "items": [9]}
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}, "items": [1, 2]}  # input untouched


def test_extends_chain_applies_base_first(tmp_path):
    write(tmp_path / "base.yaml", "seed: 0\nlr: 0.1\nnested: {a: 1, b: 2}\n")
    write(tmp_path / "mid.yaml", "extends: base.yaml\nlr: 0.01\nnested: {b: 3}\n")
    child = write(tmp_path / "sub" / "child.yaml", "extends: ../mid.yaml\nseed: 7\n")
    config = load_config(child)
    assert config["seed"] == 7 and config["lr"] == 0.01
    assert config["nested"] == {"a": 1, "b": 3}
    assert "extends" not in config
    assert config["config_file"] == str(child.resolve())


def test_circular_extends_is_rejected(tmp_path):
    write(tmp_path / "a.yaml", "extends: b.yaml\n")
    write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_settings_file_must_be_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        load_config(write(tmp_path / "list.yaml", "- 1\n- 2\n"))


def test_base_config_loads_relative_to_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the working directory must not matter
    config = load_config("configs/base.yaml")
    assert config["classes"] == ["healthy", "leaf_blast", "brown_spot", "hispa"]
    assert config["config_file"] == "configs/base.yaml"


def test_paths_on_local_machine(monkeypatch, base_config):
    monkeypatch.setattr(cfgmod, "on_kaggle", lambda: False)
    assert path_for(base_config, "runs") == REPO_ROOT / "runs"
    assert path_for(base_config, "splits") == REPO_ROOT / "splits"


def test_paths_on_kaggle_write_to_working_and_read_code_from_dataset(kaggle, base_config):
    assert path_for(base_config, "runs") == kaggle / "working" / "runs"
    assert path_for(base_config, "cache") == kaggle / "working" / "cache"
    assert path_for(base_config, "splits") == REPO_ROOT / "splits"  # shipped inside ricetein-code


def make_primary(root: Path) -> Path:
    for folder in ("Healthy", "LeafBlast", "BrownSpot", "Hispa"):
        (root / "LabelledRice" / "Labelled" / folder).mkdir(parents=True)
    return root


def test_dataset_dir_on_kaggle_finds_dataset_by_name(monkeypatch, kaggle, base_config):
    mounted = make_primary(kaggle / "input" / "datasets" / "minhhuy2810" / "rice-diseases-image-dataset")
    slug = "rice-diseases-image-dataset"
    monkeypatch.setattr(cfgmod, "find_kaggle_input", lambda name: mounted if name == slug else None)
    assert dataset_dir(base_config, "primary") == mounted / "LabelledRice" / "Labelled"
    with pytest.raises(FileNotFoundError, match="not attached"):
        dataset_dir(base_config, "paddy")


def test_dataset_dir_rejects_folder_without_the_class_folders(monkeypatch, kaggle, base_config):
    mounted = kaggle / "input" / "paddy-doctor-full"
    for folder in ("normal", "blast", "brown_spot", "hispa"):  # nested one level too deep
        (mounted / "paddy_doctor" / folder).mkdir(parents=True)
    monkeypatch.setattr(cfgmod, "find_kaggle_input", lambda name: mounted)
    with pytest.raises(FileNotFoundError, match="class folders"):
        dataset_dir(base_config, "paddy")


def test_dataset_dir_missing_locally_raises(monkeypatch, base_config):
    monkeypatch.setattr(cfgmod, "on_kaggle", lambda: False)
    base_config["datasets"]["primary"]["local_dir"] = "data/raw/does_not_exist"
    with pytest.raises(FileNotFoundError):
        dataset_dir(base_config, "primary")
