"""Read settings files and turn their paths into real paths on this machine.

Every experiment settings file starts with ``extends: <file>`` (usually ``base.yaml``,
relative to the file's own folder) and only adds or overrides what it needs. Nested
sections are merged key by key; any other value (numbers, strings, lists) is replaced
as a whole.

Paths in settings files are relative to the repository root. On Kaggle the code
(``src/``, ``configs/``, ``splits/``) lives in the read-only ``ricetein-code`` dataset,
so outputs go under ``/kaggle/working/`` and datasets are found by name.

Print a merged settings file:  uv run python -m src.config configs/skeleton.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.kaggle_bootstrap import find_kaggle_input

REPO_ROOT = Path(__file__).resolve().parents[1]  # on Kaggle: the ricetein-code dataset folder
KAGGLE_WORKING = Path("/kaggle/working")
# Shipped inside ricetein-code (keep in step with scripts/kaggle_sync_code.sh); other `paths` are outputs.
READ_ONLY_PATHS = {"configs", "splits"}


def on_kaggle() -> bool:
    return KAGGLE_WORKING.exists()


def output_root() -> Path:
    """Where outputs go: ``/kaggle/working`` on Kaggle (the code there is read-only), else the repo."""
    return KAGGLE_WORKING if on_kaggle() else REPO_ROOT


def cache_folder(name: str, *, write: bool = False) -> Path:
    """Writable cache output, or a completed cache attached as ricetein-features on Kaggle."""
    folder = output_root() / "cache" / name
    if write or folder.exists() or not on_kaggle():
        return folder
    mounted = find_kaggle_input("ricetein-features")
    if mounted is not None:
        for candidate in (mounted / "cache" / name, mounted / name):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"cache {name}: build it first or attach ricetein-features")


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a settings file must be a mapping of names to values")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated by ``override``; nested dicts merge, everything else replaces.

    ``base`` itself is not modified (values are shared, not copied: settings are read-only).
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a settings file, following its ``extends`` chain.

    The result records which file it came from as ``config_file`` (repo-relative).
    """
    path = Path(path)
    path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    chain: list[Path] = []
    layers: list[dict[str, Any]] = []
    while True:
        if path in chain:
            raise ValueError(f"Circular 'extends' in {[str(p) for p in chain + [path]]}")
        chain.append(path)
        layers.append(read_yaml(path))
        parent = layers[-1].pop("extends", None)
        if parent is None:
            break
        path = (path.parent / parent).resolve()

    config: dict[str, Any] = {}
    for layer in reversed(layers):  # base first, the experiment file last
        config = deep_merge(config, layer)
    try:
        config["config_file"] = str(chain[0].relative_to(REPO_ROOT))
    except ValueError:
        config["config_file"] = str(chain[0])
    return config


def path_for(config: dict[str, Any], key: str) -> Path:
    """Real folder for ``config['paths'][key]`` on this machine (e.g. ``runs``, ``splits``)."""
    root = REPO_ROOT if key in READ_ONLY_PATHS else output_root()
    return root / config["paths"][key]


def dataset_dir(config: dict[str, Any], name: str) -> Path:
    """Folder that holds the class folders of dataset ``name`` (``primary`` or ``paddy``)."""
    spec = config["datasets"][name]
    if on_kaggle():
        root = find_kaggle_input(spec["kaggle_slug"])
        if root is None:
            raise FileNotFoundError(f"Kaggle dataset '{spec['kaggle_slug']}' is not attached")
    else:
        root = REPO_ROOT / spec["local_dir"]
    folder = root / spec["image_subdir"]
    missing = [f for f in spec["class_folders"].values() if not (folder / f).is_dir()]
    if missing:
        raise FileNotFoundError(f"{name} dataset: class folders {missing} not found in {folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a settings file with its 'extends' chain merged.")
    parser.add_argument("config", help="settings file, e.g. configs/skeleton.yaml")
    args = parser.parse_args()
    print(yaml.safe_dump(load_config(args.config), sort_keys=False), end="")


if __name__ == "__main__":
    main()
