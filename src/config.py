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
import copy
from pathlib import Path
from typing import Any

import yaml

from src.kaggle_bootstrap import find_kaggle_input

REPO_ROOT = Path(__file__).resolve().parents[1]  # on Kaggle: the ricetein-code dataset folder
KAGGLE_WORKING = Path("/kaggle/working")
READ_ONLY_PATHS = {"configs", "splits"}  # shipped with the code; everything else in `paths` is written to


def on_kaggle() -> bool:
    return KAGGLE_WORKING.exists()


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated by ``override``; nested dicts merge, everything else replaces."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a settings file, following its ``extends`` chain.

    The result records which file it came from as ``config_file`` (repo-relative).
    """
    path = _resolve(path).resolve()
    chain: list[Path] = []
    while True:
        if path in chain:
            raise ValueError(f"Circular 'extends' in {[str(p) for p in chain + [path]]}")
        chain.append(path)
        parent = read_yaml(path).get("extends")
        if parent is None:
            break
        path = (path.parent / parent).resolve()

    config: dict[str, Any] = {}
    for file in reversed(chain):  # base first, the experiment file last
        layer = read_yaml(file)
        layer.pop("extends", None)
        config = deep_merge(config, layer)
    try:
        config["config_file"] = str(chain[0].relative_to(REPO_ROOT))
    except ValueError:
        config["config_file"] = str(chain[0])
    return config


def path_for(config: dict[str, Any], key: str) -> Path:
    """Real folder for ``config['paths'][key]`` on this machine (e.g. ``runs``, ``splits``)."""
    relative = config["paths"][key]
    if on_kaggle() and key not in READ_ONLY_PATHS:
        return KAGGLE_WORKING / relative
    return REPO_ROOT / relative


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
    if not folder.is_dir():
        raise FileNotFoundError(f"{name} dataset not found at {folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a settings file with its 'extends' chain merged.")
    parser.add_argument("config", help="settings file, e.g. configs/skeleton.yaml")
    args = parser.parse_args()
    print(yaml.safe_dump(load_config(args.config), sort_keys=False), end="")


if __name__ == "__main__":
    main()
