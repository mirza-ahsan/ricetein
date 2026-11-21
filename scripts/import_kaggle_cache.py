"""Bring one cache built on another machine into cache/, after checking that it is the cache we asked for.

    ./scripts/kaggle_pull.sh <kernel>                                            (Kaggle), or rsync into kaggle/outputs/pc/cache/ (the PC)
    uv run python scripts/import_kaggle_cache.py <source> <cache name>           e.g. 06b-caches-a e3_dinov2_vitb14, or pc paddy_e1_native

``<source>`` is the folder under kaggle/outputs/ that holds ``cache/<name>``. Works for the Stage 6 caches
(``configs/embed/<name>.yaml``) and the Stage 10 Paddy Doctor caches (``configs/paddy_zero_shot.yaml``).
The cache is refused, before anything moves, unless it is complete, has the spec computed here, and does not
replace an existing cache (same filesystem, so the move is one rename). After the move every photo file is
validated again; a cache that fails (or an interrupted validation) is renamed ``<name>_rejected`` (numbered if
that exists).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src import embed, paddy_embed
from src.config import REPO_ROOT, cache_folder, load_config

PADDY = "configs/paddy_zero_shot.yaml"


def plan(name: str) -> tuple[dict[str, Any], Callable[[], dict[str, Any]]]:
    """The spec this cache must have, and the function that validates it once it is in cache/."""
    paddy = load_config(PADDY)
    caches = paddy_embed.description_caches(paddy)
    if name in caches:
        return paddy_embed.cache_spec(paddy, caches[name]), lambda: paddy_embed.validate(paddy, name)
    config = load_config(f"configs/embed/{name}.yaml")
    if config["embed"]["name"] != name:
        raise ValueError(f"configs/embed/{name}.yaml describes the cache {config['embed']['name']}, not {name}")
    return embed.cache_spec(config), lambda: embed.validate_cache(config)


def check(source: Path, expected_spec: dict[str, Any], target: Path) -> None:
    if not (source / "complete.json").exists():
        raise ValueError(f"{source}: no complete.json; the build did not finish this cache")
    if list(source.glob("v*/*.tmp.npz")):
        raise ValueError(f"{source}: unfinished files (*.tmp.npz)")
    if json.loads((source / "spec.json").read_text()) != expected_spec:
        raise ValueError(f"{source}: built with other settings, code or inputs than the ones checked out here")
    if target.exists():
        raise ValueError(f"{target} exists; archive it first, an existing cache is never overwritten")
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError(f"{source} and {target.parent} are on different filesystems; the move must be one rename")


def reject(target: Path) -> None:
    n, rejected = 0, target.with_name(f"{target.name}_rejected")
    while rejected.exists():
        n += 1
        rejected = target.with_name(f"{target.name}_rejected{n}")
    target.rename(rejected)


def bring_in(source_name: str, name: str) -> dict[str, Any]:
    expected_spec, validate = plan(name)
    source = REPO_ROOT / "kaggle" / "outputs" / source_name / "cache" / name
    target = cache_folder(name, write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    check(source, expected_spec, target)
    source.rename(target)
    try:
        complete = validate()
        if complete["spec_sha256"] != json.loads((target / "complete.json").read_text())["spec_sha256"]:
            raise ValueError(f"{name}: the manifest from Kaggle does not match the validated cache")
    except BaseException:                  # an interrupted validation must not leave an unchecked cache in cache/
        reject(target)
        raise
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("name")
    args = parser.parse_args()
    complete = bring_in(args.source, args.name)
    print(f"{args.name}: {complete['photos']} photos x {complete['versions']} versions, "
          f"{complete['descriptor_rows']} descriptor rows, {complete['bytes'] / 1e9:.1f} GB, validated")


if __name__ == "__main__":
    main()
