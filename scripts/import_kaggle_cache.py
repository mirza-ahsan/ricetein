"""Bring one Stage 6 cache built on Kaggle into cache/, after checking that it is the cache we asked for.

    ./scripts/kaggle_pull.sh <kernel>
    uv run python scripts/import_kaggle_cache.py <kernel> <cache name>      e.g. 06b-caches-a e3_dinov2_vitb14

Checks, in this order (a failure stops before anything is moved):
1. the cache folder exists in kaggle/outputs/<kernel>/cache/ and has a complete.json;
2. its spec.json equals the spec computed here from configs/embed/<name>.yaml and the local source files, so
   encoder, tile settings, source code and inputs are the same as for the caches built on this machine;
3. cache/<name> does not exist yet (an existing cache is never overwritten).
After the move every photo-version file is validated again with the same function that ends a local build.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from src import embed
from src.config import REPO_ROOT, cache_folder, load_config


def check(source: Path, config: dict, target: Path) -> None:
    if not (source / "complete.json").exists():
        raise ValueError(f"{source}: no complete.json; the Kaggle run did not finish this cache")
    if json.loads((source / "spec.json").read_text()) != embed.cache_spec(config):
        raise ValueError(f"{source}: built with other settings, code or inputs than the ones checked out here")
    if target.exists():
        raise ValueError(f"{target} exists; archive it first, an existing cache is never overwritten")


def bring_in(kernel: str, name: str) -> dict:
    config = load_config(f"configs/embed/{name}.yaml")
    source = REPO_ROOT / "kaggle" / "outputs" / kernel / "cache" / name
    target = cache_folder(name, write=True)
    check(source, config, target)
    shutil.move(str(source), str(target))
    try:
        complete = embed.validate_cache(config)
    except Exception:
        target.rename(target.with_name(f"{name}_rejected"))
        raise
    if complete["spec_sha256"] != json.loads((target / "complete.json").read_text())["spec_sha256"]:
        target.rename(target.with_name(f"{name}_rejected"))
        raise ValueError(f"{name}: the manifest from Kaggle does not match the validated cache")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kernel")
    parser.add_argument("name")
    args = parser.parse_args()
    complete = bring_in(args.kernel, args.name)
    print(f"{args.name}: {complete['photos']} photos x {complete['versions']} versions, "
          f"{complete['descriptor_rows']} descriptor rows, {complete['bytes'] / 1e9:.1f} GB, validated")


if __name__ == "__main__":
    sys.exit(main())
