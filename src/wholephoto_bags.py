"""Whole-photo descriptions re-filed as one-tile bags (declared 2026-11-14, handbook §24).

The Stage 7.1 probe is a linear model on the class token, while the tile models have a hidden layer and
choose their description by tuning, so "tiles vs whole photo" also compares two different heads. This
module copies one size of the whole-photo cache (Stage 6) into the tile-cache layout, one tile per photo,
so ``src.train_mil`` can train the *same* head with the *same* search on whole-photo descriptions. No
encoder is run and no number changes: each file is one row of its source file.

    uv run python -m src.wholephoto_bags --config configs/mil/whole224.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np
import torch

from src.config import cache_folder, load_config
from src.embed import completed_cache, save
from src.records import require_clean_code


def fingerprint(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build(config: dict[str, Any]) -> dict[str, Any]:
    """Write cache/<mil.cache> from one size of cache/<whole_bag.source>; returns its completion record."""
    require_clean_code()
    name, source, size = config["mil"]["cache"], config["whole_bag"]["source"], config["whole_bag"]["size"]
    completed_cache(config, source)  # the source must be complete and made with the current code
    origin, target = cache_folder(source), cache_folder(name, write=True)
    parent = json.loads((origin / "spec.json").read_text())
    row = parent["embed"]["whole_sizes"].index(size)
    spec = {**parent, "embed": {**parent["embed"], "name": name},
            "derived": {"from": source, "from_spec_sha256": fingerprint(parent), "whole_size": size,
                        "by": "src/wholephoto_bags.py"}}
    if (target / "complete.json").exists():
        if json.loads((target / "spec.json").read_text()) != spec:
            raise ValueError(f"{target}: built from a different source; archive it before rebuilding")
        return json.loads((target / "complete.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    (target / "spec.json").write_text(json.dumps(spec, indent=2))
    files = rows = size_bytes = 0
    versions = parent["embed"]["versions"]
    for version in range(versions):
        for path in sorted((origin / f"v{version}").glob("*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                cls, mean, boxes = (saved[key][row:row + 1] for key in ("cls", "mean", "boxes"))
            out = target / f"v{version}" / path.name
            save(out, torch.from_numpy(cls), torch.from_numpy(mean), boxes)
            files, rows, size_bytes = files + 1, rows + 1, size_bytes + out.stat().st_size
    complete = {"name": name, "photos": files // versions, "versions": versions, "files": files,
                "descriptor_rows": rows, "bytes": size_bytes, "spec_sha256": fingerprint(spec)}
    (target / "complete.json").write_text(json.dumps(complete, indent=2))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    print(build(load_config(parser.parse_args().config)))


if __name__ == "__main__":
    main()
