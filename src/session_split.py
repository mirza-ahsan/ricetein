"""The session-blocked sensitivity split (declared 2026-11-14; handbook §24).

Why: the primary photos were shot in long sessions, a few seconds apart, and the same leaf was often
photographed several times. The Stage 2 audit groups pixel-level copies only, so in the locked split most
test photos have a training photo taken seconds earlier or later. This module builds a second, clearly
separate split that keeps whole capture blocks together, to measure how much that matters.

A capture block is a run of consecutive photos from one camera with no pause longer than
``session.max_gap_seconds``; blocks that share a near-duplicate group are merged. The same photos, classes,
fold rule and seed as the locked split are used (``src.split.make_folds``); only the grouping changes. The
file has the columns of ``splits/folds.csv`` (``group_id`` holds the block), so every loader reads it
unchanged. ``splits/folds.csv`` itself is never touched.

    uv run python -m src.session_split        # writes splits/session_blocked/folds.csv and its report
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config, path_for
from src.inventory import read_inventory
from src.inventory_report import md_table
from src.split import ROLES, check_folds, make_folds, round_column, sha256

TIME_FORMAT = "%Y:%m:%d %H:%M:%S"  # EXIF capture time, as stored in the inventory


def capture_seconds(taken_at: pd.Series) -> pd.Series:
    """Capture time in seconds since 2019-01-01 (NaN where the photo has no capture time)."""
    times = pd.to_datetime(taken_at, format=TIME_FORMAT, errors="coerce")
    return (times - pd.Timestamp("2019-01-01")).dt.total_seconds()


def capture_blocks(photos: pd.DataFrame, max_gap: float) -> pd.Series:
    """Block id per photo (indexed like ``photos``: photo_id, camera, taken_at, group_id).

    A new block starts when the camera changes, the pause exceeds ``max_gap`` seconds, or a capture time
    is missing. Blocks linked by a shared near-duplicate group become one block.
    """
    ordered = photos.assign(seconds=capture_seconds(photos["taken_at"]),
                            camera=photos["camera"].fillna("unknown"))
    ordered = ordered.sort_values(["camera", "seconds", "photo_id"])
    seconds, camera = ordered["seconds"].to_numpy(), ordered["camera"].to_numpy()
    pause = np.diff(seconds)
    starts = np.r_[True, (camera[1:] != camera[:-1]) | ~(pause <= max_gap)]  # a NaN pause starts a block
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    runs = [f"run{n}" for n in np.cumsum(starts)]
    for run, group in zip(runs, ordered["group_id"]):
        parent[find(run)] = find(f"group{group}")
    roots = pd.Series([find(run) for run in runs], index=ordered.index)
    first_photo = ordered["photo_id"].groupby(roots).min().sort_values()
    names = {root: f"b{n:03d}" for n, root in enumerate(first_photo.index, start=1)}
    return roots.map(names).reindex(photos.index)


def nearest_in_sorted(times: np.ndarray, query: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """For each query time: the index of the nearest of the sorted ``times`` (the earlier one on a tie) and
    its distance (NaN when the query time is missing)."""
    at = np.searchsorted(times, query)
    before, after = np.clip(at - 1, 0, len(times) - 1), np.clip(at, 0, len(times) - 1)
    d_before, d_after = np.abs(query.to_numpy() - times[before]), np.abs(query.to_numpy() - times[after])
    return np.where(d_before <= d_after, before, after), np.fmin(d_before, d_after)


def nearest_training_gap(table: pd.DataFrame, r: int) -> pd.Series:
    """Per test photo of round ``r``: seconds to the nearest training photo from the same camera
    (inf when there is none or a time is missing). ``table`` has camera and seconds columns."""
    role = table[round_column(r)]
    gaps = pd.Series(np.inf, index=table.index[role == "test"])
    for camera, test in table[role == "test"].groupby("camera"):
        train = np.sort(table.loc[(role == "train") & (table["camera"] == camera), "seconds"].dropna())
        if not len(train):
            continue
        gaps[test.index] = np.nan_to_num(nearest_in_sorted(train, test["seconds"])[1], nan=np.inf)
    return gaps


def with_capture(folds: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """A fold table plus each photo's camera and capture time in seconds."""
    inventory = read_inventory(config["inventory"]["files"]["primary"]).set_index("photo_id")
    capture = inventory.loc[folds["photo_id"], ["camera", "taken_at"]].reset_index(drop=True)
    return folds.reset_index(drop=True).assign(camera=capture["camera"].fillna("unknown"),
                                               taken_at=capture["taken_at"],
                                               seconds=capture_seconds(capture["taken_at"]))


def exposure(table: pd.DataFrame, n_folds: int, windows: tuple[int, ...] = (5, 10, 60)) -> dict[int, float]:
    """Share of test photos (all rounds) with a same-camera training photo within each window of seconds."""
    gaps = pd.concat([nearest_training_gap(table, r) for r in range(1, n_folds + 1)])
    return {w: float((gaps <= w).mean()) for w in windows}


def build(config: dict[str, Any]) -> dict[str, Any]:
    """Write the session-blocked fold file (refusing to replace it) and return what the report needs."""
    settings, session = config["split"], config["session"]
    target = REPO_ROOT / config["split_files"]["folds"]
    if target.exists():
        raise SystemExit(f"{target} exists and is never regenerated")
    locked = with_capture(pd.read_csv(REPO_ROOT / session["source_folds"], dtype=str), config)
    blocks = capture_blocks(locked[["photo_id", "camera", "taken_at", "group_id"]], session["max_gap_seconds"])
    photos = locked[["photo_id", "class"]].assign(group_id=blocks)
    table = make_folds(photos, settings["n_folds"], settings["val_fraction"], config["split_seed"])
    checks = check_folds(table, settings["n_folds"], settings["max_class_share_deviation"])
    if not (checks["every photo tested exactly once"] and checks["no group in two roles within a round"]):
        raise SystemExit(f"split checks failed: {checks}")
    target.parent.mkdir(parents=True, exist_ok=True)
    table.drop(columns="fold").to_csv(target, index=False)
    return {"table": table, "locked": locked, "checks": checks, "sha256": sha256(target)}


def report(config: dict[str, Any], result: dict[str, Any]) -> str:
    table, settings, session = result["table"], config["split"], config["session"]
    n = settings["n_folds"]
    sizes = table["group_id"].value_counts()
    counts = [{"round": r, "role": role, "photos": int((table[round_column(r)] == role).sum()),
               **table.loc[table[round_column(r)] == role, "class"].value_counts()
               .reindex(config["classes"], fill_value=0).to_dict()}
              for r in range(1, n + 1) for role in ROLES]
    blocked = with_capture(table, config)
    shares = pd.DataFrame([{"split": name, **{f"within {w} s": f"{100 * share:.1f}%"
                                                for w, share in exposure(frame, n).items()}}
                           for name, frame in (("locked (`splits/folds.csv`)", result["locked"]),
                                               ("session-blocked", blocked))])
    checks = "\n".join(f"- {name}: **{'yes' if ok is True else ('no' if ok is False else f'{ok:.2%}')}**"
                       for name, ok in result["checks"].items())
    text = (
        "# Session-blocked split report\n\nBuilt by `src/session_split.py`; do not edit by hand. "
        "A sensitivity split declared on 2026-11-14 (handbook §24). The locked split is unchanged.\n\n"
        f"The same {len(table)} photos as the locked split, grouped into {len(sizes)} capture blocks: runs "
        f"of consecutive photos from one camera with no pause longer than {session['max_gap_seconds']} s, "
        "merged where they share a near-duplicate group. Block sizes: largest "
        f"{int(sizes.iloc[0])} photos ({100 * sizes.iloc[0] / len(table):.1f}% of all), median "
        f"{int(sizes.median())}. StratifiedGroupKFold, {n} folds, seed {config['split_seed']}; validation = "
        f"a stratified, grouped 1/{round(1 / settings['val_fraction'])} of the other four folds.\n\n"
        f"**`{config['split_files']['folds']}` SHA-256:** `{result['sha256']}`\n\n"
        f"Checks (class shares are reported, not enforced: whole blocks limit how even they can be):\n\n"
        f"{checks}\n\n" + md_table(pd.DataFrame(counts)) + "\n\n"
        "Share of test photos with a training photo from the same camera taken close in time "
        "(all five rounds):\n\n" + md_table(shares) + "\n")
    (path_for(config, "reports") / "session_split_report.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/session/split.yaml")
    config = load_config(parser.parse_args().config)
    result = build(config)
    report(config, result)
    print(f"session-blocked folds: {len(result['table'])} photos, sha256 {result['sha256']}; "
          f"checks {result['checks']}")


if __name__ == "__main__":
    main()
