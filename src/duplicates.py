"""Stage 2b/2c: near-duplicates within the primary dataset, and overlap with Paddy Doctor.

1. ``flag``: a pair is flagged by either pre-declared rule (pre-declared methodology, "Data audits"):
   64-bit perceptual hashes differ in at most ``max_hash_bits`` bits, comparing each photo with all
   8 rotations/flips of the other; or the whole-photo E1 class tokens have cosine > ``min_cosine``.
   Flagged pairs get an automatic check: SIFT keypoints matched with Lowe's ratio test, then a RANSAC
   homography (also against the mirrored photo); a pair passes with at least ``min_inliers`` inliers.
   The review queue holds every passed pair plus a random sample of rejected ones (to measure what the
   check misses), shuffled and blind: reviewers see two photos, not classes or check results.
   Deviation from "review every flagged pair": handbook §24, 2026-11-07.
2. ``review --reviewer <name>``: two people each judge every queued pair alone (S same leaf / D different).
3. ``settle``: both look together at the pairs they disagreed on.
4. ``combine``: confirmed primary pairs form groups (connected components; singletons included); groups
   mixing classes are label conflicts (excluded from all experiments); confirmed cross-dataset pairs are
   the overlap list.

    uv run python -m src.duplicates flag                       # caches, flags, check, review queue
    uv run python -m src.duplicates review --reviewer <name>   # each reviewer, alone
    uv run python -m src.duplicates settle                     # both reviewers together
    uv run python -m src.duplicates combine                    # pairs, groups, conflicts, overlap
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import cohen_kappa_score

from src import review_tool
from src.config import REPO_ROOT, dataset_dir, load_config
from src.inventory import load_for_hash, perceptual_hash, read_inventory

DATASETS = ("primary", "paddy")
SETTLED = "settled"
REVIEW_COLUMNS = ["pair_id", "reviewer", "verdict", "reviewed_at"]
INSTRUCTIONS = ("Same leaf (S): a copy, crop, resize, rotation or re-shot of one leaf.   "
                "Different leaf (D).   Esc = stop")
T = Image.Transpose
D4 = [(), (T.ROTATE_90,), (T.ROTATE_180,), (T.ROTATE_270,),  # 4 rotations, each with and without a mirror
      (T.FLIP_LEFT_RIGHT,), (T.FLIP_LEFT_RIGHT, T.ROTATE_90), (T.FLIP_LEFT_RIGHT, T.ROTATE_180),
      (T.FLIP_LEFT_RIGHT, T.ROTATE_270)]


def photos(config: dict[str, Any], name: str) -> pd.DataFrame:
    """Every photo of a dataset that opened: photo_id, class (Paddy Doctor's untested classes: their
    folder name), path and the Stage 1 hash, sorted by photo_id."""
    inventory = read_inventory(config["inventory"]["files"][name])
    inventory = inventory[inventory["opened_ok"]].sort_values("photo_id", ignore_index=True)
    root = dataset_dir(config, name)
    return pd.DataFrame({"photo_id": inventory["photo_id"],
                         "class": inventory["class"].fillna(inventory["folder"]),
                         "path": [str(root / p) for p in inventory["photo_id"]], "phash": inventory["phash"]})


def path_lookup(lists: dict[str, pd.DataFrame]) -> dict[tuple[str, str], str]:
    return {(name, pid): path for name, frame in lists.items()
            for pid, path in zip(frame["photo_id"], frame["path"])}


def cached(target: Path, lists: dict[str, pd.DataFrame],
           build: Callable[[str, pd.DataFrame], np.ndarray]) -> dict[str, np.ndarray]:
    """{dataset: array} from a git-ignored cache, rebuilt when the photo lists no longer match it."""
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if all(saved[f"{n}_ids"].tolist() == frame["photo_id"].tolist() for n, frame in lists.items()):
                return {n: saved[n] for n in lists}
    arrays = {name: build(name, frame) for name, frame in lists.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays, **{f"{n}_ids": f["photo_id"].to_numpy(str) for n, f in lists.items()})
    return arrays


# ---------------------------------------------------------------- hashes and descriptors
def orientation_hashes(path: str, hash_size: int, draft_size: int) -> list[int]:
    """Perceptual hash of each of the 8 orientations, decoded exactly as in Stage 1."""
    image = load_for_hash(path, draft_size)
    hashes = []
    for steps in D4:
        variant = image
        for step in steps:
            variant = variant.transpose(step)
        hashes.append(int(perceptual_hash(variant, hash_size), 16))
    return hashes


def hash_cache(config: dict[str, Any], lists: dict[str, pd.DataFrame],
               workers: int | None) -> dict[str, np.ndarray]:
    """{dataset: (N, 8) uint64}, the 8 orientation hashes of every photo (about a minute to build)."""
    work = partial(orientation_hashes, **config["hashing"])

    def build(name: str, frame: pd.DataFrame) -> np.ndarray:
        with Pool(workers) as pool:
            hashes = np.array(pool.map(work, frame["path"], chunksize=32), dtype=np.uint64)
        stage1 = np.array([int(h, 16) for h in frame["phash"]], dtype=np.uint64)
        if not np.array_equal(hashes[:, 0], stage1):  # a different decode moves hashes by up to 4 bits
            raise RuntimeError(f"{name}: upright hashes differ from the Stage 1 inventory")
        return hashes

    return cached(REPO_ROOT / config["duplicates"]["hash_cache"], lists, build)


class WholePhotos:
    """Torch dataset: each photo decoded at reduced scale and resized to a ``size`` x ``size`` uint8 array."""

    def __init__(self, paths: list[str], size: int) -> None:
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        with Image.open(self.paths[i]) as image:
            image.draft("RGB", (self.size, self.size))
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(image.resize((self.size, self.size), Image.Resampling.BICUBIC))


def descriptor_cache(config: dict[str, Any], lists: dict[str, pd.DataFrame],
                     workers: int | None) -> dict[str, np.ndarray]:
    """{dataset: (N, dim) float32}, the unit-length whole-photo E1 class token (built on the GPU)."""
    import torch
    from torch.utils.data import DataLoader

    from src.encoders import describe, load_encoder
    from src.seeding import seed_everything

    settings, code = config["duplicates"], config["main_encoder"]
    state: dict[str, Any] = {}

    def build(name: str, frame: pd.DataFrame) -> np.ndarray:
        if not state:
            seed_everything(config["seed"], config["determinism"])
            state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state["model"] = load_encoder(config, code, state["device"])
        loader = DataLoader(WholePhotos(frame["path"].tolist(), settings["whole_size"]),
                            batch_size=settings["batch_size"], num_workers=workers or os.cpu_count() or 1)
        tokens = [describe(state["model"], batch, config["encoders"][code], config, state["device"])[0]
                  for batch in loader]
        vectors = torch.cat(tokens).numpy().astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return cached(REPO_ROOT / settings["descriptor_cache"], lists, build)


# ---------------------------------------------------------------- flagging
def hash_bits(upright: np.ndarray, variants: np.ndarray) -> np.ndarray:
    """(n, m) uint8: fewest differing bits between each upright hash and any orientation of each photo."""
    out = np.empty((len(upright), len(variants)), dtype=np.uint8)
    for start in range(0, len(upright), 64):
        chunk = upright[start:start + 64, None, None] ^ variants[None]
        out[start:start + 64] = np.bitwise_count(chunk).min(axis=2)
    return out


def flag_pairs(lists: dict[str, pd.DataFrame], hashes: dict[str, np.ndarray], vectors: dict[str, np.ndarray],
               max_bits: int, min_cosine: float) -> pd.DataFrame:
    """Every pair flagged by either rule: within the primary dataset, and primary x Paddy Doctor."""
    p, q = hashes["primary"], hashes["paddy"]
    within = hash_bits(p[:, 0], p)
    comparisons = {  # name: (hash bits, cosine, dataset of photo b)
        "primary": (np.minimum(within, within.T), vectors["primary"] @ vectors["primary"].T, "primary"),
        "cross": (np.minimum(hash_bits(p[:, 0], q), hash_bits(q[:, 0], p).T),
                  vectors["primary"] @ vectors["paddy"].T, "paddy"),
    }
    first = lists["primary"]
    frames = []
    for comparison, (bits, cosine, other) in comparisons.items():
        flagged = (bits <= max_bits) | (cosine > min_cosine)
        if comparison == "primary":
            flagged = np.triu(flagged, k=1)  # each unordered pair once, never a photo with itself
        a, b = np.nonzero(flagged)
        frames.append(pd.DataFrame({
            "comparison": comparison, "dataset_a": "primary", "photo_a": first["photo_id"].to_numpy()[a],
            "class_a": first["class"].to_numpy()[a], "dataset_b": other,
            "photo_b": lists[other]["photo_id"].to_numpy()[b], "class_b": lists[other]["class"].to_numpy()[b],
            "hash_bits": bits[a, b].astype(int), "cosine": cosine[a, b].round(4),
            "by_hash": bits[a, b] <= max_bits, "by_cosine": cosine[a, b] > min_cosine}))
    pairs = pd.concat(frames, ignore_index=True).sort_values(["comparison", "photo_a", "photo_b"],
                                                             ignore_index=True)
    prefix = pairs["comparison"].map({"primary": "p", "cross": "x"})
    return pairs.assign(pair_id=prefix + pairs.groupby("comparison").cumcount().map("{:06d}".format))


# ---------------------------------------------------------------- automatic check
_CHECK: dict[str, Any] = {}


def _init_check(check: dict[str, Any]) -> None:
    import cv2

    cv2.setNumThreads(1)
    _CHECK.update(check)


@lru_cache(maxsize=256)
def keypoints(path: str, mirror: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """SIFT keypoint positions and descriptors of a photo shrunk to ``long_side`` (optionally mirrored)."""
    import cv2

    side = _CHECK["long_side"]
    with Image.open(path) as image:
        image.draft("L", (side, side))
        image = ImageOps.exif_transpose(image).convert("L")
        image.thumbnail((side, side), Image.Resampling.BICUBIC)
    gray = np.asarray(image)
    if mirror:
        gray = np.ascontiguousarray(gray[:, ::-1])
    found, descriptors = cv2.SIFT_create(nfeatures=_CHECK["max_features"]).detectAndCompute(gray, None)
    return np.float32([k.pt for k in found]).reshape(-1, 2), descriptors


def inliers(a: tuple[np.ndarray, np.ndarray | None], b: tuple[np.ndarray, np.ndarray | None]) -> int:
    """Keypoint matches (ratio test) that agree with one RANSAC homography from photo a to photo b."""
    import cv2

    (points_a, desc_a), (points_b, desc_b) = a, b
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
            if m.distance < _CHECK["ratio"] * n.distance]
    if len(good) < 4:
        return 0
    cv2.setRNGSeed(0)  # RANSAC is random: same answer for a pair whatever order pairs are checked in
    _, mask = cv2.findHomography(np.float32([points_a[m.queryIdx] for m in good]),
                                 np.float32([points_b[m.trainIdx] for m in good]), cv2.RANSAC,
                                 _CHECK["ransac_px"])
    return 0 if mask is None else int(mask.sum())


def check_pair(paths: tuple[str, str]) -> int:
    """Most inliers between photo a and photo b, or photo a and the mirrored photo b."""
    a = keypoints(paths[0], False)
    return max(inliers(a, keypoints(paths[1], False)), inliers(a, keypoints(paths[1], True)))


def run_check(pairs: pd.DataFrame, lists: dict[str, pd.DataFrame], check: dict[str, Any],
              workers: int | None = None) -> np.ndarray:
    """Inlier count for every pair (pairs are sorted by photo a, so each worker reuses its keypoints)."""
    path = path_lookup(lists)
    jobs = [(path[(da, a)], path[(db, b)])
            for da, a, db, b in pairs[["dataset_a", "photo_a", "dataset_b", "photo_b"]].itertuples(False)]
    with Pool(workers, initializer=_init_check, initargs=(check,)) as pool:
        return np.array(pool.map(check_pair, jobs, chunksize=64), dtype=int)


# ---------------------------------------------------------------- review queue
def review_queue(candidates: pd.DataFrame, n_sample: int, seed: int) -> pd.DataFrame:
    """Every passed pair plus a random sample of rejected pairs per comparison, shuffled, without results."""
    rng = np.random.default_rng(seed)
    parts = [candidates[candidates["passed"]]]
    for _, group in candidates[~candidates["passed"]].groupby("comparison"):
        parts.append(group.iloc[np.sort(rng.choice(len(group), min(n_sample, len(group)), replace=False))])
    queue = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return queue[["pair_id", "comparison", "dataset_a", "photo_a", "dataset_b", "photo_b"]]


def flag(config: dict[str, Any], workers: int | None = None) -> dict[str, Any]:
    """Flag, check and queue. Refuses to replace a queue that anyone has started reviewing."""
    settings = config["duplicates"]
    queue_path = REPO_ROOT / settings["review_file"]
    if queue_path.exists() and any((REPO_ROOT / settings["review_dir"]).glob("*.csv")):
        raise RuntimeError(f"{queue_path} is frozen: reviews exist in {settings['review_dir']}")
    lists = {name: photos(config, name) for name in DATASETS}
    hashes, vectors = hash_cache(config, lists, workers), descriptor_cache(config, lists, workers)
    candidates = flag_pairs(lists, hashes, vectors, settings["max_hash_bits"], settings["min_cosine"])
    candidates["inliers"] = run_check(candidates, lists, settings["check"], workers)
    candidates["passed"] = candidates["inliers"] >= settings["check"]["min_inliers"]
    queue = review_queue(candidates, settings["rejected_sample"], config["seed"])
    candidates.to_csv(REPO_ROOT / settings["candidates_file"], index=False)
    queue.to_csv(queue_path, index=False)
    return {"candidates": candidates, "queue": queue}


# ---------------------------------------------------------------- reviewing
def read_reviews(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return pd.read_csv(path, dtype=str)


def show(path: str) -> Image.Image:
    with Image.open(path) as image:
        image.draft("RGB", (1200, 1200))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1200, 1200))
        return image


def ask_pairs(pairs: pd.DataFrame, config: dict[str, Any], who: str, target: Path, ask: Any) -> bool:
    """Show each pair not yet answered in ``target`` and save every answer at once; True when all are done."""
    path = path_lookup({name: photos(config, name) for name in DATASETS})
    rows = read_reviews(target)
    todo = pairs[~pairs["pair_id"].isin(rows["pair_id"])]
    for number, pair in enumerate(todo.itertuples(index=False), start=len(pairs) - len(todo) + 1):
        images = [show(path[(pair.dataset_a, pair.photo_a)]), show(path[(pair.dataset_b, pair.photo_b)])]
        answer, _ = ask(images, f"{who}: pair {number}/{len(pairs)}\n{INSTRUCTIONS}",
                        {"s": "same", "d": "different"})
        if answer == review_tool.STOP:
            print("Stopped; progress saved.")
            return False
        row = {"pair_id": pair.pair_id, "reviewer": who, "verdict": answer,
               "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        rows = pd.concat([rows, pd.DataFrame([row], columns=REVIEW_COLUMNS)], ignore_index=True)
        rows.to_csv(target.with_suffix(".tmp"), index=False)  # rewritten whole, then swapped in atomically
        os.replace(target.with_suffix(".tmp"), target)
    return True


def review(config: dict[str, Any], reviewer: str, ask: Any = review_tool.ask) -> None:
    """One reviewer judges every queued pair alone, in their own shuffled order."""
    settings = config["duplicates"]
    queue = pd.read_csv(REPO_ROOT / settings["review_file"], dtype=str)
    name_seed = int.from_bytes(hashlib.sha256(reviewer.encode()).digest()[:4], "big")
    order = np.random.default_rng([config["seed"], name_seed]).permutation(len(queue))
    target = REPO_ROOT / settings["review_dir"] / f"{reviewer}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if ask_pairs(queue.iloc[order], config, reviewer, target, ask):
        print("All pairs reviewed.")


def verdicts(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(queue, one row per queued pair with each reviewer's verdict); raises on unexpected reviewers."""
    settings = config["duplicates"]
    queue = pd.read_csv(REPO_ROOT / settings["review_file"], dtype=str)
    files = sorted((REPO_ROOT / settings["review_dir"]).glob("*.csv"))
    reviews = [read_reviews(f) for f in files if f.stem != SETTLED]
    data = pd.concat(reviews, ignore_index=True) if reviews else pd.DataFrame(columns=REVIEW_COLUMNS)
    names = sorted(data["reviewer"].unique())
    if any(n not in config["reviewers"] for n in names) or len(names) > 2:
        raise ValueError(f"reviewers {names}: expected two of {config['reviewers']}")
    if data.duplicated(["pair_id", "reviewer"]).any():
        raise ValueError("a pair was answered twice by the same reviewer")
    return queue, data.pivot(index="pair_id", columns="reviewer", values="verdict").reindex(queue["pair_id"])


def settle(config: dict[str, Any], ask: Any = review_tool.ask) -> None:
    """Both reviewers together decide every pair they disagreed on."""
    queue, table = verdicts(config)
    if table.shape[1] != 2 or table.isna().any().any():
        raise RuntimeError("both reviewers must finish every pair before settling")
    split = queue[queue["pair_id"].isin(table.index[table.iloc[:, 0] != table.iloc[:, 1]])]
    target = REPO_ROOT / config["duplicates"]["review_dir"] / f"{SETTLED}.csv"
    if ask_pairs(split, config, SETTLED, target, ask):
        print(f"All {len(split)} disagreements settled.")


# ---------------------------------------------------------------- results
def groups_of(photo_ids: pd.Series, confirmed: pd.DataFrame) -> pd.Series:
    """group_id per photo (photo_ids sorted): connected components of confirmed pairs, numbered in order
    of each group's first photo_id."""
    index = {pid: i for i, pid in enumerate(photo_ids)}
    a, b = confirmed["photo_a"].map(index).to_numpy(int), confirmed["photo_b"].map(index).to_numpy(int)
    _, labels = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(index),) * 2),
                                     directed=False)
    first = pd.Series(photo_ids.to_numpy()).groupby(labels).transform("min")
    numbers = {pid: n for n, pid in enumerate(sorted(first.unique()), start=1)}
    return first.map(lambda pid: f"g{numbers[pid]:04d}")


def combine(config: dict[str, Any]) -> dict[str, Any]:
    """Final verdicts, groups, label conflicts and cross-dataset overlap; writes nothing until complete."""
    settings = config["duplicates"]
    queue, table = verdicts(config)
    answered = table.notna().sum().to_dict()
    if table.shape[1] != 2 or table.isna().any().any():
        return {"complete": False, "answered": answered, "queued": len(queue)}
    first, second = table.columns
    settled = read_reviews(REPO_ROOT / settings["review_dir"] / f"{SETTLED}.csv")
    settled = settled.set_index("pair_id")["verdict"]
    agree = table[first] == table[second]
    final = table[first].where(agree, settled.reindex(table.index))
    if final.isna().any():
        return {"complete": False, "answered": answered, "queued": len(queue),
                "unsettled": int(final.isna().sum())}

    candidates = pd.read_csv(REPO_ROOT / settings["candidates_file"], dtype={"pair_id": str})
    pairs = queue.merge(candidates[["pair_id", "class_a", "class_b", "hash_bits", "cosine", "by_hash",
                                    "by_cosine", "inliers", "passed"]], on="pair_id")
    added = {first: table[first], second: table[second], "settled": ~agree, "final": final}
    for column, values in added.items():
        pairs[column] = pairs["pair_id"].map(values)
    pairs = pairs.sort_values("pair_id", ignore_index=True)
    confirmed = pairs[pairs["final"] == "same"]
    primary = photos(config, "primary")
    groups = primary[["photo_id", "class"]].assign(
        group_id=groups_of(primary["photo_id"], confirmed[confirmed["comparison"] == "primary"]))
    groups["group_size"] = groups.groupby("group_id")["photo_id"].transform("size")
    mixed = groups.groupby("group_id")["class"].transform("nunique") > 1
    overlap = confirmed[confirmed["comparison"] == "cross"]
    pairs.to_csv(REPO_ROOT / settings["pairs_file"], index=False)
    groups.to_csv(REPO_ROOT / settings["groups_file"], index=False)
    groups[mixed].to_csv(REPO_ROOT / settings["conflicts_file"], index=False)
    overlap.to_csv(REPO_ROOT / settings["overlap_file"], index=False)
    kappa = float(cohen_kappa_score(table[first], table[second]))
    return {"complete": True, "pairs": pairs, "groups": groups, "conflicts": groups[mixed],
            "overlap": overlap, "kappa": kappa, "reviewers": [first, second]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["flag", "review", "settle", "combine"])
    parser.add_argument("--reviewer", help="your first name in lower case (for 'review')")
    parser.add_argument("--config", default="configs/audits.yaml")
    parser.add_argument("--workers", type=int, help="processes to use (default: all CPU cores)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "flag":
        result = flag(config, args.workers)
        for comparison, group in result["candidates"].groupby("comparison"):
            print(f"{comparison}: {len(group)} flagged ({int(group['by_hash'].sum())} by hash, "
                  f"{int(group['by_cosine'].sum())} by cosine), {int(group['passed'].sum())} pass the check")
        print(f"review queue: {len(result['queue'])} pairs -> {config['duplicates']['review_file']}")
    elif args.action == "review":
        name = (args.reviewer or "").strip().lower()
        if name not in config["reviewers"]:
            parser.error(f"--reviewer must be one of {config['reviewers']}")
        review(config, name)
    elif args.action == "settle":
        settle(config)
    else:
        result = combine(config)
        if not result["complete"]:
            print(f"Not complete: {result}")
            return
        print(f"{int((result['pairs']['final'] == 'same').sum())} confirmed pairs; "
              f"{result['groups']['group_id'].nunique()} groups; {len(result['conflicts'])} photos in label "
              f"conflicts; {len(result['overlap'])} cross-dataset matches; kappa {result['kappa']:.2f}")


if __name__ == "__main__":
    main()
