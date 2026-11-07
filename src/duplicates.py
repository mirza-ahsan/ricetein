"""Stage 2b/2c: near-duplicates within the primary dataset, and overlap with Paddy Doctor.

1. ``flag``: a pair is flagged by either pre-declared rule (pre-declared methodology, "Data audits"):
   64-bit perceptual hashes differ in at most ``max_hash_bits`` bits, comparing each photo with all
   8 rotations/flips of the other; or the whole-photo E1 class tokens have cosine > ``min_cosine``.
   Flagged pairs get an automatic check: SIFT keypoints matched with Lowe's ratio test, then a RANSAC
   similarity transform (also against the mirrored photo); a pair passes with at least ``min_inliers``
   distinct inlier keypoints.
   The review queue holds every passed pair plus a random sample of rejected ones (to measure what the
   check misses), shuffled and blind: reviewers see two photos, not classes or check results.
   Deviation from "review every flagged pair": handbook §24, 2026-11-07.
2. ``review --reviewer <name>``: two people each judge every queued pair alone (Y same leaf / N different).
3. ``settle``: both look together at the pairs they disagreed on.
4. ``extend``: the declared miss rule, applied once: if people confirmed any sampled rejected pair, every
   pair with at least the lowest inlier count among those misses is queued as round 2 (then 2-3 again).
5. ``combine``: confirmed primary pairs form groups (connected components; singletons included); the two
   photos of any confirmed pair with different classes are label conflicts (excluded from all experiments);
   confirmed cross-dataset pairs are the overlap list.

    uv run python -m src.duplicates flag                       # caches, flags, check, review queue
    uv run python -m src.duplicates review --reviewer <name>   # each reviewer, alone
    uv run python -m src.duplicates settle                     # both reviewers together
    uv run python -m src.duplicates extend                     # once, after round 1 is settled
    uv run python -m src.duplicates combine                    # pairs, groups, conflicts, overlap
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src import review_tool
from src.config import REPO_ROOT, dataset_dir, load_config
from src.inventory import load_for_hash, perceptual_hash, read_inventory

DATASETS = ("primary", "paddy")
REVIEW_COLUMNS = ["pair_id", "photo_a", "photo_b", "reviewer", "verdict", "reviewed_at"]
QUEUE_COLUMNS = ["pair_id", "comparison", "dataset_a", "photo_a", "dataset_b", "photo_b", "round"]
INSTRUCTIONS = ("Y same leaf (a copy, crop, resize, rotation, mirror or re-shot of one leaf)   "
                "N different leaf   Backspace = undo   Esc = stop")
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


def cached(target: Path, lists: dict[str, pd.DataFrame], made_with: dict[str, Any],
           build: Callable[[str, pd.DataFrame], np.ndarray]) -> dict[str, np.ndarray]:
    """{dataset: array} from a git-ignored cache; rebuilt when the photo lists or the settings it was made
    with (``made_with``) no longer match it."""
    key = json.dumps(made_with, sort_keys=True)
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            same_photos = all(saved[f"{n}_ids"].tolist() == f["photo_id"].tolist() for n, f in lists.items())
            if same_photos and str(saved["made_with"]) == key:
                return {n: saved[n] for n in lists}
    arrays = {name: build(name, frame) for name, frame in lists.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, made_with=np.array(key), **arrays,
             **{f"{n}_ids": f["photo_id"].to_numpy(str) for n, f in lists.items()})
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
            return np.array(pool.map(work, frame["path"], chunksize=32), dtype=np.uint64)

    hashes = cached(REPO_ROOT / config["duplicates"]["hash_cache"], lists, config["hashing"], build)
    for name, frame in lists.items():  # a different decode moves hashes by up to 4 of the 8 allowed bits
        if not np.array_equal(hashes[name][:, 0], np.array([int(h, 16) for h in frame["phash"]], np.uint64)):
            raise RuntimeError(f"{name}: upright hashes differ from the Stage 1 inventory")
    return hashes


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

    made_with = {"encoder": config["encoders"][code], "size": settings["whole_size"],
                 "normalisation": config["normalisation"]}
    return cached(REPO_ROOT / settings["descriptor_cache"], lists, made_with, build)


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
_CHECK: dict[str, Any] = {}  # settings, plus (in matching workers) memory-mapped keypoints and their index


def _init_check(check: dict[str, Any], features: dict[str, Any] | None = None) -> None:
    import cv2

    cv2.setNumThreads(1)
    _CHECK.update(check)
    if features:
        rows = features["rows"]
        _CHECK["points"] = np.memmap(features["points"], np.float32, "r", shape=(rows, 2))
        _CHECK["descriptors"] = np.memmap(features["descriptors"], np.uint8, "r", shape=(rows, 128))
        _CHECK["index"] = features["index"]


def keypoints(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """SIFT keypoint positions and descriptors (uint8: SIFT values are whole numbers 0-255) of a photo shrunk
    to ``long_side``: [as is, mirrored]."""
    import cv2

    image = review_tool.load_photo(path, _CHECK["long_side"], "L")
    sift = cv2.SIFT_create(nfeatures=_CHECK["max_features"], contrastThreshold=_CHECK["contrast_threshold"])
    out = []
    for gray in (np.asarray(image), np.ascontiguousarray(np.asarray(image)[:, ::-1])):
        found, descriptors = sift.detectAndCompute(gray, None)
        descriptors = np.zeros((0, 128), np.uint8) if descriptors is None else descriptors.astype(np.uint8)
        out.append((np.float32([k.pt for k in found]).reshape(-1, 2), descriptors))
    return out


def stored(path: str, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    start, stop = _CHECK["index"][(path, mirror)]
    return _CHECK["points"][start:stop], _CHECK["descriptors"][start:stop]


def inliers(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> int:
    """Distinct keypoints that agree with one similarity transform (rotation, scale, shift) from photo a to
    photo b: matches pass Lowe's ratio test, each point of b is used once, the transform's scale must be
    plausible, and inliers are counted as distinct positions in both photos. These guards matter: without
    them RANSAC accepted a near-zero scale that maps every point of a onto one spot of b (hundreds of
    "inliers" between different leaves)."""
    import cv2

    (points_a, desc_a), (points_b, desc_b) = a, b
    if len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(np.float32(desc_a), np.float32(desc_b), k=2)
    best: dict[int, Any] = {}
    passing = [m for m, n in matches if m.distance < _CHECK["ratio"] * n.distance]
    for m in sorted(passing, key=lambda m: m.distance):  # each point of b keeps only its closest match
        best.setdefault(m.trainIdx, m)
    if len(best) < 4:
        return 0
    pa = np.float32([points_a[m.queryIdx] for m in best.values()])
    pb = np.float32([points_b[m.trainIdx] for m in best.values()])
    cv2.setRNGSeed(0)  # RANSAC is random: same answer for a pair whatever order pairs are checked in
    transform, mask = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                                  ransacReprojThreshold=_CHECK["ransac_px"])
    if transform is None or not _CHECK["min_scale"] <= np.hypot(*transform[:, 0]) <= _CHECK["max_scale"]:
        return 0
    keep = mask.ravel().astype(bool)
    return min(len(np.unique(pa[keep].round(), axis=0)), len(np.unique(pb[keep].round(), axis=0)))


def check_pair(paths: tuple[str, str]) -> int:
    """Most inliers between photo a and photo b, or photo a and the mirrored photo b."""
    a = stored(paths[0], False)
    return max(inliers(a, stored(paths[1], False)), inliers(a, stored(paths[1], True)))


def keypoint_files(needed: list[str], check: dict[str, Any], folder: Path,
                   workers: int | None) -> dict[str, Any]:
    """Keypoints of every photo in ``needed`` (as is and mirrored), written to raw files in ``folder`` as they
    arrive (one photo in memory at a time) and reused while the photos and settings are unchanged. Returns
    what the matching workers need: file paths, row count and the (path, mirrored) -> rows index."""
    from tqdm import tqdm

    meta_path = folder / "meta.json"
    made_with = {"check": check, "photos": needed}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["made_with"] == made_with:
            return {**meta["files"], "index": {(p, m == "1"): tuple(v) for (p, m), v in meta["index"]}}
    folder.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)  # a half-written cache must never look complete
    files = {"points": str(folder / "points.f32"), "descriptors": str(folder / "descriptors.u8")}
    index, rows = {}, 0
    with (open(files["points"], "wb") as points, open(files["descriptors"], "wb") as descriptors,
          get_context("spawn").Pool(workers, initializer=_init_check, initargs=(check,)) as pool):
        found = pool.imap(keypoints, needed, chunksize=8)
        for p, both in tqdm(zip(needed, found), total=len(needed), desc="keypoints"):
            for mirror, (pts, desc) in zip((False, True), both):
                index[(p, mirror)] = (rows, rows + len(desc))
                rows += len(desc)
                points.write(pts.tobytes())
                descriptors.write(desc.tobytes())
    files["rows"] = rows
    meta = {"made_with": made_with, "files": files,
            "index": [[(p, "1" if m else "0"), list(v)] for (p, m), v in index.items()]}
    meta_path.write_text(json.dumps(meta))
    return {**files, "index": index}


def run_check(pairs: pd.DataFrame, lists: dict[str, pd.DataFrame], check: dict[str, Any], folder: Path,
              workers: int | None = None) -> np.ndarray:
    """Inlier count for every pair. Keypoints are found once per photo (``keypoint_files``); the matching
    workers read them memory-mapped (shared, not copied). Spawned, not forked: the parent may already run
    GPU and loader threads."""
    from tqdm import tqdm

    path = path_lookup(lists)
    jobs = [(path[(da, a)], path[(db, b)])
            for da, a, db, b in pairs[["dataset_a", "photo_a", "dataset_b", "photo_b"]].itertuples(False)]
    features = keypoint_files(sorted({p for job in jobs for p in job}), check, folder, workers)
    with get_context("spawn").Pool(workers, initializer=_init_check, initargs=(check, features)) as pool:
        counts = list(tqdm(pool.imap(check_pair, jobs, chunksize=256), total=len(jobs), desc="pairs"))
    return np.array(counts, dtype=int)


# ---------------------------------------------------------------- review queue
def review_queue(candidates: pd.DataFrame, n_sample: int, seed: int) -> pd.DataFrame:
    """Every passed pair plus a random sample of rejected pairs per comparison, shuffled, without results."""
    rng = np.random.default_rng(seed)
    parts = [candidates[candidates["passed"]]]
    for _, group in candidates[~candidates["passed"]].groupby("comparison"):
        parts.append(group.iloc[np.sort(rng.choice(len(group), min(n_sample, len(group)), replace=False))])
    queue = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True).assign(round=1)
    return queue[QUEUE_COLUMNS]


def flag(config: dict[str, Any], workers: int | None = None) -> dict[str, Any]:
    """Flag, check and queue. Refuses to replace a queue that anyone has started reviewing."""
    settings = config["duplicates"]
    queue_path = REPO_ROOT / settings["review_file"]
    if any((REPO_ROOT / settings["review_dir"]).glob("*.csv")):  # verdicts are tied to this exact queue
        raise RuntimeError(f"reviews exist in {settings['review_dir']}: the queue is frozen")
    lists = {name: photos(config, name) for name in DATASETS}
    hashes, vectors = hash_cache(config, lists, workers), descriptor_cache(config, lists, workers)
    candidates = flag_pairs(lists, hashes, vectors, settings["max_hash_bits"], settings["min_cosine"])
    candidates["inliers"] = run_check(candidates, lists, settings["check"],
                                      REPO_ROOT / settings["keypoint_cache"], workers)
    candidates["passed"] = candidates["inliers"] >= settings["check"]["min_inliers"]
    queue = review_queue(candidates, settings["rejected_sample"], config["seed"])
    candidates.to_csv(REPO_ROOT / settings["candidates_file"], index=False)
    queue.to_csv(queue_path, index=False)
    return {"candidates": candidates, "queue": queue}


# ---------------------------------------------------------------- reviewing
def ask_pairs(pairs: pd.DataFrame, config: dict[str, Any], who: str, ask: Any) -> bool:
    """Collect ``who``'s verdict on every pair (saved with both photo ids, so a verdict can never attach to a
    different pair); True when all are answered."""
    path = path_lookup({name: photos(config, name) for name in DATASETS})
    target = REPO_ROOT / config["duplicates"]["review_dir"] / f"{who}.csv"

    def make_row(pair: dict[str, Any], answer: str, flags: set[str]) -> dict[str, Any]:
        return {"pair_id": pair["pair_id"], "photo_a": pair["photo_a"], "photo_b": pair["photo_b"],
                "reviewer": who, "verdict": answer,
                "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    return review_tool.collect(
        pairs.to_dict("records"), "pair_id", target, REVIEW_COLUMNS,
        show=lambda pair: [review_tool.load_photo(path[(pair["dataset_a"], pair["photo_a"])], 1200),
                           review_tool.load_photo(path[(pair["dataset_b"], pair["photo_b"])], 1200)],
        title=lambda pair, n, total: f"{who}: pair {n}/{total}\n{INSTRUCTIONS}",
        choices={"y": "same", "n": "different"}, make_row=make_row, ask=ask)


def review(config: dict[str, Any], reviewer: str, ask: Any = review_tool.ask) -> None:
    """One reviewer judges every queued pair alone, in their own shuffled order."""
    settings = config["duplicates"]
    review_tool.refuse_newcomer(REPO_ROOT / settings["review_dir"], reviewer)
    queue = pd.read_csv(REPO_ROOT / settings["review_file"], dtype=str)
    if ask_pairs(queue.iloc[review_tool.person_order(config["seed"], reviewer, len(queue))], config,
                 reviewer, ask):
        print("All pairs reviewed.")


def verdicts(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(queue, one row per queued pair with each reviewer's verdict). Raises on unknown or extra reviewers,
    and on any saved verdict whose photos do not match the queue."""
    settings = config["duplicates"]
    queue = pd.read_csv(REPO_ROOT / settings["review_file"], dtype=str)
    folder = REPO_ROOT / settings["review_dir"]
    data = review_tool.people_rows(folder, REVIEW_COLUMNS, "reviewer", review_tool.allowed_people(config),
                                   "pair_id")
    settled = review_tool.read_rows(folder / f"{review_tool.SETTLED}.csv", REVIEW_COLUMNS)
    expected = queue.set_index("pair_id")[["photo_a", "photo_b"]]
    for rows in (data, settled):
        saved = rows.set_index("pair_id")[["photo_a", "photo_b"]]
        if not (saved.to_numpy() == expected.reindex(saved.index).to_numpy()).all():
            raise ValueError("saved verdicts do not match the review queue's photos")
    return queue, data.pivot(index="pair_id", columns="reviewer", values="verdict").reindex(queue["pair_id"])


def settle(config: dict[str, Any], ask: Any = review_tool.ask) -> None:
    """Both reviewers together decide every pair they disagreed on."""
    queue, table = verdicts(config)
    if not review_tool.both_done(table):
        raise SystemExit("both reviewers must finish every pair before settling")
    split = queue[queue["pair_id"].isin(review_tool.disagreements(table))]
    if ask_pairs(split, config, review_tool.SETTLED, ask):
        print(f"All {len(split)} disagreements settled.")


# ---------------------------------------------------------------- results
def groups_of(photo_ids: pd.Series, confirmed: pd.DataFrame) -> pd.Series:
    """group_id per photo (photo_ids sorted): connected components of confirmed pairs, numbered in order
    of each group's first photo_id."""
    index = {pid: i for i, pid in enumerate(photo_ids)}
    a, b = confirmed["photo_a"].map(index).to_numpy(int), confirmed["photo_b"].map(index).to_numpy(int)
    _, labels = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(index),) * 2),
                                     directed=False)
    first = pd.Series(photo_ids.to_numpy(), index=photo_ids.index).groupby(labels).transform("min")
    numbers = {pid: n for n, pid in enumerate(sorted(first.unique()), start=1)}
    return first.map(lambda pid: f"g{numbers[pid]:04d}")


def outcome(config: dict[str, Any]) -> dict[str, Any]:
    """Final verdicts, groups, label conflicts and cross-dataset overlap (reads only)."""
    settings = config["duplicates"]
    queue, table = verdicts(config)
    folder = REPO_ROOT / settings["review_dir"]
    settled = review_tool.settled_answers(folder, REVIEW_COLUMNS, "pair_id", "verdict")
    status = review_tool.two_person(table, settled, table.notna().sum().to_dict())
    if not status["complete"]:
        return {**status, "queued": len(queue)}

    candidates = pd.read_csv(REPO_ROOT / settings["candidates_file"], dtype={"pair_id": str})
    pairs = (queue.merge(candidates[["pair_id", "class_a", "class_b", "hash_bits", "cosine", "by_hash",
                                     "by_cosine", "inliers", "passed"]], on="pair_id")
             .join(table.assign(settled=~status["agree"], final=status["final"]), on="pair_id")
             .sort_values("pair_id", ignore_index=True))
    confirmed = pairs[pairs["final"] == "same"]
    primary = photos(config, "primary")
    groups = primary[["photo_id", "class"]].assign(
        group_id=groups_of(primary["photo_id"], confirmed[confirmed["comparison"] == "primary"]))
    groups["group_size"] = groups.groupby("group_id")["photo_id"].transform("size")
    # Label conflicts (Ahsan, 2026-11-08): only the two photos of a confirmed pair with different classes
    # are excluded; the rest of their group stays.
    clash = confirmed[(confirmed["comparison"] == "primary") & (confirmed["class_a"] != confirmed["class_b"])]
    conflicts = groups[groups["photo_id"].isin(set(clash["photo_a"]) | set(clash["photo_b"]))]
    overlap = confirmed[confirmed["comparison"] == "cross"]
    return {"complete": True, "pairs": pairs, "groups": groups, "conflicts": conflicts, "overlap": overlap,
            "kappa": status["kappa"], "reviewers": status["people"], "misses": misses(pairs)}


def misses(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per comparison: round-1 sampled rejected pairs, how many people confirmed (what the check missed),
    and the lowest inlier count among those confirmed."""
    sampled = pairs[~pairs["passed"] & (pairs["round"].astype(int) == 1)]
    same = sampled["final"] == "same"
    return pd.DataFrame({"sampled": sampled.groupby("comparison").size(),
                         "confirmed": same.groupby(sampled["comparison"]).sum(),
                         "lowest_inliers": sampled[same].groupby("comparison")["inliers"].min()})


def extend(config: dict[str, Any]) -> pd.DataFrame:
    """The declared miss rule (Ahsan, 2026-11-08), applied once after round 1 is reviewed and settled: if
    any sampled rejected pair was confirmed, every not-yet-queued pair with at least the lowest inlier count
    among those misses joins the queue as round 2. Returns the added pairs (empty when nothing was missed)."""
    settings = config["duplicates"]
    queue_path = REPO_ROOT / settings["review_file"]
    queue = pd.read_csv(queue_path, dtype=str)
    if (queue["round"] == "2").any():
        raise SystemExit("the miss rule was already applied (round 2 exists); it is applied once")
    result = outcome(config)
    if not result["complete"]:
        raise SystemExit("round 1 must be reviewed and settled first")
    candidates = pd.read_csv(REPO_ROOT / settings["candidates_file"], dtype={"pair_id": str})
    lowest = candidates["comparison"].map(result["misses"]["lowest_inliers"])  # NaN (no miss): never passes
    added = candidates[(candidates["inliers"] >= lowest) & ~candidates["pair_id"].isin(queue["pair_id"])]
    added = added.sample(frac=1, random_state=config["seed"]).assign(round=2)[QUEUE_COLUMNS]
    pd.concat([queue, added.astype(str)]).to_csv(queue_path, index=False)
    return added


def combine(config: dict[str, Any]) -> dict[str, Any]:
    """``outcome`` plus, once complete, the pairs, groups, conflicts and overlap files."""
    result = outcome(config)
    if result["complete"]:
        settings = config["duplicates"]
        for key, name in (("pairs", "pairs_file"), ("groups", "groups_file"), ("conflicts", "conflicts_file"),
                          ("overlap", "overlap_file")):
            result[key].to_csv(REPO_ROOT / settings[name], index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["flag", "review", "settle", "extend", "combine"])
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
        allowed = review_tool.allowed_people(config)
        review(config, review_tool.person_name(parser, args.reviewer, allowed, "--reviewer"))
    elif args.action == "settle":
        settle(config)
    elif args.action == "extend":
        added = extend(config)
        print(f"{len(added)} pairs added as round 2" if len(added) else "No misses: nothing to add.")
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
