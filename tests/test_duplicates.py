"""Checks for src/duplicates.py: orientation hashes, flagging, the automatic check, the blind queue,
the review loop and the final groups (synthetic photos; no real data or GPU needed)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import duplicates as dup
from src import review_tool
from src.inventory import load_for_hash, perceptual_hash

CHECK = {"long_side": 512, "contrast_threshold": 0.01, "max_features": 2000, "ratio": 0.75, "ransac_px": 8.0,
         "min_scale": 0.25, "max_scale": 4.0,
         "min_inliers": 15}


def textured(seed, size=600):
    """A photo-like image with plenty of corners for keypoints (random blobs on a gradient)."""
    rng = np.random.default_rng(seed)
    image = np.tile(np.linspace(60, 200, size, dtype=np.uint8)[None, :, None], (size, 1, 3))
    for _ in range(300):
        y, x, r = rng.integers(0, size, 2).tolist() + [int(rng.integers(3, 15))]
        image[max(0, y - r):y + r, max(0, x - r):x + r] = rng.integers(0, 255, 3)
    return Image.fromarray(image)


@pytest.fixture
def photos_on_disk(tmp_path):
    base = textured(1)
    images = {
        "a.jpg": base,
        "a_rotated.jpg": base.transpose(Image.Transpose.ROTATE_90),
        "a_mirrored_crop.jpg": base.crop((40, 30, 560, 580)).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "other.jpg": textured(2),
    }
    paths = {}
    for name, image in images.items():
        image.save(tmp_path / name, quality=95)
        paths[name] = str(tmp_path / name)
    return paths


def test_upright_orientation_hash_is_the_stage1_hash_and_rotations_match(photos_on_disk):
    hashes = dup.orientation_hashes(photos_on_disk["a.jpg"], hash_size=8, draft_size=256)
    stage1 = int(perceptual_hash(load_for_hash(photos_on_disk["a.jpg"], 256), 8), 16)
    assert len(hashes) == 8 and hashes[0] == stage1
    rotated = dup.orientation_hashes(photos_on_disk["a_rotated.jpg"], hash_size=8, draft_size=256)
    bits = dup.hash_bits(np.array([rotated[0]], dtype=np.uint64), np.array([hashes], dtype=np.uint64))
    assert bits[0, 0] <= 2  # the rotated copy matches one of the original's orientations


def frame(ids, classes):
    return pd.DataFrame({"photo_id": ids, "class": classes, "path": ids, "phash": "0"})


def test_flag_pairs_uses_either_rule_once_per_pair_and_never_self():
    lists = {"primary": frame(["A", "B", "C"], ["healthy", "hispa", "healthy"]),
             "paddy": frame(["P"], ["normal"])}
    far = np.uint64(0xFFFFFFFFFFFFFFFF)
    hashes = {"primary": np.array([[0] * 8, [1] * 8, [far] * 8], dtype=np.uint64),  # A~B by hash (1 bit)
              "paddy": np.array([[0xFFFFFFFF] * 8], dtype=np.uint64)}  # 31-32 bits from everyone
    vectors = {"primary": np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32),  # no pair above 0.95
               "paddy": np.array([[0.6, 0.8]], dtype=np.float32)}  # C~P by cosine (1.0)
    pairs = dup.flag_pairs(lists, hashes, vectors, max_bits=8, min_cosine=0.95)
    got = {(r.comparison, r.photo_a, r.photo_b, r.by_hash, r.by_cosine) for r in pairs.itertuples()}
    assert got == {("primary", "A", "B", True, False), ("cross", "C", "P", False, True)}
    assert set(pairs["pair_id"]) == {"p000000", "x000000"}


def test_check_passes_rotated_and_mirrored_crops_but_not_a_different_photo(photos_on_disk, tmp_path):
    names = list(photos_on_disk)
    lists = {"primary": pd.DataFrame({"photo_id": names, "path": [photos_on_disk[n] for n in names]})}
    pairs = pd.DataFrame({"dataset_a": "primary", "photo_a": "a.jpg", "dataset_b": "primary",
                          "photo_b": ["a_rotated.jpg", "a_mirrored_crop.jpg", "other.jpg"]})
    counts = dup.run_check(pairs, lists, CHECK, tmp_path / "sift", workers=2)
    assert (counts[:2] >= CHECK["min_inliers"]).all() and counts[2] < CHECK["min_inliers"]
    again = dup.run_check(pairs, lists, CHECK, tmp_path / "sift", workers=1)
    assert (again == counts).all()  # the same answer however many workers


def test_check_rejects_every_point_collapsing_onto_one_spot():
    """The first full run passed different leaves with hundreds of "inliers": every keypoint of photo a
    matched to one spot of photo b, which RANSAC fitted with a near-zero scale."""
    dup._init_check(CHECK)
    rng = np.random.default_rng(0)
    descriptors = rng.integers(0, 255, (50, 128)).astype(np.uint8)
    points = rng.uniform(0, 500, (50, 2)).astype(np.float32)
    one_spot = np.float32([[250, 250]]) + rng.uniform(0, 0.4, (50, 2)).astype(np.float32)
    assert dup.inliers((points, descriptors), (one_spot, descriptors)) == 0
    shifted = points * 1.5 + 20  # a genuine resize and shift keeps every point
    assert dup.inliers((points, descriptors), (shifted, descriptors)) == 50


def candidates(n_passed, n_rejected):
    rows = [{"pair_id": f"p{i:06d}", "comparison": "primary", "dataset_a": "primary", "photo_a": f"A{i}",
             "dataset_b": "primary", "photo_b": f"B{i}", "passed": i < n_passed}
            for i in range(n_passed + n_rejected)]
    return pd.DataFrame(rows)


def test_queue_has_every_passed_pair_a_sample_of_the_rest_and_hides_results():
    queue = dup.review_queue(candidates(5, 50), n_sample=10, seed=0)
    assert len(queue) == 15 and queue["pair_id"].is_unique
    assert {f"p{i:06d}" for i in range(5)} <= set(queue["pair_id"])
    assert "passed" not in queue.columns and "inliers" not in queue.columns
    assert queue.equals(dup.review_queue(candidates(5, 50), n_sample=10, seed=0))  # reproducible


@pytest.fixture
def review_setup(tmp_path, monkeypatch):
    """A config whose queue, review folder and outputs live in tmp_path; photos are blank images."""
    monkeypatch.setattr(dup, "REPO_ROOT", tmp_path)
    ids = ["H/1.jpg", "H/2.jpg", "S/3.jpg", "S/4.jpg"]
    lists = {"primary": frame(ids, ["healthy", "healthy", "hispa", "healthy"]),
             "paddy": frame(["P/9.jpg"], ["normal"])}
    monkeypatch.setattr(dup, "photos", lambda config, name: lists[name])
    monkeypatch.setattr(review_tool, "load_photo", lambda path, side, mode="RGB": Image.new("RGB", (4, 4)))
    pairs = [("p000000", "primary", "primary", "H/1.jpg", "primary", "H/2.jpg"),
             ("p000001", "primary", "primary", "H/2.jpg", "primary", "S/3.jpg"),
             ("p000002", "primary", "primary", "S/3.jpg", "primary", "S/4.jpg"),
             ("x000000", "cross", "primary", "H/1.jpg", "paddy", "P/9.jpg")]
    columns = ["pair_id", "comparison", "dataset_a", "photo_a", "dataset_b", "photo_b"]
    queue = pd.DataFrame(pairs, columns=columns).assign(round=1)
    queue.to_csv(tmp_path / "queue.csv", index=False)
    # p000002 was rejected by the check (sampled for review); p000003/4 were rejected and not sampled.
    extra = pd.DataFrame([("p000003", "primary", "primary", "H/1.jpg", "primary", "S/4.jpg"),
                          ("p000004", "primary", "primary", "H/2.jpg", "primary", "S/4.jpg")],
                         columns=columns)
    classes = {**dict(zip(ids, lists["primary"]["class"])), "P/9.jpg": "normal"}
    cand = pd.concat([queue.drop(columns="round"), extra], ignore_index=True)
    cand = cand.assign(class_a=cand["photo_a"].map(classes), class_b=cand["photo_b"].map(classes),
                       hash_bits=3, cosine=0.9, by_hash=True, by_cosine=False,
                       inliers=[50, 50, 25, 50, 20, 28],
                       passed=[True, True, False, True, False, False])
    cand.to_csv(tmp_path / "candidates.csv", index=False)
    config = {"seed": 0, "reviewers": ["ahsan", "harris", "umer"],
              "duplicates": {"review_file": "queue.csv", "review_dir": "reviews",
                             "candidates_file": "candidates.csv",
                             "pairs_file": "pairs.csv", "groups_file": "groups.csv",
                             "conflicts_file": "conflicts.csv", "overlap_file": "overlap.csv"}}
    return config, tmp_path


def answering(answers):
    """A fake ``ask`` that gives the next answer each time, recording the titles it was shown."""
    shown = []

    def ask(images, title, choices, toggles=None):
        shown.append(title)
        return next(answers), set()
    return ask, shown


def test_review_saves_resumes_and_combine_builds_groups_conflicts_and_overlap(review_setup):
    config, tmp = review_setup
    ask, shown = answering(iter(["different", review_tool.UNDO, "same", review_tool.STOP]))
    dup.review(config, "ahsan", ask=ask)
    saved = pd.read_csv(tmp / "reviews" / "ahsan.csv")
    assert saved["verdict"].tolist() == ["same"] and shown[0] == shown[2]  # undo re-asks the same pair
    assert {"photo_a", "photo_b"} <= set(saved.columns)  # verdicts carry their photos
    assert dup.combine(config)["complete"] is False

    # Both say "same" for every pair except p000002, where they disagree; settling makes it "same".
    queue = pd.read_csv(tmp / "queue.csv")
    for who, p2 in (("ahsan", "same"), ("harris", "different")):
        rows = queue.assign(reviewer=who, reviewed_at="t",
                            verdict=[p2 if pid == "p000002" else "same" for pid in queue["pair_id"]])
        rows[dup.REVIEW_COLUMNS].to_csv(tmp / "reviews" / f"{who}.csv", index=False)
    assert dup.combine(config) == {"complete": False, "answered": {"ahsan": 4, "harris": 4}, "queued": 4,
                                   "unsettled": 1}
    ask, shown = answering(iter(["same"]))
    dup.settle(config, ask=ask)
    assert len(shown) == 1  # only the disagreement is shown

    result = dup.combine(config)
    groups = result["groups"].set_index("photo_id")["group_id"]
    assert groups.nunique() == 1  # 1-2, 2-3 and 3-4 chain all four photos into one group
    # Only photos in a confirmed pair with different classes are excluded: 2-3 and 3-4 (hispa vs healthy).
    assert sorted(result["conflicts"]["photo_id"]) == ["H/2.jpg", "S/3.jpg", "S/4.jpg"]
    assert result["overlap"]["photo_b"].tolist() == ["P/9.jpg"]
    assert (tmp / "groups.csv").exists() and (tmp / "overlap.csv").exists()

    # Miss rule: the sampled rejected pair p000002 (25 inliers) was confirmed, so pairs with >= 25 inliers
    # that were never queued join as round 2: p000004 (28), not p000003 (20). Applied once.
    misses = result["misses"].loc["primary"]
    assert (misses["sampled"], misses["confirmed"], misses["lowest_inliers"]) == (1, 1, 25)
    assert dup.extend(config)["pair_id"].tolist() == ["p000004"]
    assert dup.outcome(config)["complete"] is False  # round 2 still needs both reviewers
    with pytest.raises(SystemExit, match="once"):
        dup.extend(config)


def test_a_third_reviewer_is_stopped_before_starting_and_verdicts_must_match_the_queue(review_setup):
    config, tmp = review_setup
    for who in ("ahsan", "harris"):
        dup.review(config, who, ask=answering(iter(["same", review_tool.STOP]))[0])
    with pytest.raises(SystemExit, match="already"):
        dup.review(config, "umer", ask=answering(iter([]))[0])
    queue = pd.read_csv(tmp / "queue.csv")
    queue.assign(photo_b=queue["photo_a"]).to_csv(tmp / "queue.csv", index=False)  # queue rebuilt differently
    with pytest.raises(ValueError, match="do not match"):
        dup.outcome(config)


def test_unknown_reviewer_is_refused(review_setup):
    config, tmp = review_setup
    (tmp / "reviews").mkdir()
    pd.DataFrame([{"pair_id": "p000000", "photo_a": "H/1.jpg", "photo_b": "H/2.jpg", "reviewer": "bob",
                   "verdict": "same", "reviewed_at": "t"}]).to_csv(tmp / "reviews" / "bob.csv", index=False)
    with pytest.raises(ValueError, match="reviewers"):
        dup.combine(config)


def test_groups_number_singletons_and_components_by_first_photo():
    ids = pd.Series(["a", "b", "c", "d", "e"])
    confirmed = pd.DataFrame({"photo_a": ["d", "b"], "photo_b": ["e", "c"]})
    assert dup.groups_of(ids, confirmed).tolist() == ["g0001", "g0002", "g0002", "g0003", "g0003"]
