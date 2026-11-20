"""Stage 12 checks on synthetic bags and a tiny head: no GPU, no caches, no research data."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import explain as ex
from src import train_mil as tm
from src.mil_models import MILModel

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]


class Store:
    """Bags of 20 to 30 tiles whose descriptors depend on the photo, as a stand-in for BagStore."""

    dim = 6

    def bag(self, photo_id, version):
        rng = np.random.default_rng(int(photo_id))
        return rng.normal(size=(20 + int(photo_id) % 11, self.dim)).astype(np.float16)


def folds(n=8):
    return pd.DataFrame(
        {"photo_id": [str(k) for k in range(n)], "class": (CLASSES * n)[:n], "round_1": ["test"] * n}
    )


def model(aggregator="attention", seed=0):
    torch.manual_seed(seed)
    return MILModel(Store.dim, 8, 0.0, aggregator, len(CLASSES)).eval()


def batch(n=4):
    bags = tm.Bags(Store(), folds(n), 1, "test", CLASSES, 1, 0.0, n, 0)
    tiles, mask, labels, rows = next(iter(bags))
    return bags, tiles, mask, labels, rows


# ------------------------------------------------------------------ attention and deletion


def test_attention_sums_to_one_over_real_tiles_and_is_zero_on_padding():
    _, tiles, mask, _, _ = batch()
    assert not mask.all()  # the bags differ in size, so there is padding
    _, weights = model()(tiles, mask)
    assert torch.allclose(weights.sum(1), torch.ones(len(weights)), atol=1e-5)
    assert (weights[~mask] == 0).all()


@pytest.mark.parametrize("k", ex.KS)
def test_top_k_removes_exactly_k_real_tiles_and_the_highest_weights(k):
    _, tiles, mask, _, _ = batch()
    _, weights = model()(tiles, mask)
    kept = ex.top_k_mask(weights, mask, k)
    assert ((mask.sum(1) - kept.sum(1)) == k).all()
    assert not (kept & ~mask).any()  # padding stays off
    for i in range(len(mask)):
        removed = torch.nonzero(mask[i] & ~kept[i]).flatten()
        expected = torch.argsort(weights[i].masked_fill(~mask[i], -1.0), descending=True)[:k]
        assert set(removed.tolist()) == set(expected.tolist())


@pytest.mark.parametrize("k", ex.KS)
def test_random_k_removes_exactly_k_real_tiles_and_never_touches_padding(k):
    _, _, mask, _, _ = batch()
    for seed in ex.RANDOM_SEEDS:
        kept = ex.random_k_mask(mask, k, torch.Generator().manual_seed(seed))
        assert ((mask.sum(1) - kept.sum(1)) == k).all()
        assert not (kept & ~mask).any()  # nothing switched on, and removed tiles were real ones


def test_random_deletion_is_repeatable_and_depends_on_the_seed():
    _, _, mask, _, _ = batch()
    a = ex.random_k_mask(mask, 4, torch.Generator().manual_seed(1))
    b = ex.random_k_mask(mask, 4, torch.Generator().manual_seed(1))
    c = ex.random_k_mask(mask, 4, torch.Generator().manual_seed(2))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_cannot_remove_all_tiles():
    _, _, mask, _, _ = batch()
    with pytest.raises(ValueError, match="cannot remove"):
        ex.top_k_mask(torch.rand(mask.shape), mask, int(mask.sum(1).min()))


@pytest.mark.parametrize("aggregator", ["attention", "mean", "max"])
def test_masking_tiles_equals_removing_them_from_the_bag(aggregator):
    _, tiles, mask, _, _ = batch()
    head = model(aggregator)
    _, weights = head(tiles, mask) if aggregator == "attention" else (None, torch.rand(mask.shape))
    kept = ex.top_k_mask(weights, mask, 4)
    masked_logits = head(tiles, kept)[0]
    for i in range(len(tiles)):
        shortened = tiles[i : i + 1, kept[i]]
        short_logits = head(shortened, torch.ones(shortened.shape[:2], dtype=torch.bool))[0]
        assert torch.allclose(masked_logits[i], short_logits[0], atol=1e-5)


def test_deletion_table_has_every_condition_and_matches_the_saved_attention():
    bags, tiles, mask, _, rows = batch()
    head = model()
    with torch.no_grad():
        saved = head(tiles, mask)[1]
    stored = {pid: w[:n].numpy() for pid, w, n in zip(bags.photo_ids[rows], saved, mask.sum(1))}
    long = ex.delete_and_score(head, bags, CLASSES, torch.device("cpu"), stored)
    per_photo = 1 + len(ex.KS) * (1 + len(ex.RANDOM_SEEDS))
    assert len(long) == 4 * per_photo
    assert set(long["condition"]) == {"none", "top", "random"}
    assert long["p_true"].between(0, 1).all()
    scores = ex.score_conditions(long, CLASSES)
    assert scores[["accuracy", "macro_f1", "p_true"]].apply(lambda c: c.between(0, 1).all()).all()
    photos = ex.photo_table(long, CLASSES)
    assert len(photos) == 4 * len(ex.KS) and {"p_top", "p_random", "p_full", "true_class"} <= set(photos)
    stored[next(iter(stored))] = stored[next(iter(stored))][::-1].copy()  # a wrong checkpoint/bag pairing
    with pytest.raises(ValueError, match="recomputed attention differs"):
        ex.delete_and_score(head, bags, CLASSES, torch.device("cpu"), stored)


def test_removing_top_tiles_changes_the_answer_at_least_as_much_as_a_tiny_random_change():
    bags, _, _, _, _ = batch()
    long = ex.delete_and_score(model(), bags, CLASSES, torch.device("cpu"))
    full = long[long["condition"] == "none"].set_index("photo_id")["p_true"]
    top = long[(long["condition"] == "top") & (long["k"] == 16)].set_index("photo_id")["p_true"]
    assert not np.allclose(full.loc[top.index], top)  # removing 16 tiles does change the probabilities


def test_deletion_summary_verdict_follows_rule_nine():
    rng = np.random.default_rng(0)
    runs = [(f"run{r}", r) for r in range(1, 6)]
    rows, photo_rows = [], []
    for run, r in runs:
        rows.append(
            {
                "run": run,
                "round": r,
                "condition": "none",
                "k": 0,
                "seed": -1,
                "macro_f1": 0.8,
                "accuracy": 0.8,
                "p_true": 0.7,
            }
        )
        rows.append(
            {
                "run": run,
                "round": r,
                "condition": "top",
                "k": 4,
                "seed": -1,
                "macro_f1": 0.4,
                "accuracy": 0.4,
                "p_true": 0.3,
            }
        )
        for seed in (0, 1):
            rows.append(
                {
                    "run": run,
                    "round": r,
                    "condition": "random",
                    "k": 4,
                    "seed": seed,
                    "macro_f1": 0.78,
                    "accuracy": 0.78,
                    "p_true": 0.69,
                }
            )
        for i in range(40):
            photo_rows.append(
                {
                    "photo_id": f"{run}_{i}",
                    "run": run,
                    "round": r,
                    "k": 4,
                    "true_class": CLASSES[i % 4],
                    "p_top": 0.3 + rng.normal(0, 0.02),
                    "p_random": 0.69 + rng.normal(0, 0.02),
                }
            )
    settings = {"resamples": 100, "confidence": 0.95, "stratified": True}
    blocks = pd.Series({f"{run}_{i}": f"b{i % 8}" for run, _ in runs for i in range(40)})
    comparison = {"ci_excludes_zero": True, "min_round_wins": 4}
    summary = ex.deletion_summary(pd.DataFrame(rows), pd.DataFrame(photo_rows), blocks, settings, comparison)
    assert (
        summary["verdict"].iloc[0] == "top-k matters more than random-k"
        and summary["round_wins"].iloc[0] == 5
    )
    same = pd.DataFrame(photo_rows).assign(p_top=lambda t: t["p_random"])
    flat = pd.DataFrame(rows).assign(p_true=lambda t: np.where(t["condition"] == "top", 0.69, t["p_true"]))
    assert (
        ex.deletion_summary(flat, same, blocks, settings, comparison)["verdict"].iloc[0]
        == "no clear difference"
    )


# ------------------------------------------------------------------ example selection


def predictions(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for round_ in (1, 2):
        for i in range(120):
            probs = rng.dirichlet(np.ones(4) * 0.4)
            true = CLASSES[i % 4]
            rows.append(
                {
                    "round": round_,
                    "photo_id": f"p{i % 100:03d}",
                    "true_class": true,
                    "pred_class": CLASSES[probs.argmax()],
                    **{f"prob_{c}": p for c, p in zip(CLASSES, probs)},
                }
            )
    return pd.DataFrame(rows)


def test_example_selection_is_deterministic_has_no_duplicates_and_follows_the_rule():
    table = predictions()
    first = ex.rank_examples(table, CLASSES, 3, 0)
    again = ex.rank_examples(table.sample(frac=1, random_state=1), CLASSES, 3, 0)  # row order must not matter
    pd.testing.assert_frame_equal(first.reset_index(drop=True), again.reset_index(drop=True))
    assert not first["photo_id"].duplicated().any()
    assert (first.groupby(["true_class", "category"]).size() <= 3).all()
    correct, wrong = first[first["category"] == "correct"], first[first["category"] == "wrong"]
    assert (correct["pred_class"] == correct["true_class"]).all()
    assert (wrong["pred_class"] != wrong["true_class"]).all()
    for name in CLASSES:  # the most confident ones come first
        confidence = correct[correct["true_class"] == name]["confidence"].tolist()
        assert confidence == sorted(confidence, reverse=True)
    assert not ex.rank_examples(table, CLASSES, 3, 1).equals(first)  # the random part follows the seed


def test_extras_complete_a_blind_fifty_item_sheet():
    table = predictions()
    rule = ex.rank_examples(table, CLASSES, 3, 0)
    sheet = ex.add_extras(rule, table, 14, 0)
    assert len(sheet) == len(rule) + 14 and not sheet["photo_id"].duplicated().any()
    assert sheet["item_id"].is_unique and set(sheet["item_id"]) == {
        f"H{i:02d}" for i in range(1, len(sheet) + 1)
    }
    assert (sheet[sheet["category"] == "extra"].shape[0]) == 14


# ------------------------------------------------------------------ background and heat maps


def test_tile_background_fraction_on_a_known_mask():
    leaf = np.zeros((100, 100), bool)
    leaf[:, 50:] = True  # right half is leaf
    boxes = np.array([[0, 0, 0, 0], [50, 0, 0, 1], [25, 0, 0, 2], [80, 80, 1, 1]], np.int32)
    fraction = ex.tile_background_fraction(leaf, boxes, 40)
    assert np.allclose(fraction, [1.0, 0.0, (25 * 40) / 1600, 1 - (20 * 20) / 1600])  # last box hangs outside


def test_background_share_is_a_share():
    weights = np.random.default_rng(0).dirichlet(np.ones(30))
    fraction = np.random.default_rng(1).uniform(size=30)
    result = ex.background_share(weights, fraction)
    assert 0 <= result["attention_share"] <= 1 and 0 <= result["tile_share"] <= 1
    assert ex.background_share(weights, np.ones(30)) == {"attention_share": 1.0, "tile_share": 1.0}
    assert ex.background_share(weights, np.zeros(30)) == {"attention_share": 0.0, "tile_share": 0.0}
    assert (
        ex.background_share(weights, np.full(30, 0.9))["tile_share"] == 0.0
    )  # "more than 90%", not "at least"


def test_paint_puts_the_heat_on_the_attended_tile_and_grid_check_catches_a_wrong_scale():
    from src import tiling

    boxes = tiling.boxes(448, 224, 224)
    heat = ex.paint(boxes, np.array([0.9, 0.1]), (224, 448), 224, 0.5)
    assert heat.shape == (112, 224) and heat.max() == 1.0
    assert heat[:, :112].mean() > heat[:, 112:].mean()
    ex.check_grid(np.zeros((224, 448, 3), np.uint8), boxes, 224)
    with pytest.raises(ValueError, match="tile grid"):
        ex.check_grid(np.zeros((224, 672, 3), np.uint8), boxes, 224)


# ------------------------------------------------------------------ test discipline


def test_test_photos_need_the_declaration_and_the_marker_never_restarts_under_another_plan(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ex, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(ex, "require_clean_code", lambda: "abc123")
    with pytest.raises(RuntimeError, match="plan"):
        ex.start_marker("test", "deletion")
    (tmp_path / "explain_plan.json").write_text(json.dumps({"test_runs": []}))
    ex.start_marker("test", "deletion")
    ex.start_marker("test", "deletion")  # resuming under the same plan is allowed
    (tmp_path / "explain_plan.json").write_text(json.dumps({"test_runs": ["changed"]}))
    with pytest.raises(ValueError, match="another plan"):
        ex.start_marker("test", "deletion")
    ex.start_marker("val", "deletion")  # validation photos need no marker
    assert not (tmp_path / "explain_deletion_val_started.json").exists()


def test_only_the_five_tested_runs_are_used_on_test_photos():
    assert ex.runs_for("test") == [f"stage7_session_attention_r{r}" for r in range(1, 6)]
    assert len(ex.runs_for("val")) == 15
