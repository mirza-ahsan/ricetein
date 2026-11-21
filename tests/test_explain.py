"""Stage 12 checks on synthetic bags and a tiny head: no GPU, no caches, no research data."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import explain as ex
from src import tiling
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


def bags_of(n=4):
    return tm.Bags(Store(), folds(n), 1, "test", CLASSES, 1, 0.0, n, 0)


def batch(n=4):
    tiles, mask, _, _ = next(iter(bags_of(n)))
    return tiles, mask


# ------------------------------------------------------------------ attention and deletion


def test_attention_sums_to_one_over_real_tiles_and_is_zero_on_padding():
    tiles, mask = batch()
    assert not mask.all()  # the bags differ in size, so there is padding
    _, weights = model()(tiles, mask)
    assert torch.allclose(weights.sum(1), torch.ones(len(weights)), atol=1e-5)
    assert (weights[~mask] == 0).all()


@pytest.mark.parametrize("k", ex.KS)
def test_top_k_removes_exactly_k_real_tiles_and_the_highest_weights(k):
    tiles, mask = batch()
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
    _, mask = batch()
    for seed in ex.RANDOM_SEEDS:
        kept = ex.random_k_mask(mask, k, torch.Generator().manual_seed(seed))
        assert ((mask.sum(1) - kept.sum(1)) == k).all()
        assert not (kept & ~mask).any()  # nothing switched on, and removed tiles were real ones


def test_random_deletion_is_repeatable_and_depends_on_the_seed():
    _, mask = batch()
    a = ex.random_k_mask(mask, 4, torch.Generator().manual_seed(1))
    b = ex.random_k_mask(mask, 4, torch.Generator().manual_seed(1))
    c = ex.random_k_mask(mask, 4, torch.Generator().manual_seed(2))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_cannot_remove_all_tiles():
    _, mask = batch()
    with pytest.raises(ValueError, match="cannot remove"):
        ex.top_k_mask(torch.rand(mask.shape), mask, int(mask.sum(1).min()))


@pytest.mark.parametrize("aggregator", ["attention", "mean", "max"])
def test_masking_tiles_equals_removing_them_from_the_bag(aggregator):
    tiles, mask = batch()
    head = model(aggregator)
    weights = head(tiles, mask)[1] if aggregator == "attention" else torch.rand(mask.shape)
    kept = ex.top_k_mask(weights, mask, 4)
    masked_logits = head(tiles, kept)[0]
    for i in range(len(tiles)):
        shortened = tiles[i : i + 1, kept[i]]
        short_logits = head(shortened, torch.ones(shortened.shape[:2], dtype=torch.bool))[0]
        assert torch.allclose(masked_logits[i], short_logits[0], atol=1e-5)


def test_deletion_table_has_every_condition_and_matches_the_saved_attention():
    bags = bags_of()
    tiles, mask, _, rows = next(iter(bags))
    head = model()
    with torch.no_grad():
        saved = head(tiles, mask)[1]
    stored = {pid: w[:n].numpy() for pid, w, n in zip(bags.photo_ids[rows], saved, mask.sum(1))}
    long = ex.delete_and_score(head, bags, torch.device("cpu"), stored)
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
        ex.delete_and_score(head, bags, torch.device("cpu"), stored)


def test_removing_many_tiles_changes_the_probabilities():
    bags = bags_of()
    long = ex.delete_and_score(model(), bags, torch.device("cpu"))
    full = long[long["condition"] == "none"].set_index("photo_id")["p_true"]
    top = long[(long["condition"] == "top") & (long["k"] == 16)].set_index("photo_id")["p_true"]
    assert not np.allclose(full.loc[top.index], top)


def synthetic_deletion(gap_by_block, wins, n_runs=5):
    """Scores and per-photo table with a chosen true-class-probability gap in each of 8 capture blocks."""
    rng = np.random.default_rng(0)
    rows, photo_rows = [], []
    for r in range(1, n_runs + 1):
        run = f"run{r}"
        good = r <= wins  # the gap is positive in this round
        base = {"run": run, "round": r, "k": 4, "accuracy": 0.5, "macro_f1": 0.5}
        rows.append({**base, "condition": "none", "k": 0, "seed": -1, "p_true": 0.7})
        rows.append({**base, "condition": "top", "seed": -1, "p_true": 0.3 if good else 0.6})
        rows.append({**base, "condition": "random", "seed": 0, "p_true": 0.5})
        for i in range(40):
            gap = gap_by_block[i % 8] + rng.normal(0, 0.05)
            photo_rows.append(
                {
                    "photo_id": f"{run}_{i}",
                    "run": run,
                    "round": r,
                    "k": 4,
                    "true_class": CLASSES[i % 4],
                    "p_full": 0.7,
                    "p_top": 0.5 - gap / 2,
                    "p_random": 0.5 + gap / 2,
                }
            )
    blocks = pd.Series({f"run{r}_{i}": f"b{i % 8}" for r in range(1, n_runs + 1) for i in range(40)})
    return pd.DataFrame(rows), pd.DataFrame(photo_rows), blocks


SETTINGS = {"resamples": 200, "confidence": 0.95, "stratified": True}
RULE = {"ci_excludes_zero": True, "min_round_wins": 4}


def verdict(gap_by_block, wins):
    scores, photos, blocks = synthetic_deletion(gap_by_block, wins)
    return ex.deletion_summary(scores, photos, blocks, SETTINGS, RULE).iloc[0]


def test_verdict_needs_both_ranges_and_four_round_wins():
    clear = verdict([0.3] * 8, 5)
    assert clear["verdict"] == "top-k matters more than random-k" and clear["round_wins"] == 5
    assert verdict([0.3] * 8, 3)["verdict"] == "no clear difference"  # ranges fine, too few round wins
    assert verdict([0.3] * 8, 4)["verdict"] == "top-k matters more than random-k"
    assert verdict([0.0] * 8, 5)["verdict"] == "no clear difference"  # no gap at all
    # Three blocks with a big gap, five with a small negative one: photos agree on the sign, blocks do not.
    mixed = verdict([0.3] * 3 + [-0.1] * 5, 5)
    assert mixed["gap_low"] > 0 and mixed["gap_block_low"] < 0
    assert mixed["verdict"] == "no clear difference"


def test_summary_refuses_incomplete_rounds():
    scores, photos, blocks = synthetic_deletion([0.3] * 8, 5, n_runs=3)
    with pytest.raises(ValueError, match="expected 5"):
        ex.deletion_summary(scores, photos, blocks, SETTINGS, RULE)


def test_gap_by_class_has_one_row_per_class():
    _, photos, _ = synthetic_deletion([0.3] * 8, 5)
    table = ex.gap_by_class(photos)
    assert list(table["true_class"]) == sorted(CLASSES) and np.allclose(table[4], 0.3, atol=0.05)


def test_stability_seeds_are_left_out_of_the_headline_rows():
    names = pd.DataFrame(
        {
            "run": [
                "stage7_session_attention_r1",
                "stage7_session_attention_s1_r1",
                "stage7_session_attention_s2_r5",
            ]
        }
    )
    assert list(ex.seed0(names)["run"]) == ["stage7_session_attention_r1"]


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
    assert (sheet["category"] == "extra").sum() == 14


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
    use_runs_dir(monkeypatch, tmp_path)
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


def test_markers_are_written_whole_and_never_replaced(tmp_path):
    path = tmp_path / "marker.json"
    ex.write_once(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1} and not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        ex.write_once(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1} and not list(tmp_path.glob("*.tmp"))


def test_example_list_survives_the_csv_round_trip_that_select_uses_to_check_itself(tmp_path):
    table = predictions()
    sheet = ex.add_extras(ex.rank_examples(table, CLASSES, 3, 0), table, 14, 0)  # extras have no confidence
    ex.write_csv(sheet, tmp_path / "examples.csv")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "examples.csv"), sheet, check_dtype=False, atol=1e-6)


# --------------------------------------------- Paddy Doctor extension (handbook §24, 2026-11-21)


def use_runs_dir(monkeypatch, path):
    """Point this module and the Stage 10 module (whose file names it reuses) at the runs folder ``path``."""
    for module in (ex, ex.pz):
        monkeypatch.setattr(module, "get_default_runs_dir", lambda: path)


def paddy_runs(root):
    """The fifteen runs `plan` hashes; the five seed-0 ones also have test and saved Stage 10 predictions."""
    for name in ex.runs_for("val"):
        run = root / name
        run.mkdir(parents=True)
        (run / "record.json").write_text(json.dumps({"name": name}))
        (run / "model.pt").write_bytes(name.encode())
    for r in ex.ROUNDS:
        run = root / ex.run_name(0, r)
        (run / "test_predictions.csv").write_text("x\n")
        predictions().query("round == 1").assign(round=r).to_csv(
            run / "paddy_predictions_native.csv", index=False
        )


def plan_with_paddy(tmp_path, monkeypatch):
    use_runs_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(ex, "require_clean_code", lambda: "abc123")
    monkeypatch.setattr(
        ex, "verify_paddy_cache", lambda config, name: {"spec_sha256": "f" * 64, "photos": 8086}
    )
    paddy_runs(tmp_path)
    ex.plan()
    return json.loads((tmp_path / "explain_plan.json").read_text())


def test_plan_declares_the_paddy_extension_with_the_same_five_models(tmp_path, monkeypatch):
    declaration = plan_with_paddy(tmp_path, monkeypatch)
    paddy = declaration["paddy"]
    assert paddy["zoom"] == "native" and paddy["cache"] == "paddy_e1_native" and paddy["photos"] == 8086
    assert paddy["cache_spec_sha256"] == "f" * 64 and "photo only" in paddy["bootstrap"]
    assert sorted(paddy["predictions_sha256"]) == ex.runs_for("paddy") == ex.runs_for("test")
    assert ex.names_for("paddy", declaration) == declaration["test_runs"] == ex.runs_for("test")
    assert (
        "paddy" in declaration["deletion"]["partitions"]
        and "paddy" in declaration["background"]["partitions"]
    )


def test_paddy_photos_need_a_plan_that_has_the_extension_and_their_own_run_once_markers(
    tmp_path, monkeypatch
):
    use_runs_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(ex, "require_clean_code", lambda: "abc123")
    with pytest.raises(RuntimeError, match="plan"):
        ex.start_marker("paddy", "deletion")
    plan = tmp_path / "explain_plan.json"
    plan.write_text(json.dumps({"test_runs": []}))  # a plan declared before the extension
    with pytest.raises(RuntimeError, match="Paddy extension"):
        ex.start_marker("paddy", "deletion")
    assert not list(tmp_path.glob("explain_*_started.json"))
    plan.write_text(json.dumps({"test_runs": [], "paddy": {}}))
    ex.start_marker("paddy", "deletion")
    ex.start_marker("paddy", "deletion")  # resuming under the same plan is allowed
    assert [f.name for f in tmp_path.glob("explain_*_started.json")] == [
        "explain_paddy_deletion_started.json"
    ]
    ex.start_marker("test", "deletion")  # the test marker is separate and keeps its old name
    assert (tmp_path / "explain_deletion_started.json").exists()
    plan.write_text(json.dumps({"test_runs": ["changed"], "paddy": {}}))
    with pytest.raises(ValueError, match="another plan"):
        ex.start_marker("paddy", "deletion")


def test_paddy_examples_follow_the_rule_on_the_saved_stage_10_predictions_only(tmp_path, monkeypatch):
    plan_with_paddy(tmp_path, monkeypatch)
    monkeypatch.setattr(ex, "OUT", tmp_path / "explain")
    monkeypatch.setattr(ex, "RATINGS", tmp_path / "ratings.csv")
    for name in ex.runs_for("paddy"):  # test photos must not be needed: Paddy reads its own predictions only
        (tmp_path / name / "test_predictions.csv").unlink()
        (tmp_path / name / "predictions.csv").unlink(missing_ok=True)
    ex.select("paddy")
    chosen = pd.read_csv(tmp_path / "explain" / "examples_paddy.csv")
    assert 0 < len(chosen) <= 36 and "item_id" not in chosen and not chosen["photo_id"].duplicated().any()
    assert not (tmp_path / "explain" / "blind").exists()
    expected = ex.rank_examples(ex.read_run_predictions("paddy"), CLASSES, ex.PER_CELL, ex.SELECT_SEED)
    pd.testing.assert_frame_equal(chosen, expected, check_dtype=False, atol=1e-6)
    ex.select("paddy")  # again: the same rule gives the same list
    saved = tmp_path / ex.run_name(0, 3) / "paddy_predictions_native.csv"
    saved.write_text(saved.read_text() + "\n")  # a predictions file changed after the declaration
    with pytest.raises(ValueError, match="changed after the declaration"):
        ex.select("paddy")


def test_paddy_select_is_refused_before_the_plan(tmp_path, monkeypatch):
    use_runs_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(ex, "OUT", tmp_path / "explain")
    with pytest.raises(RuntimeError, match="plan"):
        ex.select("paddy")
    assert not (tmp_path / "explain").exists()


def write_cache(folder, photo_ids, sizes, dim=3):
    folder.joinpath("v0").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for photo_id, n in zip(photo_ids, sizes):
        np.savez(
            folder / "v0" / f"{ex.photo_key(photo_id)}.npz",
            cls=rng.normal(size=(n, dim)).astype(np.float16),
            mean=rng.normal(size=(n, dim)).astype(np.float16),
        )


def paddy_bags(tmp_path, monkeypatch, ids=None, sizes=(20, 25, 18, 30, 22)):
    names = ["leaf_blast/a.jpg", "hispa/b.jpg", "healthy/c.jpg", "brown_spot/d.jpg", "hispa/e.jpg"]
    classes = ["leaf_blast", "hispa", "healthy", "brown_spot", "hispa"]
    write_cache(tmp_path / "cache", names, sizes)
    monkeypatch.setattr(ex, "cache_folder", lambda name: tmp_path / "cache")
    monkeypatch.setattr(
        ex.pz,
        "truth",
        lambda config: pd.DataFrame(
            {"photo_id": sorted(names), "class": [dict(zip(names, classes))[n] for n in sorted(names)]}
        ),
    )
    monkeypatch.setattr(ex.pz, "photo_ids", lambda config: sorted(names))
    config = {"paddy": {"caches": {"tiles": {"native": "x"}}}, "classes": CLASSES}
    return ex.PaddyBags(config, ["cls", "mean"], ids, batch=2), names


def test_paddy_bags_have_the_layout_of_the_primary_bags(tmp_path, monkeypatch):
    bags, names = paddy_bags(tmp_path, monkeypatch)
    batches = list(bags)
    assert [len(rows) for *_, rows in batches] == [2, 2, 1]
    seen = []
    for tiles, mask, labels, rows in batches:
        ids = bags.photo_ids[rows]
        seen += list(ids)
        assert tiles.shape[2] == 6 and tiles.dtype == torch.float32  # class token and patch mean joined
        for i, photo_id in enumerate(ids):
            with np.load(tmp_path / "cache" / "v0" / f"{ex.photo_key(photo_id)}.npz") as saved:
                n = len(saved["cls"])
                assert int(mask[i].sum()) == n and mask[i, :n].all() and not mask[i, n:].any()
                assert torch.equal(tiles[i, :n, :3], torch.from_numpy(saved["cls"].astype(np.float32)))
                assert (tiles[i, n:] == 0).all()
        truth = {n: c for n, c in zip(sorted(names), bags.labels.tolist())}
        assert [CLASSES[k] for k in labels.tolist()] == [CLASSES[truth[p]] for p in ids]
    assert seen == sorted(names)


def test_paddy_bags_can_be_limited_to_chosen_photos_and_attention_matches_the_head(tmp_path, monkeypatch):
    bags, names = paddy_bags(tmp_path, monkeypatch, ids=["hispa/b.jpg", "healthy/c.jpg"])
    head = MILModel(6, 8, 0.0, "attention", len(CLASSES)).eval()
    weights = ex.attention_of(head, bags, torch.device("cpu"))
    assert set(weights) == {"hispa/b.jpg", "healthy/c.jpg"}
    assert len(weights["hispa/b.jpg"]) == 25 and len(weights["healthy/c.jpg"]) == 18
    assert all(np.isclose(w.sum(), 1.0, atol=1e-5) for w in weights.values())
    long = ex.delete_and_score(head, bags, torch.device("cpu"))
    assert set(long["photo_id"]) == set(weights) and len(long) == 2 * (
        1 + len(ex.KS) * (1 + len(ex.RANDOM_SEEDS))
    )


def test_no_removal_probabilities_must_match_the_saved_stage_10_ones(tmp_path, monkeypatch):
    use_runs_dir(monkeypatch, tmp_path)
    run = tmp_path / "r"
    run.mkdir()
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1]])
    saved = pd.DataFrame(probs, columns=[f"prob_{c}" for c in CLASSES]).assign(photo_id=["a", "b"])
    saved.to_csv(run / "paddy_predictions_native.csv", index=False)
    long = pd.DataFrame({"photo_id": ["a", "b"], "condition": "none", "true": [0, 1], "p_true": [0.7, 0.6]})
    ex.check_against_stage10(long, "r", CLASSES)
    long["p_true"] = [0.7, 0.5]  # a wrong cache, zoom or checkpoint
    with pytest.raises(ValueError, match="differ from the saved Stage 10"):
        ex.check_against_stage10(long, "r", CLASSES)


def test_photo_only_verdict_averages_the_round_models_per_photo_and_needs_no_blocks():
    scores, photos, blocks = synthetic_deletion([0.3] * 8, 5)
    photos = photos.assign(
        photo_id=photos["photo_id"].str.split("_").str[1]
    )  # the same 40 photos in every round
    summary = ex.deletion_summary(scores, photos, None, SETTINGS, RULE).iloc[0]
    assert "gap_block_low" not in summary
    assert summary["verdict"] == "top-k matters more than random-k (photo range only)"
    assert summary["gap_low"] > 0 and abs(summary["gap_p_true"] - 0.3) < 0.02
    few = ex.deletion_summary(*synthetic_deletion([0.3] * 8, 3)[:2], None, SETTINGS, RULE).iloc[0]
    assert (
        few["verdict"] == "no clear difference (photo range only)"
    )  # a clear range still needs the round wins
    none = ex.deletion_summary(*synthetic_deletion([0.0] * 8, 5)[:2], None, SETTINGS, RULE).iloc[0]
    assert none["verdict"] == "no clear difference (photo range only)"


def test_photo_only_range_resamples_photos_not_rows(monkeypatch):
    """Each round-model sees every photo; rows of one photo must stay together and be averaged first."""
    scores, photos, _ = synthetic_deletion([0.3] * 8, 5)
    photos = photos.assign(photo_id=photos["photo_id"].str.split("_").str[1])
    offset = photos["round"].map({1: 0.2, 2: -0.2, 3: 0.1, 4: -0.1, 5: 0.0})  # cancels in the photo mean
    photos = photos.assign(p_random=photos["p_random"] + offset)
    sizes, real = [], ex.stats.resample_indices
    monkeypatch.setattr(ex.stats, "resample_indices", lambda *a: sizes.append(len(a[0])) or real(*a))
    summary = ex.deletion_summary(scores, photos, None, SETTINGS, RULE).iloc[0]
    assert sizes == [40] and abs(summary["gap_p_true"] - 0.3) < 0.02  # 40 photos, not 200 rows


class FakePhotos:
    """A 20-tile photo for every id, standing in for Photos("paddy")."""

    tile = 224

    def __init__(self, partition="paddy"):
        pass

    def rgb(self, photo_id):
        return np.random.default_rng(len(photo_id)).integers(0, 255, (1120, 896, 3), dtype=np.uint8)

    def boxes(self, photo_id):
        return tiling.boxes(896, 1120, 224)


def paddy_world(tmp_path, monkeypatch):
    """Five tiny trained round-models (and their ten stability twins), a 20-tile cache of five photos and the
    saved Stage 10 predictions, with every path and the Paddy config pointed at ``tmp_path``."""
    bags, names = paddy_bags(tmp_path, monkeypatch, sizes=(20,) * 5)
    fake = {
        "paddy": {
            "caches": {"tiles": {"native": "paddy_e1_native"}},
            "encoder": "E1",
            "split_file": "splits/paddy4.csv",
            "zoom_factors": {"native": 1.0},
        },
        "encoders": {"E1": {"dim": 3}},
        "classes": CLASSES,
        "seed": 0,
    }
    real = ex.load_config
    monkeypatch.setattr(ex, "load_config", lambda name: fake if name == ex.PADDY_CONFIG else real(name))
    use_runs_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(ex, "require_clean_code", lambda: "abc123")
    monkeypatch.setattr(ex, "verify_paddy_cache", lambda config, name: {"spec_sha256": "f" * 64, "photos": 5})
    monkeypatch.setattr(ex, "Photos", FakePhotos)
    for name, folder in (("OUT", "explain"), ("TABLES", "tables"), ("FIGURES", "figures")):
        monkeypatch.setattr(ex, name, tmp_path / folder)
    monkeypatch.setattr(ex, "REPORT", tmp_path / "explain_report.md")
    monkeypatch.setattr(ex, "RATINGS", tmp_path / "ratings.csv")
    for name in ex.runs_for("val"):
        r = int(name.rsplit("_r", 1)[1])
        run = tmp_path / name
        run.mkdir()
        torch.manual_seed(r)
        head = MILModel(6, 8, 0.0, "attention", len(CLASSES)).eval()
        torch.save(head.state_dict(), run / "model.pt")
        config = {"round": r, "params": {"descriptor": "cm", "width": 8, "dropout": 0.0}, "classes": CLASSES}
        (run / "record.json").write_text(
            json.dumps({"status": "completed", "config": {**config, "mil": {"aggregator": "attention"}}})
        )
        if name in ex.runs_for("test"):
            (run / "test_predictions.csv").write_text("x\n")
            rows = []
            for tiles, mask, _, idx in bags:
                probs = torch.softmax(head(tiles, mask)[0], 1).detach().numpy()
                for photo_id, p in zip(bags.photo_ids[idx], probs):
                    rows.append(
                        {
                            "round": r,
                            "partition": "paddy",
                            "photo_id": photo_id,
                            "true_class": truth_of(photo_id),
                            "pred_class": CLASSES[p.argmax()],
                            **{f"prob_{c}": v for c, v in zip(CLASSES, p)},
                        }
                    )
            pd.DataFrame(rows).to_csv(run / "paddy_predictions_native.csv", index=False)
    return bags, names


def truth_of(photo_id):
    return photo_id.split("/")[0]


def test_paddy_deletion_runs_end_to_end_resumes_and_feeds_the_report(tmp_path, monkeypatch):
    paddy_world(tmp_path, monkeypatch)
    ex.plan()
    ex.deletion("paddy")
    assert (tmp_path / "explain_paddy_deletion_started.json").exists()
    assert not (tmp_path / "explain_deletion_started.json").exists()  # the test marker is untouched
    assert len(list((tmp_path / "explain").glob("deletion_paddy_*.csv"))) == 5
    first = {f.name: f.read_bytes() for f in (tmp_path / "explain").glob("deletion_*paddy_*.csv")}
    ex.deletion("paddy")  # a resume under the same plan finds everything done
    assert first == {f.name: f.read_bytes() for f in (tmp_path / "explain").glob("deletion_*paddy_*.csv")}
    ex.report()
    text = (tmp_path / "explain_report.md").read_text()
    assert "## Deletion test, Paddy Doctor photos at native zoom" in text and "photos only" in text
    summary = pd.read_csv(tmp_path / "tables" / "explain_deletion_paddy.csv")
    assert list(summary["k"]) == list(ex.KS) and "gap_block_low" not in summary
    assert summary["verdict"].str.endswith("(photo range only)").all()
    assert (
        "## Deletion test, test photos" in text and "Not run yet" in text
    )  # the primary sections are untouched


def test_paddy_deletion_refuses_a_model_that_changed_after_the_plan(tmp_path, monkeypatch):
    paddy_world(tmp_path, monkeypatch)
    ex.plan()
    (tmp_path / ex.run_name(0, 4) / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed after the declaration"):
        ex.deletion("paddy")
    assert not list((tmp_path / "explain").glob("deletion_paddy_*r4.csv"))


def test_paddy_background_and_heat_maps_run_end_to_end(tmp_path, monkeypatch):
    paddy_world(tmp_path, monkeypatch)
    ex.plan()
    ex.background("paddy")
    table = pd.read_csv(tmp_path / "explain" / "background_paddy.csv")
    assert (
        len(table) == 25
        and table["attention_share"].between(0, 1).all()
        and set(table["round"]) == set(ex.ROUNDS)
    )
    assert (tmp_path / "explain_paddy_background_started.json").exists()
    ex.select("paddy")
    ex.heatmaps("paddy")
    drawn = list((tmp_path / "explain" / "heatmaps" / "paddy").glob("*.png"))
    examples = pd.read_csv(tmp_path / "explain" / "examples_paddy.csv")
    assert len(drawn) == len(examples) and not (tmp_path / "explain" / "blind").exists()
    ex.report()
    assert (
        "## Attention on background tiles, Paddy Doctor photos at native zoom"
        in (tmp_path / "explain_report.md").read_text()
    )


def test_paddy_deletion_stops_when_the_saved_predictions_changed_and_writes_nothing(tmp_path, monkeypatch):
    paddy_world(tmp_path, monkeypatch)
    ex.plan()
    saved = tmp_path / ex.run_name(0, 2) / "paddy_predictions_native.csv"
    frame = pd.read_csv(saved)
    columns = [f"prob_{c}" for c in CLASSES]
    frame[columns] = frame[columns].iloc[::-1].to_numpy()
    frame.to_csv(saved, index=False)
    with pytest.raises(ValueError, match="changed after the declaration"):
        ex.deletion("paddy")
    assert not list((tmp_path / "explain").glob("deletion_*paddy_*.csv"))
    with pytest.raises(ValueError, match="changed after the declaration"):
        ex.background("paddy")
    with pytest.raises(ValueError, match="changed after the declaration"):
        ex.select("paddy")


def test_the_deletion_check_against_stage_10_is_called(tmp_path, monkeypatch):
    paddy_world(tmp_path, monkeypatch)
    ex.plan()
    called = []
    real = ex.check_against_stage10
    monkeypatch.setattr(ex, "check_against_stage10", lambda *a: called.append(a[1]) or real(*a))
    ex.deletion("paddy")
    assert called == ex.runs_for("paddy")


# ------------------------------------------------------------------ blind ratings


def sheet(n=10):
    return pd.DataFrame(
        {"item_id": [f"H{i:02d}" for i in range(n)], "category": (["correct", "wrong"] * n)[:n]}
    )


def ratings_of(answers):
    rows = []
    for rater, given in answers.items():
        for i, answer in enumerate(given):
            rows.append(
                {"item_id": f"H{i:02d}", "rater": rater, "rater_kind": "model", "rating": answer, "note": ""}
            )
    return pd.DataFrame(rows)


def test_fleiss_kappa_on_known_cases():
    assert ex.fleiss_kappa(np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 0, 0]])) == pytest.approx(1.0)
    # Fleiss et al.'s classic example (10 items, 14 raters, 5 categories): kappa = 0.210
    table = np.array(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ]
    )
    assert ex.fleiss_kappa(table) == pytest.approx(0.210, abs=0.001)
    with pytest.raises(ValueError, match="same number"):
        ex.fleiss_kappa(np.array([[2, 0, 0], [1, 0, 0]]))


def test_rating_summary_gives_shares_agreement_and_checks_the_file():
    a = ["yes"] * 5 + ["partly"] * 3 + ["no"] * 2
    b = ["yes"] * 5 + ["partly"] * 2 + ["no"] * 3
    c = ["yes"] * 4 + ["no"] * 1 + ["partly"] * 3 + ["no"] * 2
    out = ex.rating_summary(ratings_of({"r1": a, "r2": b, "r3": c}), sheet())
    assert out["shares"].loc["r1"].tolist() == [0.5, 0.3, 0.2] and out["items"] == 10
    assert out["shares"].loc["all ratings"].sum() == pytest.approx(1.0)
    assert out["all_agree"] == pytest.approx(0.8)  # items 4 and 7 differ; the other eight agree
    assert -1 <= out["kappa"] <= 1 and set(out["by_category"].index) == {"correct", "wrong"}
    bad = ratings_of({"r1": a, "r2": b})
    with pytest.raises(ValueError, match="one of"):
        ex.rating_summary(bad.assign(rating="maybe"), sheet())
    with pytest.raises(ValueError, match="rater id"):
        ex.rating_summary(bad.assign(rater=[None] + list(bad["rater"][1:])), sheet())
    with pytest.raises(ValueError, match="twice"):
        ex.rating_summary(pd.concat([bad, bad.head(1)]), sheet())
    with pytest.raises(ValueError, match="every blind item"):
        ex.rating_summary(bad.iloc[:-1], sheet())
    with pytest.raises(ValueError, match="every blind item"):
        ex.rating_summary(bad.assign(item_id=bad["item_id"].str.replace("H", "Z")), sheet())


def test_report_reads_the_ratings_file(tmp_path, monkeypatch):
    for name, path in (("OUT", "explain"), ("TABLES", "tables"), ("RATINGS", "ratings.csv")):
        monkeypatch.setattr(ex, name, tmp_path / path)
    (tmp_path / "explain").mkdir()
    sheet().to_csv(tmp_path / "explain" / "examples_val.csv", index=False)
    assert ex.ratings_section()[2] == "No ratings yet."
    pd.DataFrame(columns=ex.RATING_COLUMNS).to_csv(ex.RATINGS, index=False)
    assert ex.ratings_section()[2] == "No ratings yet."  # header only
    answers = {"r1": ["yes", "no"] * 5, "r2": ["yes", "partly"] * 5}
    ratings_of(answers).to_csv(ex.RATINGS, index=False)
    text = "\n".join(ex.ratings_section())
    assert "Fleiss' kappa" in text and "`r1` (model)" in text
    assert (tmp_path / "tables" / "explain_ratings.csv").exists()
