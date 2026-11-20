"""Checks for Stage 9 (src/partition_splits.py, src/partition_order.py): the three flawed split files differ from
the correct split and from each other, the leak counter measures each mistake (B leaks the source photo, C ignores
groups, D scores on training photos, A leaks nothing), no condition touches the locked splits, the training loop
equals ``train_mil.fit`` for the correct protocol, the options of the declared protocol work, a frozen score plan
and a started marker are never bypassed, and the inflation statistics are right. CPU only, tiny synthetic data."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import augment, metrics
from src import partition_order as po
from src import partition_splits as ps
from src import train_mil as tm
from src.config import REPO_ROOT, load_config
from src.split import make_folds, round_column

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
N, VERSIONS = 200, 3


@pytest.fixture(scope="module")
def config():
    c = load_config("configs/partition_order.yaml")
    c["partition_order"]["versions"] = VERSIONS
    return c


@pytest.fixture(scope="module")
def world(config):
    """200 photos in 20 capture blocks of 10 (5 s apart, blocks 10,000 s apart) and 100 near-duplicate pairs."""
    ids = [f"p{i:03d}" for i in range(N)]
    photos = pd.DataFrame({"photo_id": ids, "class": [CLASSES[i % 4] for i in range(N)]})
    block, pair = [f"b{i // 10:02d}" for i in range(N)], [f"g{i // 2:03d}" for i in range(N)]
    session = make_folds(photos.assign(group_id=block), 5, 0.125, 2026).drop(columns="fold")
    locked = make_folds(photos.assign(group_id=pair), 5, 0.125, 2026).drop(columns="fold")
    keys = pd.DataFrame({"group": pair, "block": block, "camera": "cam",
                         "seconds": [(i // 10) * 10_000 + (i % 10) * 5 for i in range(N)]}, index=ids)
    keys.index.name = "photo_id"
    return SimpleNamespace(photos=photos, session=session, locked=locked, keys=keys,
                           tables=ps.condition_tables(session, config))


# ------------------------------------------------------------------ the split files and the leak counter

def test_b_and_c_differ_from_a_and_from_each_other(world):
    b, c, d = (world.tables[k] for k in "BCD")
    assert len(b) == N * VERSIONS and len(c) == N and len(d) == N
    assert b["sample_id"].is_unique and b.groupby("photo_id")["version"].apply(list).map(lambda v: v == [0, 1, 2]).all()
    for r in range(1, 6):
        col = round_column(r)
        assert not np.array_equal(c[col], world.session[col])           # C differs from A
        assert not np.array_equal(b[b["version"] == 0][col], c[col])     # B differs from C
        assert not np.array_equal(b[b["version"] == 0][col], world.session[col])
    assert not b.drop(columns="sample_id").equals(c.drop(columns="sample_id"))


def test_every_condition_tests_each_unit_exactly_once(world):
    for k in "BC":
        roles = world.tables[k][[round_column(r) for r in range(1, 6)]]
        assert ((roles == "test").sum(axis=1) == 1).all()
        assert set(roles.to_numpy().ravel()) == {"train", "val", "test"}


def test_b_puts_the_same_source_photo_on_both_sides_and_the_counter_sees_it(world):
    b = world.tables["B"]
    in_both = b.groupby("photo_id")[round_column(1)].agg(lambda roles: {"train", "test"} <= set(roles))
    assert in_both.mean() > 0.3        # with 3 copies a photo often has none in test; the 8-copy files are checked below
    assert ps.leak_counter(b, 1, world.keys)["source_photo"] > 0.9
    clean = ps.leak_counter(b, 1, world.keys, (b["version"] == 0).to_numpy())
    assert clean["scored_units"] == (world.tables["B"].query("version == 0")[round_column(1)] == "test").sum()


def test_the_correct_splits_leak_nothing_and_the_locked_split_leaks_only_sessions(world):
    for r in range(1, 6):
        a = ps.leak_counter(world.session, r, world.keys)
        assert a["source_photo"] == a["near_duplicate_group"] == a["capture_block"] == 0
        assert a["same_camera_within_10s"] == 0
        locked = ps.leak_counter(world.locked, r, world.keys)
        assert locked["source_photo"] == locked["near_duplicate_group"] == 0
    assert np.mean([ps.leak_counter(world.locked, r, world.keys)["capture_block"] for r in range(1, 6)]) > 0.3


def test_c_ignores_groups_and_blocks(world):
    c = world.tables["C"].merge(world.keys, left_on="photo_id", right_index=True)
    roles = c.groupby("group")[round_column(1)].agg(set)
    assert (roles.map(lambda s: len(s & {"train", "test"}) == 2)).any()      # a pair split over both sides
    shares = ps.leak_counter(world.tables["C"], 1, world.keys)
    assert shares["source_photo"] == 0 and shares["near_duplicate_group"] > 0 and shares["capture_block"] > 0.9


def test_d_scores_on_its_training_photos_and_no_standard_loader_accepts_the_file(world):
    d = world.tables["D"]
    assert (d[[round_column(r) for r in range(1, 6)]] == ps.BOTH).all().all()
    assert set(ps.leak_counter(d, 1, world.keys).values()) == {1.0, float(N)}
    assert (po.with_role(d, 1, "test")[round_column(1)] == "test").all()
    with pytest.raises(ValueError, match="no train photos"):
        tm.Bags(Store(), d, 1, "train", CLASSES, 1, 0.0, 4, 0)


def test_split_files_are_written_once_apart_from_the_locked_splits(world, tmp_path):
    paths = ps.lock_tables(world.tables, tmp_path / "partition_order")
    assert sorted(p.name for p in paths.values()) == sorted(ps.FILENAMES.values())
    assert sorted(p.name for p in (tmp_path / "partition_order").iterdir()) == sorted(ps.FILENAMES.values())
    ps.lock_tables(world.tables, tmp_path / "partition_order")                     # the same content: fine
    with pytest.raises(ValueError, match="never regenerated"):
        ps.lock_tables({**world.tables, "C": world.tables["C"].iloc[::-1]}, tmp_path / "partition_order")
    assert not {"folds.csv"} & set(ps.FILENAMES.values())
    assert load_config("configs/partition_order.yaml")["partition_order"]["split_dir"] == "splits/partition_order"


def test_the_committed_split_files_are_a_fresh_build_of_the_declared_settings():
    config = load_config("configs/partition_order.yaml")
    session = pd.read_csv(REPO_ROOT / config["split_files"]["folds"], dtype=str)
    folder = REPO_ROOT / config["partition_order"]["split_dir"]
    for condition, table in ps.condition_tables(session, config).items():
        assert (folder / ps.FILENAMES[condition]).read_text() == table.to_csv(index=False)
    b = ps.read_condition("B")
    assert len(b) == 8 * len(session) and b["version"].max() == 7
    both = b.groupby("photo_id")["round_1"].agg(lambda roles: {"train", "test"} <= set(roles))
    assert both.mean() > 0.7                                     # the mistake B imitates is really in the file
    assert config["split_files"]["folds"] == "splits/session_blocked/folds.csv"


# ------------------------------------------------------------------ training loops

class Store:
    """Deterministic descriptors for any (photo, version); records what was read."""
    dim = 6

    def __init__(self):
        self.calls = []

    def bag(self, photo_id, version):
        self.calls.append((photo_id, version))
        rng = np.random.default_rng([int(photo_id[1:]), version])
        return rng.normal(size=(2 + int(photo_id[1:]) % 3, self.dim)).astype(np.float16)


def mil_config(max_epochs=3):
    return {"mil": {"versions": VERSIONS, "batch": 8, "aggregator": "attention", "class_weighting": True,
                    "adam_eps": 1e-8, "max_epochs": max_epochs, "patience": 5, "device": "cpu"},
            "classes": CLASSES, "determinism": "strict", "loss": {"label_smoothing": 0.1},
            "params": {"width": 8, "dropout": 0.1, "tile_dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-3}}


def test_the_mil_loop_equals_train_mil_fit_for_the_correct_protocol(world):
    config = mil_config()
    expected = tm.fit(Store(), world.tables["C"], 1, config["params"], config, 0)
    actual = po.fit_mil(Store(), world.tables["C"], 1, "C", config, 0)
    assert actual["epoch"] == expected["epoch"] and actual["f1"] == expected["f1"]
    np.testing.assert_array_equal(actual["probs"], expected["probs"])
    np.testing.assert_array_equal(actual["rows"], expected["rows"])
    for key, value in expected["state"].items():
        torch.testing.assert_close(actual["state"][key], value)


def test_d_trains_on_every_photo_for_a_fixed_number_of_epochs_without_validation(world):
    store, d = Store(), world.tables["D"]
    best = po.fit_mil(store, d, 1, "D", mil_config(max_epochs=5), 3, fixed_epochs=2)
    assert best["epoch"] == 2 and best["f1"] is None and "rows" not in best
    assert {p for p, _ in store.calls} == set(d["photo_id"])                 # every photo trained on
    assert {v for _, v in store.calls} <= set(range(VERSIONS))


def test_b_draws_as_many_copies_per_epoch_as_asked_and_reads_the_fixed_copy_of_each(world):
    store, b = Store(), world.tables["B"]
    config = mil_config(max_epochs=1)
    train_pairs = set(map(tuple, b[b["round_1"] == "train"][["photo_id", "version"]].to_numpy()))
    po.fit_mil(store, b, 1, "B", config, 0, epoch_size=24)
    trained = store.calls[:24]                                               # the training pass comes first
    assert len(set(trained)) == 24 and set(trained) <= train_pairs
    val_pairs = set(map(tuple, b[b["round_1"] == "val"][["photo_id", "version"]].to_numpy()))
    assert set(store.calls[24:]) == val_pairs
    store.calls.clear()
    bags = po.mil_bags(store, b, 1, "test", "B", config, config["params"], 0)
    rows = np.concatenate([batch[3] for batch in bags])
    expected = b.iloc[rows][["photo_id", "version"]].to_numpy()
    assert store.calls == list(map(tuple, expected)) and (b["round_1"].iloc[rows] == "test").all()


def tiny_net(config, size):
    return nn.Sequential(nn.Conv2d(3, 4, 3, stride=16), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 4))


def whole_config():
    return {"wholeimage": {"batch_size": {224: 4}, "workers": 0, "weight_decay": 0.05, "warmup_epochs": 1,
                           "max_epochs": 3, "patience": 2, "determinism": "fast", "device": "cpu"},
            "classes": CLASSES, "lr": 1e-3, "loss": {"label_smoothing": 0.1},
            "normalisation": {"mean": [0.5] * 3, "std": [0.25] * 3}}


@pytest.fixture(scope="module")
def pictures(world):
    rng = np.random.default_rng(0)
    ids = world.photos["photo_id"].tolist()[:40]
    photos = rng.integers(0, 255, (40, 40, 40, 3), dtype=np.uint8)
    copies = rng.integers(0, 255, (VERSIONS, 40, 224, 224, 3), dtype=np.uint8)
    units = {k: t[t["photo_id"].isin(ids)].reset_index(drop=True) for k, t in world.tables.items()}
    return SimpleNamespace(images=SimpleNamespace(photos=photos, ids=ids, copies=copies), units=units)


@pytest.mark.parametrize("condition,options", [("C", {}), ("B", {"epoch_size": 8}), ("D", {"fixed_epochs": 1})])
def test_the_whole_image_loop_runs_each_condition(pictures, condition, options):
    best = po.fit_whole(whole_config(), pictures.units[condition], 1, condition, pictures.images, 0,
                        make_model=tiny_net, **options)
    assert best["epoch"] >= 1 and set(best["state"]) == set(tiny_net(None, 224).state_dict())
    assert ("rows" in best) == (condition != "D")
    if condition == "B":                                   # step-matched: 8 copies a pass at batch 4
        data = po.photo_data(whole_config(), pictures.units["B"], 1, "train", "B", pictures.images)
        assert len(po.loader(data, whole_config(), True, 8)) == 2 and len(po.loader(data, whole_config(), True, None)) > 2


def test_b_copies_are_fixed_and_the_clean_copy_is_the_main_protocols_evaluation_view(tmp_path):
    rng = np.random.default_rng(0)
    photos, ids = rng.integers(0, 255, (3, 40, 40, 3), dtype=np.uint8), ["a", "b", "c"]
    config = {"seed": 0, "partition_order": {"versions": VERSIONS}}
    copies = po.build_copies(photos, ids, tmp_path, config)
    clean = augment.evaluation(224)(torch.from_numpy(photos[1]).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(copies[0, 1], clean)
    assert not np.array_equal(copies[1, 1], copies[2, 1]) and not np.array_equal(copies[1, 1], clean)
    np.testing.assert_array_equal(po.build_copies(photos, ids, tmp_path, config), copies)   # built once, reused
    with pytest.raises(ValueError, match="frozen"):
        po.build_copies(photos[:2], ids[:2], tmp_path, config)
    dataset = po.CopyDataset(pd.DataFrame({"photo_id": ["b", "b"], "class": CLASSES[:2], "version": [0, 2],
                                           "round_1": ["train", "test"]}), copies, ids, 1, "test", CLASSES)
    image, label, row = dataset[0]
    assert row == 1 and label == 1 and torch.equal(image.permute(1, 2, 0), torch.from_numpy(copies[2, 1]))


# ------------------------------------------------------------------ the runs, the declaration and the plan

def test_runs_reuse_a_settings_with_the_conditions_own_split_and_a_fenced_name(config):
    record = {"config": {"lr": 0.001, "seed": 0, "split_files": {"folds": "splits/session_blocked/folds.csv"},
                         "wholeimage": {"batch_size": {"224": 32}, "models": {"m": {"size_kwargs": {"224": {"a": 1}}}}}}}
    names = [n for n, *_ in po.all_runs(config)]
    assert len(names) == len(set(names)) == 30 and all(n.startswith("stage9_") for n in names)
    d = po.run_config(config, record, "efficientnetv2s", "D", 3)
    assert d["lr"] == 0.001 and d["run_seed"] == 2 and d["wholeimage"]["batch_size"] == {224: 32}
    assert d["split_files"]["folds"] == "splits/partition_order/condition_D_train_score_all.csv"
    assert po.run_config(config, record, "attention", "B", 3)["run_seed"] == 0


@pytest.fixture
def runs_dir(tmp_path, monkeypatch, config):
    monkeypatch.setattr(po, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(po, "require_clean_code", lambda: "abc1234")
    for name, _, condition, _ in po.all_runs(config):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "record.json").write_text(json.dumps({"status": "completed", "config": {}}))
        for artifact in po.ARTIFACTS[condition]:
            (folder / artifact).write_text(name + artifact)
    monkeypatch.setattr(po, "completed_run", lambda name, artifacts: (tmp_path / name / "record.json").exists())
    return tmp_path


def stub_scores(monkeypatch, fail_on=None):
    def score(name, model, condition, r):
        if name == fail_on:
            raise RuntimeError("crash")
        return pd.DataFrame({"true_class": CLASSES, "pred_class": CLASSES})
    monkeypatch.setattr(po, "score_run", score)


def test_every_run_is_scored_once_after_the_plan_is_frozen(runs_dir, config, monkeypatch):
    stub_scores(monkeypatch)
    po.score_all(config)
    first = "stage9_efficientnetv2s_B_r1"
    assert (runs_dir / first / "score_started.json").exists() and (runs_dir / first / "score_metrics.json").exists()
    po.score_all(config)                                                   # nothing is scored twice
    (runs_dir / first / "score_metrics.json").unlink()                     # a scoring that did not finish
    with pytest.raises(RuntimeError, match="started earlier"):
        po.score_all(config)
    monkeypatch.setattr(po, "require_declared", lambda config: None)
    with pytest.raises(RuntimeError, match="frozen"):
        po.fit_all(config)


def test_a_checkpoint_that_changes_after_the_plan_is_never_scored(runs_dir, config, monkeypatch):
    stub_scores(monkeypatch, fail_on="stage9_efficientnetv2s_B_r1")
    with pytest.raises(RuntimeError, match="crash"):
        po.score_all(config)
    (runs_dir / "stage9_efficientnetv2s_B_r1" / "score_started.json").unlink()
    (runs_dir / "stage9_efficientnetv2s_B_r1" / "model.pt").write_text("changed")
    stub_scores(monkeypatch)
    with pytest.raises(ValueError, match="changed after the evaluation plan"):
        po.score_all(config)


def test_scoring_waits_until_every_run_is_trained(runs_dir, config, monkeypatch):
    (runs_dir / "stage9_attention_D_r5" / "record.json").unlink()
    stub_scores(monkeypatch)
    with pytest.raises(RuntimeError, match="not trained yet"):
        po.score_all(config)
    assert not (runs_dir / po.PLAN).exists()


def test_stage9_runs_stay_reusable_while_only_their_own_code_is_unchanged():
    from src import records
    paths = records.code_paths("stage9_attention_B_r1")
    assert "src/partition_order.py" in paths and "src/partition_splits.py" in paths and "src/train_mil.py" in paths
    assert "src/records.py" not in paths and "src/explain.py" not in paths and "configs" not in paths


def test_a_declaration_is_frozen_and_checked_before_training(tmp_path, monkeypatch, config):
    monkeypatch.setattr(po, "get_default_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(po, "protocol", lambda c: {"settings": 1})
    with pytest.raises(RuntimeError, match="missing"):
        po.require_declared(config)
    po.declare(config)
    po.require_declared(config)
    monkeypatch.setattr(po, "protocol", lambda c: {"settings": 2})
    with pytest.raises(RuntimeError, match="differs"):
        po.require_declared(config)
    with pytest.raises(ValueError, match="frozen"):
        po.declare(config)


# ------------------------------------------------------------------ the inflation statistics

def random_predictions(n_photos=60, rows_per_photo=1, accuracy=0.7, seed=0):
    rng = np.random.default_rng(seed)
    true = np.tile(np.array(CLASSES)[np.arange(n_photos) % 4], rows_per_photo)
    wrong = rng.random(len(true)) > accuracy
    pred = np.where(wrong, rng.choice(CLASSES, len(true)), true)
    return pd.DataFrame({"photo_id": np.tile([f"p{i:03d}" for i in range(n_photos)], rows_per_photo),
                         "true_class": true, "pred_class": pred,
                         "round": np.repeat(np.arange(1, rows_per_photo + 1), n_photos)})


def test_scores_from_confusion_equal_the_main_scoring_code():
    t = random_predictions(80, 3, 0.6)
    photos = pd.Index(sorted(t["photo_id"].unique()))
    f1, balanced = po.scores_from_confusion(po.confusion_by_photo(t, photos, CLASSES).sum(0), 4)
    expected = metrics.scores(t["true_class"], t["pred_class"], CLASSES)
    assert f1 == pytest.approx(expected["macro_f1"]) and balanced == pytest.approx(expected["balanced_accuracy"])


def test_inflation_is_the_difference_to_a_and_its_range_is_paired_over_photos(config):
    a, b = random_predictions(120, 1, 0.55, seed=1), random_predictions(120, 4, 0.95, seed=2)
    photos = pd.Index(sorted(a["photo_id"]))
    counts = {"A": po.confusion_by_photo(a, photos, CLASSES), "B": po.confusion_by_photo(b, photos, CLASSES),
              "A2": po.confusion_by_photo(a, photos, CLASSES)}
    photo_class = np.arange(120) % 4
    ranges = po.inflation_ranges(counts, photo_class, {"resamples": 300, "confidence": 0.95}, 0, 4)
    assert ranges["A"]["macro_f1"]["difference"] == 0 and ranges["A2"]["macro_f1"]["difference_low"] == 0
    r = ranges["B"]["macro_f1"]
    assert r["difference"] == pytest.approx(r["value"] - ranges["A"]["macro_f1"]["value"]) and r["difference_low"] > 0
    assert r["low"] <= r["value"] <= r["high"]


def test_a_difference_is_called_clear_only_when_the_range_and_the_rounds_agree():
    assert po.reading(0.02, 0.1, 5, 0, 4) == "clearly above A"
    assert po.reading(0.02, 0.1, 3, 2, 4) == "no clear difference"       # the range excludes 0, too few rounds
    assert po.reading(-0.01, 0.1, 5, 0, 4) == "no clear difference"
    assert po.reading(-0.1, -0.02, 0, 5, 4) == "clearly below A"


def test_the_inflation_table_and_the_report_carry_the_leak_counter(config, world):
    tables = {"A": random_predictions(60, 1, 0.5, 1), "L": random_predictions(60, 1, 0.6, 2),
              "B": random_predictions(60, 3, 0.95, 3), "B_clean": random_predictions(60, 1, 0.95, 4),
              "C": random_predictions(60, 1, 0.8, 5), "D": random_predictions(60, 5, 0.99, 6)}
    tables["B"]["version"] = np.repeat(np.arange(3), 60)
    tables["B_clean"]["round"] = np.tile(np.arange(1, 6), 12)
    for t in tables.values():
        t["round"] = t["round"].clip(1, 5)
    leaks = {k: {"source_photo": 1.0 if k in ("B", "B_clean", "D") else 0.0, "near_duplicate_group": 0.1,
                 "capture_block": 0.9, "same_camera_within_10s": 0.5} for k in tables}
    config = {**config, "evaluation": {**config["evaluation"], "bootstrap": {"resamples": 100, "confidence": 0.95}}}
    table = po.inflation_table(config, tables, leaks, "attention")
    assert list(table["condition"]) == list(tables) and table.loc[0, "macro_f1_reading"] == "reference"
    assert table.loc[table["condition"] == "B", "leak_source_photo"].item() == 1.0
    assert table.loc[table["condition"] == "A", "leak_source_photo"].item() == 0.0
    text = po.report_text(pd.concat([table, table.assign(model="efficientnetv2s")], ignore_index=True))
    assert "same source photo" in text and "Xu et al., 2024" in text and "inflation vs A" in text
