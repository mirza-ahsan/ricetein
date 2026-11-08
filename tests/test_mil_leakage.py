"""Behavioral checks: role isolation, clean evaluation, padding and deterministic CPU training."""

import numpy as np
import pandas as pd
import pytest
import torch

from src import train_mil as tm
from src.mil_models import MILModel

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]


class Store:
    dim = 4

    def __init__(self):
        self.calls = []

    def bag(self, photo_id, version):
        self.calls.append((photo_id, version))
        return np.full((1 + int(photo_id) % 3, 4), int(photo_id) + 1000 * version, np.float16)


def folds():
    return pd.DataFrame(
        {
            "photo_id": [str(k) for k in range(12)],
            "class": CLASSES * 3,
            "round_1": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
        }
    )


@pytest.mark.parametrize("role", ["val", "test"])
def test_evaluation_only_reads_its_clean_photos_in_repeatable_order(role):
    store = Store()
    bags = tm.Bags(store, folds(), 1, role, CLASSES, 1, 0, 3, 0)
    first = list(bags)
    second = list(bags)
    expected = folds().query("round_1 == @role").photo_id.tolist()
    assert store.calls == [(p, 0) for p in expected] * 2
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second))
    assert np.concatenate([b[3] for b in first]).tolist() == [int(p) for p in expected]
    for versions, dropout in ((8, 0), (1, 0.3)):
        with pytest.raises(ValueError, match="version 0"):
            tm.Bags(store, folds(), 1, role, CLASSES, versions, dropout, 3, 0)


def test_training_versions_and_dropout_never_cross_roles_or_empty_bags():
    store = Store()
    bags = tm.Bags(store, folds(), 1, "train", CLASSES, 8, 1, 4, 0)
    for _ in range(10):
        for tiles, mask, _, _ in bags:
            assert mask.sum(1).tolist() == [1, 1, 1, 1]
            assert torch.isfinite(tiles).all()
    assert {p for p, _ in store.calls} == {"0", "1", "2", "3"}
    assert {v for _, v in store.calls} - {0}


@pytest.mark.parametrize("aggregator", ["mean", "max", "attention"])
def test_masked_padding_cannot_change_diagnosis(aggregator):
    torch.manual_seed(1)
    model = MILModel(4, 8, 0.5, aggregator, 4).eval()
    tiles = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    padded = torch.cat([tiles, torch.full((2, 5, 4), 1e6)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(2, 5, dtype=torch.bool)], dim=1)
    a, _ = model(tiles, mask)
    b, weights = model(padded, padded_mask)
    torch.testing.assert_close(a, b)
    if weights is not None:
        assert torch.equal(weights[:, 3:], torch.zeros(2, 5))
        torch.testing.assert_close(weights.sum(1), torch.ones(2))


def test_small_cpu_fit_reproduces_validation_predictions_without_reading_test():
    config = {
        "mil": {
            "versions": 2,
            "batch": 4,
            "aggregator": "attention",
            "class_weighting": True,
            "adam_eps": 1e-8,
            "max_epochs": 2,
            "patience": 1,
            "device": "cpu",
        },
        "classes": CLASSES,
        "determinism": "strict",
        "loss": {"label_smoothing": 0.1},
    }
    params = {"width": 8, "dropout": 0.1, "tile_dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3}
    one, two = Store(), Store()
    a = tm.fit(one, folds(), 1, params, config, 0)
    b = tm.fit(two, folds(), 1, params, config, 0)
    np.testing.assert_array_equal(a["probs"], b["probs"])
    assert a["rows"].tolist() == [4, 5, 6, 7]
    assert all(int(p) < 8 for p, _ in one.calls)


def test_packing_keeps_previous_cache_intact_when_new_source_fails(tmp_path):
    source, target = tmp_path / "source", tmp_path / "packed"
    (source / "v0").mkdir(parents=True)
    target.mkdir()
    np.savez(source / "v0/a.npz", cls=np.ones((3, 4), np.float16))
    array, index = tm.pack(source, target, "cls", ["a.jpg"], 1, "first")
    assert index["a.jpg|0"] == (0, 3)
    expected = array.copy()
    with pytest.raises(FileNotFoundError):
        tm.pack(source, target, "cls", ["missing.jpg"], 1, "new")
    again, _ = tm.pack(source, target, "cls", ["a.jpg"], 1, "first")
    np.testing.assert_array_equal(expected, again)
