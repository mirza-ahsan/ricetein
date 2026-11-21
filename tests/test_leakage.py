"""Leakage rules for the whole-photo loader (leakage rules 2 and 5; written commitment P1-4): validation and
test photos are never augmented, each dataset serves exactly its role, and roles never mix within a round."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.photo_dataset import PhotoDataset

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    n = 40
    folds = pd.DataFrame({"photo_id": [f"p{i:02d}" for i in range(n)], "class": rng.choice(CLASSES, n)})
    folds["round_1"] = np.where(np.arange(n) < 8, "test", np.where(np.arange(n) < 12, "val", "train"))
    images = rng.integers(0, 255, (n, 40, 40, 3), dtype=np.uint8)
    ids = folds["photo_id"].sample(frac=1, random_state=1).tolist()  # cache order differs from folds order
    images = images[pd.Index(folds["photo_id"]).get_indexer(ids)]
    return folds, images, ids


@pytest.mark.parametrize("role", ["val", "test"])
def test_validation_and_test_photos_can_never_be_augmented(setup, role):
    folds, images, ids = setup
    with pytest.raises(ValueError, match="training photos only"):
        PhotoDataset(folds, images, ids, 1, role, 32, CLASSES, augmented=True)


@pytest.mark.parametrize("role", ["val", "test"])
def test_evaluation_photos_are_deterministic(setup, role):
    folds, images, ids = setup
    data = PhotoDataset(folds, images, ids, 1, role, 32, CLASSES, augmented=False)
    first = [data[i][0] for i in range(len(data))]
    again = [data[i][0] for i in range(len(data))]
    assert all(torch.equal(a, b) for a, b in zip(first, again))


def test_each_dataset_serves_only_its_role_and_the_right_photo(setup):
    folds, images, ids = setup
    served = {}
    for role in ("train", "val", "test"):
        data = PhotoDataset(folds, images, ids, 1, role, 40, CLASSES, augmented=False)
        served[role] = {data[i][2] for i in range(len(data))}
        assert set(folds.loc[list(served[role]), "round_1"]) == {role}
        row = data.rows[0]  # unaugmented 40 px image equals the cached image of that photo
        expected = torch.from_numpy(images[ids.index(folds.loc[row, "photo_id"])]).permute(2, 0, 1)
        assert torch.equal(data[0][0], expected)
    assert not (served["train"] & served["val"]) and not (served["val"] & served["test"])
    assert len(served["train"] | served["val"] | served["test"]) == len(folds)


def test_training_photos_are_augmented(setup):
    folds, images, ids = setup
    data = PhotoDataset(folds, images, ids, 1, "train", 32, CLASSES, augmented=True)
    torch.manual_seed(0)
    assert not all(torch.equal(data[0][0], data[0][0]) for _ in range(5))
