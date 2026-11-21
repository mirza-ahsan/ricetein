"""Checks for src/distill.py (Stage 13): the student reads training photos only, the teacher is asked about training
photos only, no hard label reaches the student's training loss, and the soft-answer file must match the training
photos exactly (fake store and tiny models on the CPU; no GPU, no caches)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src import distill
from src.compress import role_rows
from src.config import load_config

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
PARAMS = {"lr": 0.01, "weight_decay": 0.0, "width": 8, "dropout": 0.0, "tile_dropout": 0.0, "descriptor": "cm"}
CONFIG = {"classes": CLASSES, "determinism": "strict",
          "mil": {"device": "cpu", "versions": 3, "batch": 4, "adam_eps": 1e-8, "max_epochs": 3, "patience": 5, "cpu_threads": 1}}


class Store:
    dim = 4

    def __init__(self):
        self.calls = []

    def bag(self, photo_id, version):
        self.calls.append((photo_id, version))
        return np.random.default_rng(int(photo_id) + 100 * version).standard_normal((3, 4)).astype(np.float16)


def folds():
    roles = ["train"] * 12 + ["val"] * 6 + ["test"] * 6
    return pd.DataFrame({"photo_id": [str(k) for k in range(24)], "class": CLASSES * 6,
                         **{f"round_{r}": roles for r in range(1, 6)}})


def one_hot_targets(table):
    targets = np.full((len(table), 4), np.nan, np.float32)
    rows = role_rows(table, "train")[1]
    targets[rows] = np.eye(4)[np.arange(len(rows)) % 4]
    return targets


def test_the_teacher_is_asked_about_training_photos_only_and_nothing_else_gets_an_answer():
    table, asked = folds(), []

    def member(rows):
        asked.append(rows.copy())
        return np.full((len(rows), 4), 0.25)

    targets = distill.teacher_targets(table, 1, [member, member], 1.0)
    assert all(table["round_1"].to_numpy()[rows].tolist() == ["train"] * len(rows) for rows in asked)
    assert np.isfinite(targets[:12]).all() and np.isnan(targets[12:]).all()
    assert np.allclose(targets[:12].sum(1), 1)


def test_the_student_trains_on_training_photos_and_validates_on_validation_photos_never_on_test():
    table, store = folds(), Store()
    best = distill.fit_student(store, table, 1, PARAMS, CONFIG, one_hot_targets(table), 0)
    roles = dict(zip(table["photo_id"], table["round_1"]))
    assert {roles[p] for p, _ in store.calls} == {"train", "val"}
    assert not any(roles[p] == "test" for p, _ in store.calls)
    assert set(table["photo_id"].to_numpy()[best["rows"]]) == {str(k) for k in range(12, 18)}


def test_a_missing_teacher_answer_for_a_training_photo_stops_the_run():
    table, targets = folds(), one_hot_targets(folds())
    targets[3] = np.nan
    with pytest.raises(ValueError, match="no answer"):
        distill.fit_student(Store(), table, 1, PARAMS, CONFIG, targets, 0)


def test_hard_labels_of_the_training_photos_do_not_reach_the_loss():
    """Same soft answers, different training labels: the trained weights are identical. (Validation labels are used
    to stop and are the same in both runs.)"""
    a = folds()
    b = a.copy()
    b.loc[:11, "class"] = b.loc[:11, "class"].map(dict(zip(CLASSES, CLASSES[1:] + CLASSES[:1])))
    first = distill.fit_student(Store(), a, 1, PARAMS, CONFIG, one_hot_targets(a), 0)
    second = distill.fit_student(Store(), b, 1, PARAMS, CONFIG, one_hot_targets(a), 0)
    assert first["epoch"] == second["epoch"]
    assert all(torch.equal(first["state"][k], second["state"][k]) for k in first["state"])


def test_student_loss_is_cross_entropy_with_the_teacher_probabilities():
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    one_hot = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    assert torch.isclose(distill.student_loss(logits, one_hot), torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])))
    soft = torch.full((2, 4), 0.25)
    assert torch.isclose(distill.student_loss(logits, soft), -torch.log_softmax(logits, 1).mean(1).mean())


def test_temperature_one_changes_nothing_and_larger_flattens():
    p = np.array([[0.7, 0.1, 0.1, 0.1]])
    assert np.allclose(distill.soften(p, 1.0), p)
    assert distill.soften(p, 2.0).max() < 0.7 and np.isclose(distill.soften(p, 2.0).sum(), 1)


def test_the_soft_answer_file_must_hold_exactly_the_training_photos_of_the_round(tmp_path):
    table = folds()
    rows = [{"photo_id": str(k), "round": 1, **{f"prob_{c}": 0.25 for c in CLASSES}} for k in range(12)]
    good = tmp_path / "good.csv"
    pd.DataFrame(rows).to_csv(good, index=False)
    targets = distill.file_targets(good, table, 1, CLASSES, 1.0)
    assert np.isfinite(targets[:12]).all() and np.isnan(targets[12:]).all()
    leaky = tmp_path / "leaky.csv"
    pd.DataFrame(rows + [{"photo_id": "13", "round": 1, **{f"prob_{c}": 0.25 for c in CLASSES}}]).to_csv(leaky, index=False)
    with pytest.raises(ValueError, match="exactly the training photos"):
        distill.file_targets(leaky, table, 1, CLASSES, 1.0)
    short = tmp_path / "short.csv"
    pd.DataFrame(rows[:-1]).to_csv(short, index=False)
    with pytest.raises(ValueError, match="exactly the training photos"):
        distill.file_targets(short, table, 1, CLASSES, 1.0)


def test_a_whole_image_teacher_member_scores_only_clean_training_photos():
    """The whole-image member sees every training photo of the round, each exactly as stored (no augmentation), and no
    validation or test photo; asking it about other photos stops before anything is loaded."""
    from src.photo_dataset import cache_size

    table, size = folds(), 14
    side = cache_size(size)
    images = np.stack([np.full((side, side, 3), 10 * (k + 1), np.uint8) for k in range(24)])   # photo k is the constant 10(k+1)
    seen = []

    class Spy(torch.nn.Module):
        def forward(self, x):
            raw = (x * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) * 255
            for image in raw:
                assert float(image.max() - image.min()) < 1e-3     # a constant image stays constant: nothing was augmented
                seen.append(int(round(float(image.mean()) / 10)) - 1)
            return torch.zeros(len(x), 4)

    config = {"classes": CLASSES, "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
              "wholeimage": {"size": size, "batch_size": {size: 5}, "workers": 0}}
    rows, probs = distill.cnn_answers(Spy(), config, table, 1, images, [str(k) for k in range(24)], torch.device("cpu"))
    assert sorted(seen) == list(range(12)) and rows.tolist() == list(range(12)) and probs.shape == (12, 4)
    with pytest.raises(ValueError, match="exactly the training photos"):
        distill.cnn_member("convnext_tiny_448", 1, table)(np.arange(18))      # includes validation photos
    with pytest.raises(ValueError, match="exactly the training photos"):
        distill.cnn_member("convnext_tiny_448", 1, table)(np.arange(12, 24))  # validation and test photos only


def test_committee_members_are_dispatched_by_kind():
    assert "convnext_tiny_448" in distill.CNN_MEMBERS and "attention" not in distill.CNN_MEMBERS
    config = load_config("configs/cost.yaml")
    assert config["distill"]["teacher"]["members"] == ["convnext_tiny_448", "attention"]
    assert config["cost"]["committee"]["members"] == config["distill"]["teacher"]["members"]
    assert config["cost"]["committee"]["test_macro_f1"] == pytest.approx(0.8155888862792725)
