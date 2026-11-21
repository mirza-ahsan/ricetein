"""Checks for src/compress.py (Stage 13): the fewer-tiles rule reads no label and only the photos asked for, the
tile cut matches the normal cut, validation work cannot touch test photos, the 8-bit subset is seeded and drawn from
validation photos, and the ONNX files match torch (fake models, no GPU, no photos, no caches)."""

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import compress, tiling
from src.config import load_config
from src.mil_models import MILModel

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
RULE = {"version": 1, "tile": 224, "tiles": 16, "shrink": 8, "min_leaf_share": 0.5}


def leaf_photo(height=1120, width=1568, patch=(448, 672)):
    """A green leaf on white with one brown patch: the tiles over the patch deviate most from the leaf colour."""
    rgb = np.full((height, width, 3), 255, np.uint8)
    rgb[100:height - 100, 100:width - 100] = (60, 140, 60)
    rgb[patch[0]:patch[0] + 224, patch[1]:patch[1] + 224] = (120, 70, 30)
    return rgb


def test_top_tiles_picks_the_tiles_over_the_deviating_patch_and_is_repeatable():
    rgb = leaf_photo()
    chosen = compress.top_tiles(rgb, RULE)
    grid = compress.grid_of(rgb)
    assert len(chosen) == 16 and chosen.tolist() == sorted(chosen.tolist()) and len(set(chosen)) == 16
    assert chosen.max() < len(grid)
    patch = [i for i, (x, y, _, _) in enumerate(grid) if x < 672 + 224 and x + 224 > 672 and y < 448 + 224 and y + 224 > 448]
    assert set(patch) <= set(chosen)
    assert np.array_equal(chosen, compress.top_tiles(rgb.copy(), RULE))


def test_the_rule_takes_pixels_and_settings_only_no_label_no_model():
    assert list(inspect.signature(compress.top_tiles).parameters) == ["rgb", "rule"]
    assert list(inspect.signature(compress._select_one).parameters) == ["job"]
    assert list(inspect.signature(compress.even_tiles).parameters) == ["n", "k"]
    # the selection is a function of the pixels: an all-leaf, uniform photo gives the same tiles whatever follows
    rgb = leaf_photo()
    assert np.array_equal(compress.top_tiles(rgb, RULE), compress.top_tiles(rgb, RULE))


def test_a_photo_with_no_more_tiles_than_wanted_keeps_them_all():
    assert compress.top_tiles(leaf_photo(448, 448, (0, 0)), RULE).tolist() == [0, 1, 2, 3]


def test_even_tiles():
    assert compress.even_tiles(79, 16).tolist() == sorted(set(compress.even_tiles(79, 16).tolist()))
    assert len(compress.even_tiles(79, 16)) == 16 and compress.even_tiles(79, 16)[[0, -1]].tolist() == [0, 78]
    assert compress.even_tiles(9, 16).tolist() == list(range(9))


@pytest.mark.parametrize("shape", [(1120, 1568), (200, 300)])      # the second is shorter than a tile: padded
def test_cut_selected_gives_the_pixels_of_the_normal_cut(shape):
    rgb = np.random.default_rng(0).integers(0, 255, (*shape, 3), dtype=np.uint8)
    everything = tiling.cut(rgb, 224)[0]
    indices = np.array([0, len(everything) - 1])
    assert np.array_equal(compress.cut_selected(rgb, indices), everything[indices])


def test_tile_selection_opens_only_the_photos_asked_for_and_caches(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(compress, "cache_folder", lambda name, write=False: tmp_path / name)
    monkeypatch.setattr("src.config.dataset_dir", lambda config, name: tmp_path)
    monkeypatch.setattr(compress, "read_folds", lambda config: (_ for _ in ()).throw(AssertionError("folds read")))
    monkeypatch.setattr(compress, "leaf_scale_photo", lambda path, decision: opened.append(path) or leaf_photo())
    config = load_config("configs/cost.yaml")
    first = compress.tile_selection(config, ["a/1.jpg", "b/2.jpg"], workers=1)
    assert [p.split("/")[-2:] for p in opened] == [["a", "1.jpg"], ["b", "2.jpg"]]
    assert all(entry["n"] == len(compress.grid_of(leaf_photo())) and len(entry["tiles"]) == 16 for entry in first.values())
    assert compress.tile_selection(config, ["b/2.jpg"], workers=1) == {"b/2.jpg": first["b/2.jpg"]} and len(opened) == 2
    config["compress"]["tiles"] = 8
    with pytest.raises(ValueError, match="another rule"):
        compress.tile_selection(config, ["a/1.jpg"], workers=1)


def test_selected_policy_refuses_a_grid_that_differs_from_the_cache():
    policy = compress.selected_policy({"p": {"n": 79, "tiles": [1, 5]}})
    assert policy("p", 79).tolist() == [1, 5]
    with pytest.raises(ValueError, match="80 tiles in the cache but 79"):
        policy("p", 80)


def fake_folds():
    return pd.DataFrame({"photo_id": [str(k) for k in range(12)], "class": CLASSES * 3,
                         **{f"round_{r}": ["train"] * 4 + ["val"] * 4 + ["test"] * 4 for r in range(1, 6)}})


def test_evaluation_applies_the_policy_and_cannot_touch_photos_of_another_role():
    folds, asked = fake_folds(), []
    heads = [MILModel(4, 8, 0.0, "attention", 4).eval() for _ in range(5)]

    def bag_of(photo):
        asked.append(photo)
        return np.random.default_rng(int(photo)).standard_normal((10, 4)).astype(np.float32)

    rows = compress.role_rows(folds, "val")
    scores, table, used = compress.evaluate_rows(heads, bag_of, folds, rows, compress.even_policy(3), CLASSES, "val")
    assert set(asked) == {"4", "5", "6", "7"} and used == 3 and len(scores) == 5
    assert set(table["partition"]) == {"val"} and len(table) == 20
    with pytest.raises(ValueError, match="not test photos"):
        compress.evaluate_rows(heads, bag_of, folds, rows, compress.all_tiles, CLASSES, "test")
    with pytest.raises(ValueError, match="not val photos"):
        compress.evaluate_rows(heads, bag_of, folds, compress.role_rows(folds, "test"), compress.all_tiles, CLASSES, "val")


def test_the_8_bit_subset_is_seeded_class_proportional_and_validation_only():
    rng = np.random.default_rng(0)
    n = 600
    folds = pd.DataFrame({"photo_id": [f"p{k}" for k in range(n)], "class": rng.choice(CLASSES, n, p=[.5, .2, .15, .15])})
    for r in range(1, 6):
        folds[f"round_{r}"] = rng.choice(["train", "val", "test"], n, p=[.7, .15, .15])
    union = folds[(folds[[f"round_{r}" for r in range(1, 6)]] == "val").any(axis=1)]
    a, b = compress.int8_subset(folds, 100, 0), compress.int8_subset(folds, 100, 0)
    assert a == b and a != compress.int8_subset(folds, 100, 1) and a == sorted(a)
    assert set(a) <= set(union["photo_id"]) and abs(len(a) - 100) <= 4
    share = folds.set_index("photo_id").loc[a, "class"].value_counts(normalize=True)
    assert abs(share["healthy"] - union["class"].value_counts(normalize=True)["healthy"]) < 0.05


def test_quantise_keeps_the_input_model_and_stays_close():
    encoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 16)
    quantised = compress.quantise(encoder)
    assert not any("quantized" in type(m).__module__ for m in encoder.modules())
    assert any("quantized" in type(m).__module__ for m in quantised.modules())
    assert (quantised(x) - encoder(x)).abs().max() < 0.1


class FakeEncoder(nn.Module):
    """Stands in for a transformers ViT: (N, 3, H, W) -> an object with last_hidden_state of 1 + 1 + 2 tokens."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 4 * 5)

    def forward(self, pixel_values):
        tokens = self.lin(pixel_values.mean(dim=(2, 3))).view(pixel_values.shape[0], 4, 5)
        return SimpleNamespace(last_hidden_state=tokens)


def test_onnx_files_match_torch_with_a_changing_number_of_tiles(tmp_path):
    encoder = compress.DescribeModule(FakeEncoder(), [0.5] * 3, [0.2] * 3, 2, "cm")
    pixels = np.random.default_rng(0).random((2, 3, 8, 8), dtype=np.float32)
    compress.export_onnx(encoder, (torch.from_numpy(pixels),), tmp_path / "e.onnx", ["pixels"],
                         {"pixels": {0: "tiles"}, "output": {0: "tiles"}})
    session = compress.onnx_session(tmp_path / "e.onnx", 1)
    assert compress.max_difference(encoder, session, {"pixels": pixels}) < 1e-5
    more = np.random.default_rng(1).random((5, 3, 8, 8), dtype=np.float32)
    assert compress.max_difference(encoder, session, {"pixels": more}) < 1e-5

    head = compress.HeadModule(MILModel(10, 16, 0.0, "attention", 4).eval())
    tiles, mask = np.random.default_rng(2).standard_normal((1, 7, 10)).astype(np.float32), np.ones((1, 7), bool)
    compress.export_onnx(head, (torch.from_numpy(tiles), torch.from_numpy(mask)), tmp_path / "h.onnx", ["tiles", "mask"],
                         {"tiles": {1: "n"}, "mask": {1: "n"}})
    session = compress.onnx_session(tmp_path / "h.onnx", 1)
    fewer = {"tiles": tiles[:, :4], "mask": np.array([[True, True, False, True]])}
    assert compress.max_difference(head, session, fewer) < 1e-5


def test_variants_and_candidates_are_consistent():
    assert compress.CANDIDATES == [n for n in compress.VARIANTS if n not in ("attention", "e1_even16", "e4_even16")]
    assert all(compress.VARIANTS[n][0] in compress.ENCODER for n in compress.VARIANTS)
    assert compress.head_runs("e4")[0] == "stage7_session_ablate_encoder_e4_r1" and len(compress.head_runs("distilled_e4")) == 5


def test_description_cosine_is_one_for_equal_descriptions_and_zero_for_orthogonal_ones():
    bags = {"a": np.eye(3, dtype=np.float32), "b": 2 * np.eye(3, dtype=np.float32)}
    assert compress.description_cosine(bags.get, bags.get, ["a", "b"]) == pytest.approx(1.0)
    flipped = {"a": np.eye(3, dtype=np.float32)[::-1].copy()}
    assert compress.description_cosine({"a": np.eye(3, dtype=np.float32)}.get, flipped.get, ["a"]) == pytest.approx(1 / 3)
