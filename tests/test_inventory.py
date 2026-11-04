"""Known-answer checks for src/inventory.py on small synthetic datasets."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import config as cfgmod
from src.config import load_config
from src.inventory import (
    COLUMNS,
    dataset_inventory,
    jpeg_quality,
    load_for_hash,
    perceptual_hash,
    read_inventory,
    scan,
)


def gradient(width=400, height=300):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack([x * 255 // (width - 1), y * 255 // (height - 1), (x + y) * 255 // (width + height - 2)],
                    axis=-1).astype(np.uint8)


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset")
    (root / "A").mkdir()
    (root / "B").mkdir()
    Image.fromarray(gradient()).save(root / "A" / "grad.jpg", quality=90)
    Image.fromarray(gradient()).save(root / "B" / "grad_q100.jpg", quality=100)
    good = (root / "A" / "grad.jpg").read_bytes()
    (root / "B" / "broken.jpg").write_bytes(good[: len(good) // 3])  # truncated JPEG
    (root / ".DS_Store").write_text("junk")  # sorts first: must not change the column layout
    noise = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    Image.fromarray(noise).save(root / "A" / "upright.png")  # lossless, so the pair can match exactly
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways; viewers rotate it 90 degrees clockwise
    Image.fromarray(np.rot90(noise, k=1)).save(root / "A" / "rotated.png", exif=exif)
    return root, scan(root, hash_size=8, draft_size=256, workers=1).set_index("photo_id")


def test_every_file_is_listed_in_id_order_with_fixed_columns(scanned):
    _, frame = scanned
    expected = [".DS_Store", "A/grad.jpg", "A/rotated.png", "A/upright.png", "B/broken.jpg",
                "B/grad_q100.jpg"]
    assert list(frame.index) == expected
    assert list(frame.reset_index().columns) == COLUMNS
    assert str(frame["width"].dtype) == "Int64"  # never floats, even with failed files present
    assert pd.isna(frame.loc[".DS_Store", "folder"]) and frame.loc["A/grad.jpg", "folder"] == "A"


def test_parallel_scan_gives_the_same_result(scanned):
    root, frame = scanned
    parallel = scan(root, hash_size=8, draft_size=256, workers=2).set_index("photo_id")
    pd.testing.assert_frame_equal(parallel, frame)


def test_empty_folder_and_csv_round_trip(tmp_path, scanned):
    assert scan(tmp_path, hash_size=8, draft_size=256, workers=1).empty
    _, frame = scanned
    frame.reset_index().to_csv(tmp_path / "inv.csv", index=False)
    back = read_inventory(tmp_path / "inv.csv")  # broken files leave gaps; integers must stay integers
    assert str(back["width"].dtype) == "Int64" and back["width"].isna().any()


def test_known_values_for_a_good_file(scanned):
    root, frame = scanned
    good = frame.loc["A/grad.jpg"]
    assert (good["width"], good["height"], good["format"], good["mode"]) == (400, 300, "JPEG", "RGB")
    assert good["sha256"] == hashlib.sha256((root / "A" / "grad.jpg").read_bytes()).hexdigest()
    assert good["phash"] == "aa58e90d9178de47"  # pinned: Stage 2b compares against hashes made this way
    assert good["phash"] == perceptual_hash(load_for_hash(root / "A" / "grad.jpg", 256), 8)
    assert good["opened_ok"] and pd.isna(good["error"])


def test_broken_and_non_image_files_are_recorded_not_skipped(scanned):
    _, frame = scanned
    for bad in (".DS_Store", "B/broken.jpg"):
        assert not frame.loc[bad, "opened_ok"] and frame.loc[bad, "error"]


def test_rotation_tag_gives_upright_size_pixels_and_hash(scanned):
    root, frame = scanned
    rotated, upright = frame.loc["A/rotated.png"], frame.loc["A/upright.png"]
    assert rotated["exif_orientation"] == 6
    assert (rotated["width"], rotated["height"]) == (400, 300)  # as the pixels are used, not as stored
    assert np.array_equal(np.asarray(load_for_hash(root / "A" / "rotated.png", 256)),
                          np.asarray(load_for_hash(root / "A" / "upright.png", 256)))
    assert rotated["phash"] == upright["phash"]


@pytest.mark.parametrize("quality", [10, 50, 75, 90, 95, 100])
def test_jpeg_quality_is_recovered_exactly(tmp_path, quality):
    Image.fromarray(gradient()).save(tmp_path / "x.jpg", quality=quality)
    with Image.open(tmp_path / "x.jpg") as image:
        assert jpeg_quality(image) == (quality, True)


def test_jpeg_quality_is_none_for_non_jpeg(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "x.png")
    with Image.open(tmp_path / "x.png") as image:
        assert jpeg_quality(image) == (None, None)


def test_dataset_inventory_maps_classes_and_checks_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(cfgmod, "on_kaggle", lambda: False)
    config = load_config("configs/inventory.yaml")
    spec = config["datasets"]["paddy"]
    spec["local_dir"] = str(tmp_path)  # absolute: replaces the repo-relative default
    for folder in [*spec["class_folders"].values(), "tungro"]:
        (tmp_path / folder).mkdir()
    Image.fromarray(gradient()).save(tmp_path / "blast" / "P1.jpg")
    Image.fromarray(gradient()).save(tmp_path / "tungro" / "P2.jpg")
    header = "image_id,label,variety,age\n"
    (tmp_path / "metadata.csv").write_text(header + "P1.jpg,blast,ADT45,60\nP2.jpg,tungro,ADT45,70\n")
    frame = dataset_inventory(config, "paddy", workers=1).set_index("photo_id")
    assert list(frame.index) == ["blast/P1.jpg", "tungro/P2.jpg"]  # metadata.csv itself is not a photo
    assert frame.loc["blast/P1.jpg", "class"] == "leaf_blast" and pd.isna(frame.loc["tungro/P2.jpg", "class"])
    assert frame.loc["tungro/P2.jpg", "age"] == 70
    (tmp_path / "metadata.csv").write_text(header + "P1.jpg,blast,ADT45,60\n")
    with pytest.raises(ValueError, match="disagree"):
        dataset_inventory(config, "paddy", workers=1)
