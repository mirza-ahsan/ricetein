"""Checks for src/embed.py: seeded versions, scaling, and the items handed to the encoder."""

import numpy as np
from PIL import Image

from src import embed

AUGMENT = {"zoom": 0.15, "jitter": 0.2, "hue": 0.02}


def test_version_zero_is_unchanged_and_others_are_seeded_per_photo():
    assert embed.version_settings(AUGMENT, 0, 0, "a.jpg")["zoom"] == 1.0
    one = embed.version_settings(AUGMENT, 0, 3, "a.jpg")
    assert one == embed.version_settings(AUGMENT, 0, 3, "a.jpg")  # same photo, version and seed
    assert one != embed.version_settings(AUGMENT, 0, 3, "b.jpg")
    assert one != embed.version_settings(AUGMENT, 0, 4, "a.jpg")
    assert 0.85 <= one["zoom"] <= 1.15 and abs(one["hue"]) <= 0.02


def test_load_version_scales_and_keeps_version_zero_pixels(tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "p.png")
    clean = embed.version_settings(AUGMENT, 0, 0, "p")
    assert np.array_equal(embed.load_version(str(tmp_path / "p.png"), 1.0, clean), image)
    assert embed.load_version(str(tmp_path / "p.png"), 0.5, clean).shape == (150, 200, 3)
    turned = {**clean, "rot90": 1}
    assert embed.load_version(str(tmp_path / "p.png"), 1.0, turned).shape == (400, 300, 3)


def test_jobs_cut_tiles_and_shrink_big_tiles(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "dataset_dir", lambda config, name: tmp_path)
    Image.fromarray(np.full((500, 500, 3), 128, np.uint8)).save(tmp_path / "x.jpg")
    config = {"seed": 0, "augment": AUGMENT, "embed": {"tile": 224, "input": 224, "otsu": False}}
    jobs = embed.Jobs(config, [("x.jpg", 0)], {"x.jpg": 1.0})
    _, _, tiles, boxes = jobs[0]
    assert tiles.shape == (9, 224, 224, 3) and boxes.shape == (9, 4)
    config["embed"].update(tile=448, input=224)
    _, _, tiles, _ = jobs[0]
    assert tiles.shape == (4, 224, 224, 3)
    assert embed.photo_key("Healthy/IMG_1.jpg") == "Healthy__IMG_1"
