"""Known-answer checks for src/inventory.py on small synthetic images."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from src.inventory import jpeg_quality, scan


@pytest.fixture
def fake_dataset(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    Image.fromarray(pixels).save(tmp_path / "A" / "q75.jpg", quality=75)
    Image.fromarray(pixels).save(tmp_path / "B" / "q95.jpg", quality=95)
    good = (tmp_path / "A" / "q75.jpg").read_bytes()
    (tmp_path / "B" / "broken.jpg").write_bytes(good[: len(good) // 3])  # truncated JPEG
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


def test_scan_records_every_file_with_known_values(fake_dataset):
    frame = scan(fake_dataset, hash_size=8, draft_size=256, workers=2).set_index("photo_id")
    # sorted, and nothing skipped (not even files that are not images)
    assert list(frame.index) == ["A/q75.jpg", "B/broken.jpg", "B/q95.jpg", "notes.txt"]
    good = frame.loc["A/q75.jpg"]
    assert (good["width"], good["height"], good["format"], good["folder"]) == (400, 300, "JPEG", "A")
    assert good["sha256"] == hashlib.sha256((fake_dataset / "A" / "q75.jpg").read_bytes()).hexdigest()
    assert len(good["phash"]) == 16 and good["opened_ok"]
    assert good["jpeg_quality"] == pytest.approx(75, abs=2)
    assert frame.loc["B/q95.jpg", "jpeg_quality"] == pytest.approx(95, abs=2)
    for bad in ("B/broken.jpg", "notes.txt"):
        assert not frame.loc[bad, "opened_ok"] and frame.loc[bad, "error"]


def test_same_pixels_get_the_same_perceptual_hash(fake_dataset):
    frame = scan(fake_dataset, hash_size=8, draft_size=256, workers=1).set_index("photo_id")
    a, b = frame.loc["A/q75.jpg", "phash"], frame.loc["B/q95.jpg", "phash"]
    differing_bits = bin(int(a, 16) ^ int(b, 16)).count("1")
    assert differing_bits <= 8  # same picture at two JPEG qualities: a near-duplicate by the Stage 2b rule


def test_jpeg_quality_is_none_for_non_jpeg(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "x.png")
    with Image.open(tmp_path / "x.png") as image:
        assert jpeg_quality(image) is None
