"""Checks for src/tiling.py: full coverage, overlap limits, and the known counts."""

import numpy as np
import pytest

from src import tiling


@pytest.mark.parametrize("side, count", [(734, 16), (1881, 81), (3120, 196), (1824, 81), (224, 1), (225, 4)])
def test_known_tile_counts(side, count):
    assert len(tiling.boxes(side, side, 224)) == count


@pytest.mark.parametrize("length", [224, 300, 734, 1000, 1488, 1881, 2056, 3120])
def test_tiles_cover_every_pixel_and_touch_both_edges(length):
    s = tiling.starts(length, 224)
    covered = np.zeros(length, bool)
    for x in s:
        covered[x:x + 224] = True
    assert covered.all() and s[0] == 0 and s[-1] + 224 == length
    if len(s) > 1:  # evenly spread: neighbouring starts differ by at most 1 px
        gaps = np.diff(s)
        assert gaps.max() - gaps.min() <= 1 and gaps.max() <= 224


def test_cut_returns_the_right_pixels_and_pads_small_images():
    image = np.random.default_rng(0).integers(0, 255, (500, 700, 3), dtype=np.uint8)
    tiles, grid = tiling.cut(image, 224)
    assert tiles.shape == (len(grid), 224, 224, 3)
    x, y, r, c = grid[-1]
    assert np.array_equal(tiles[-1], image[y:y + 224, x:x + 224]) and (r, c) == (2, 3)
    small, _ = tiling.cut(image[:100, :150], 224)
    assert small.shape == (1, 224, 224, 3) and small[0, 150:].max() == 0
