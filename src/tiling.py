"""The tiling grid (Stage 6.1; methodology eq. ntiles).

Per side, the smallest number of tiles that covers the whole side, n(L) = ceil(L / t), spread evenly so the
first and last tiles touch the edges; overlap then follows from the size. A side shorter than a tile gets
one tile, padded with black.
"""

from __future__ import annotations

import numpy as np


def starts(length: int, tile: int) -> list[int]:
    """Tile start positions along one side."""
    if length <= tile:
        return [0]
    n = -(-length // tile)
    return [i * (length - tile) // (n - 1) for i in range(n)]


def boxes(width: int, height: int, tile: int) -> np.ndarray:
    """(n, 4) int array of tile boxes (x, y, row, col), row-major."""
    return np.array([(x, y, r, c) for r, y in enumerate(starts(height, tile))
                     for c, x in enumerate(starts(width, tile))], dtype=np.int32)


def cut(image: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """(tiles (n, tile, tile, 3) uint8, boxes) for an RGB image; sides shorter than ``tile`` are padded."""
    h, w = image.shape[:2]
    if h < tile or w < tile:
        padded = np.zeros((max(h, tile), max(w, tile), 3), np.uint8)
        padded[:h, :w] = image
        image = padded
    grid = boxes(image.shape[1], image.shape[0], tile)
    return np.stack([image[y:y + tile, x:x + tile] for x, y, _, _ in grid]), grid
