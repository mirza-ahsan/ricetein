"""Otsu background removal, re-implemented as the pre-declared methodology states it (for comparability with
earlier work): grayscale, 5 x 5 Gaussian blur, Otsu's threshold, the leaf side chosen as the class that does
not dominate the image border, morphological opening and closing with an elliptical element of 1% of the
shorter side, then background pixels set to black. A fixed, per-photo operation (no data is learnt), so it
is allowed before the split (leakage rule 2).
"""

from __future__ import annotations

import cv2
import numpy as np


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean leaf mask (True = leaf) for an RGB uint8 image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([bright[0], bright[-1], bright[:, 0], bright[:, -1]])
    leaf = bright if border.mean() < 0.5 else 1 - bright  # the class that dominates the border is background
    size = max(3, round(0.01 * min(rgb.shape[:2])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    leaf = cv2.morphologyEx(leaf.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, kernel)
    return leaf.astype(bool)


def clean(rgb: np.ndarray) -> np.ndarray:
    """The image with every background pixel set to black."""
    return np.where(leaf_mask(rgb)[..., None], rgb, 0).astype(np.uint8)
