"""The shared augmentation (Stage 5a; reused for the cached tile versions in Stage 6).

Training only: a random resized crop to the input size (scale 0.8-1.0, aspect 0.9-1.1), horizontal and
vertical flips, 90-degree rotations, rotations up to +-15 degrees, and colour jitter (brightness, contrast,
saturation +-0.2, hue +-0.02). Validation and test photos are only resized: ``evaluation`` has no randomness.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2


class RandomRot90(torch.nn.Module):
    """Rotate by 0, 90, 180 or 270 degrees, each with probability 1/4 (leaves have no "up")."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.rot90(image, int(torch.randint(4, (1,))), dims=(-2, -1))


def training(size: int) -> v2.Compose:
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), RandomRot90(), v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    ])


def evaluation(size: int) -> v2.Compose:
    return v2.Compose([v2.Resize((size, size), antialias=True)])
