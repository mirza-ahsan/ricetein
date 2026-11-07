"""Frozen encoders (methodology Table I): load one at its pinned revision and describe images.

E1, E3, E4 and E5 are Hugging Face ``transformers`` vision transformers. Each returns its class token and
the average of its patch tokens; register tokens (DINOv3: 4, right after the class token) are internal
scratch space and are left out of the patch average. E2 (ConvNeXt-Tiny, ``timm``) is added in Stage 6,
the first stage that uses it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.kaggle_bootstrap import pinned_pretrained


def load_encoder(config: dict[str, Any], code: str, device: torch.device) -> nn.Module:
    """Encoder ``code`` (e.g. "E1") from ``config["encoders"]``, frozen, in eval mode, on ``device``."""
    spec = config["encoders"][code]
    if spec["library"] != "transformers":
        raise NotImplementedError(f"{code} ({spec['library']}) is added in Stage 6")
    from transformers import AutoModel

    source, pinned = pinned_pretrained(spec["name"], spec["revision"])
    model = AutoModel.from_pretrained(source, dtype=torch.float32, **pinned).eval().to(device)
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def describe(model: nn.Module, images: np.ndarray | torch.Tensor, spec: dict[str, Any],
             config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(class_token, patch_mean)``, each (N, dim) on the CPU, for uint8 images of shape (N, H, W, 3).

    Images are scaled to [0, 1] and normalised with the encoder's statistics (``config["normalisation"]``).
    """
    pixels = torch.as_tensor(images).to(device).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    tokens = model(pixel_values=(pixels - mean) / std).last_hidden_state
    first_patch = 1 + spec["register_tokens"]
    return tokens[:, 0].cpu(), tokens[:, first_patch:].mean(dim=1).cpu()
