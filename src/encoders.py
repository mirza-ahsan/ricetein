"""Frozen encoders (methodology Table I): load one at its pinned revision and describe images.

E1, E3, E4 and E5 are Hugging Face ``transformers`` vision transformers. Each returns its class token and
the average of its patch tokens; register tokens (DINOv3: 4, right after the class token) are internal
scratch space and are left out of the patch average. E2 (ConvNeXt-Tiny, ``timm``) returns its final pooled
output as the "class token" and no patch average.
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
    if spec["library"] == "timm":
        import timm
        from huggingface_hub import snapshot_download

        folder = snapshot_download(f"timm/{spec['name']}", revision=spec["revision"])
        model = timm.create_model(spec["name"], pretrained=True, num_classes=0,
                                  pretrained_cfg_overlay={"file": f"{folder}/model.safetensors"})
    else:
        from transformers import AutoModel

        source, pinned = pinned_pretrained(spec["name"], spec["revision"])
        model = AutoModel.from_pretrained(source, dtype=torch.float32, **pinned)
    model = model.eval().to(device)
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def describe(model: nn.Module, images: np.ndarray | torch.Tensor, spec: dict[str, Any],
             config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(class_token, patch_mean)``, each (N, dim) on the CPU, for uint8 images of shape (N, H, W, 3);
    for a timm encoder, ``(pooled, None)``.

    Images are scaled to [0, 1] and normalised with the encoder's statistics (``config["normalisation"]``).
    """
    pixels = torch.as_tensor(images).to(device).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    if spec["library"] == "timm":
        return model((pixels - mean) / std).float().cpu(), None
    tokens = model(pixel_values=(pixels - mean) / std).last_hidden_state
    first_patch = 1 + spec["register_tokens"]
    return tokens[:, 0].cpu(), tokens[:, first_patch:].mean(dim=1).cpu()
