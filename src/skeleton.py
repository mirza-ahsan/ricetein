"""Stage 0 walking skeleton: the whole pipeline, end to end, on about 20 photos.

Throwaway code (handbook Stage 0). Load photos -> cut native-resolution tiles -> describe
every tile with the frozen encoder -> train a tiny gated-attention head for a few steps ->
predict -> score -> write a run record. It scores badly (20 photos, 50 steps); it exists to
prove the pipeline runs and gives the same numbers on the local machine and on Kaggle.

    uv run python -m src.skeleton --config configs/skeleton.yaml --run-name stage0_skeleton_halo
    uv run python -m src.skeleton --compare runs/stage0_skeleton_halo runs/stage0_skeleton_kaggle
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

from src.config import REPO_ROOT, dataset_dir, load_config, path_for
from src.kaggle_bootstrap import pretrained_source
from src.records import RunRecorder
from src.seeding import seed_everything

CODE_PATHS = ["src", "configs", "splits", "pyproject.toml", "uv.lock"]


# ---------------------------------------------------------------- data
def sample_photos(config: dict[str, Any]) -> list[dict[str, Any]]:
    """A fixed, seeded sample per class; the first ``train_per_class`` are training photos."""
    sk = config["skeleton"]
    root = dataset_dir(config, "primary")
    ext = config["datasets"]["primary"]["image_extension"]
    rng = np.random.default_rng(config["seed"])
    photos = []
    for label, name in enumerate(config["classes"]):
        folder = config["datasets"]["primary"]["class_folders"][name]
        files = sorted((root / folder).glob(f"*{ext}"))
        for rank, index in enumerate(rng.choice(len(files), sk["photos_per_class"], replace=False)):
            photos.append({
                "photo_id": f"{folder}/{files[index].name}",
                "path": files[index],
                "label": label,
                "partition": "train" if rank < sk["train_per_class"] else "test",
            })
    return photos


def tile_starts(length: int, tile: int) -> list[int]:
    """Smallest number of tiles covering ``length``, evenly spread, first and last touching the edges."""
    if length <= tile:
        return [0]
    n = -(-(length - tile) // tile) + 1  # ceil((L - t) / t) + 1
    return [i * (length - tile) // (n - 1) for i in range(n)]


def cut_tiles(image: np.ndarray, tile: int) -> np.ndarray:
    """All tiles of an (H, W, 3) image as an (N, tile, tile, 3) array, row by row."""
    height, width = image.shape[:2]
    return np.stack([
        image[top:top + tile, left:left + tile]
        for top in tile_starts(height, tile)
        for left in tile_starts(width, tile)
    ])


# ---------------------------------------------------------------- encoder
def load_encoder(config: dict[str, Any], device: torch.device) -> nn.Module:
    from transformers import AutoModel

    spec = config["encoders"][config["skeleton"]["encoder"]]
    source = pretrained_source(spec["name"])
    pinned = {"revision": spec["revision"]} if source == spec["name"] else {}  # Kaggle copy is already pinned
    return AutoModel.from_pretrained(source, dtype=torch.float32, **pinned).eval().to(device)


@torch.inference_mode()
def describe_tiles(
    model: nn.Module, tiles: np.ndarray, config: dict[str, Any], device: torch.device
) -> torch.Tensor:
    """Class-token description of every tile, (N, dim), on the CPU."""
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    out = []
    for start in range(0, len(tiles), config["skeleton"]["encode_batch"]):
        batch = torch.from_numpy(tiles[start:start + config["skeleton"]["encode_batch"]]).to(device)
        pixels = (batch.permute(0, 3, 1, 2).float() / 255 - mean) / std
        out.append(model(pixel_values=pixels).last_hidden_state[:, 0].cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- MIL head
class GatedAttentionHead(nn.Module):
    """Gated-attention multiple instance learning (Ilse et al. 2018): bag of tiles -> class scores."""

    def __init__(self, in_dim: int, projection_dim: int, attention_dim: int, n_classes: int) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, projection_dim), nn.ReLU())
        self.attention_v = nn.Linear(projection_dim, attention_dim)
        self.attention_u = nn.Linear(projection_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classify = nn.Linear(projection_dim, n_classes)

    def forward(self, tiles: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tiles (B, N, in_dim), mask (B, N) True for real tiles -> logits (B, C), weights (B, N)."""
        h = self.project(tiles)
        gated = torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
        scores = self.attention_w(gated).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        return self.classify((weights.unsqueeze(-1) * h).sum(dim=1)), weights


def pad_bags(bags: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack bags of different sizes into (B, N_max, dim) plus a mask of the real tiles."""
    longest = max(len(b) for b in bags)
    x = torch.zeros(len(bags), longest, bags[0].shape[1])
    mask = torch.zeros(len(bags), longest, dtype=torch.bool)
    for i, bag in enumerate(bags):
        x[i, :len(bag)] = bag
        mask[i, :len(bag)] = True
    return x, mask


# ---------------------------------------------------------------- run
def run(config_path: str, run_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(config["seed"], config["determinism"])
    sk, classes = config["skeleton"], config["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rec = RunRecorder(
        run_name, config=config, base_dir=path_for(config, "runs"),
        data_version={"primary": config["datasets"]["primary"]["version"]},
    )
    photos = sample_photos(config)
    encoder = load_encoder(config, device)
    for photo in photos:  # one photo at a time keeps memory low (photos are up to 3120 x 3120)
        with Image.open(photo["path"]) as img:
            image = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
        tiles = cut_tiles(image, config["tiling"]["tile_size"])
        photo["features"] = describe_tiles(encoder, tiles, config, device)
        rec.logger.info(f"{photo['photo_id']}: {image.shape[1]} x {image.shape[0]} px -> {len(tiles)} tiles")
    del encoder

    train = [p for p in photos if p["partition"] == "train"]
    x, mask = (t.to(device) for t in pad_bags([p["features"] for p in train]))
    y = torch.tensor([p["label"] for p in train], device=device)
    counts = torch.bincount(y, minlength=len(classes)).float()
    weights = len(y) / (len(classes) * counts.clamp(min=1))  # inverse frequency (base.yaml loss)

    # Built on the CPU first, so its starting weights match on every machine.
    dims = sk["head"]
    head = GatedAttentionHead(x.shape[-1], dims["projection_dim"], dims["attention_dim"], len(classes))
    head = head.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=config["loss"]["label_smoothing"])
    training = sk["training"]
    optimiser = torch.optim.AdamW(head.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    for epoch in range(1, training["steps"] + 1):  # full batch: one step per epoch
        optimiser.zero_grad()
        loss = loss_fn(head(x, mask)[0], y)
        loss.backward()
        optimiser.step()
        rec.log_epoch(epoch, {"train_loss": loss.item()})

    head.eval()
    rows = []
    with torch.inference_mode():
        for photo in photos:
            bag = photo["features"].unsqueeze(0).to(device)
            logits, _ = head(bag, torch.ones(bag.shape[:2], dtype=torch.bool, device=device))
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            rows.append({
                "photo_id": photo["photo_id"], "partition": photo["partition"],
                "true_class": classes[photo["label"]], "pred_class": classes[int(np.argmax(probs))],
                **{f"prob_{c}": p for c, p in zip(classes, probs)}, "n_tiles": len(photo["features"]),
            })
    predictions = pd.DataFrame(rows)
    rec.save_predictions(predictions)

    test = predictions[predictions["partition"] == "test"]
    truth, pred = test["true_class"], test["pred_class"]
    metrics = {
        "partition": "test",
        "note": "walking skeleton: throwaway 3+2 per class split, not a real result",
        "n_photos": len(test),
        "macro_f1": f1_score(truth, pred, labels=classes, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "accuracy": accuracy_score(truth, pred),
        "n_tiles_total": int(predictions["n_tiles"].sum()),
    }
    rec.save_metrics(metrics)
    rec.finish()
    print(f"| {run_name} | macro-F1 {metrics['macro_f1']:.3f} "
          f"| balanced acc {metrics['balanced_accuracy']:.3f} | acc {metrics['accuracy']:.3f} "
          f"| {metrics['n_tiles_total']} tiles |")
    return metrics


# ---------------------------------------------------------------- compare
def compare_runs(run_a: str | Path, run_b: str | Path, atol: float) -> bool:
    """Stage 0 'done when': same predicted labels and scores, probabilities within ``atol``."""
    a, b = Path(run_a), Path(run_b)
    pa = pd.read_csv(a / "predictions.csv").sort_values("photo_id").reset_index(drop=True)
    pb = pd.read_csv(b / "predictions.csv").sort_values("photo_id").reset_index(drop=True)
    checks = {"same photos": pa["photo_id"].equals(pb["photo_id"])}
    if checks["same photos"]:
        prob_cols = [c for c in pa.columns if c.startswith("prob_")]
        max_diff = float((pa[prob_cols] - pb[prob_cols]).abs().max().max())
        checks["same tile counts"] = pa["n_tiles"].equals(pb["n_tiles"])
        checks["same predicted labels"] = pa["pred_class"].equals(pb["pred_class"])
        checks[f"probabilities within {atol:g} (max diff {max_diff:.2e})"] = max_diff <= atol
    ma, mb = (json.loads((r / "metrics.json").read_text()) for r in (a, b))
    checks["same scores"] = all(ma[k] == mb[k] for k in ("macro_f1", "balanced_accuracy", "accuracy"))

    va, vb = (json.loads((r / "record.json").read_text())["code_version"] for r in (a, b))
    same_code = va == vb or subprocess.run(
        ["git", "diff", "--quiet", va, vb, "--", *CODE_PATHS], cwd=REPO_ROOT, capture_output=True
    ).returncode == 0
    checks[f"same code ({va} vs {vb})"] = same_code

    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/skeleton.yaml")
    parser.add_argument("--run-name")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="check two run folders agree")
    args = parser.parse_args()
    if args.compare:
        atol = load_config(args.config)["skeleton"]["compare_atol"]
        sys.exit(0 if compare_runs(*args.compare, atol=atol) else 1)
    if not args.run_name:
        parser.error("--run-name is required (e.g. stage0_skeleton_halo)")
    run(args.config, args.run_name)


if __name__ == "__main__":
    main()
