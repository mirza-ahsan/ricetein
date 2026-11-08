"""Stage 5: fine-tune a whole-image model with the shared recipe (Stage 5a, methodology Table II).

One run = one model, input size, Otsu setting, learning rate and round. AdamW (weight decay 0.05), 2 warm-up
epochs then cosine, at most 40 epochs, early stopping on validation macro-F1 (patience 8) keeping the best
checkpoint, class-weighted cross-entropy with label smoothing 0.1, mixed precision, batch 32 (16 at 448 px).
Only validation predictions are written here; the test fold is scored once, later, by ``src.evaluate``.

    uv run python -m src.train_wholeimage --config configs/wholeimage/convnext_tiny_224.yaml # all lrs+rounds
    uv run python -m src.train_wholeimage --config ... --lr 1e-4 --round 3                   # one run
"""

from __future__ import annotations

import argparse
import math
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import metrics
from src.config import load_config
from src.photo_dataset import PhotoDataset, photo_cache, photo_ids
from src.records import RunRecorder, completed_run, get_default_runs_dir, require_clean_code
from src.seeding import seed_everything
from src.split import read_folds


def build_model(config: dict[str, Any], size: int) -> nn.Module:
    """The pretrained model at its pinned revision, with a fresh 4-class head."""
    import timm

    from src.kaggle_bootstrap import pinned_snapshot

    spec = config["wholeimage"]["models"][config["wholeimage"]["model"]]
    folder = pinned_snapshot(spec["repo"], spec["revision"])
    kwargs = spec.get("size_kwargs", {}).get(size, {})
    return timm.create_model(spec["repo"].split("/")[1], pretrained=True, num_classes=len(config["classes"]),
                             pretrained_cfg_overlay={"file": f"{folder}/model.safetensors"}, **kwargs)


def run_name(config: dict[str, Any], lr: float, round_: int) -> str:
    w = config["wholeimage"]
    return f"stage5_{w['model']}_{w['size']}{'_otsu' if w['otsu'] else ''}_lr{lr:g}_r{round_}"


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """w_c = N / (K * N_c) from the training labels of this round (base.yaml loss.class_weighting)."""
    counts = np.bincount(labels, minlength=n_classes)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


@torch.inference_mode()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, config: dict[str, Any],
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(rows, probabilities) for every photo the loader serves."""
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    model.eval()
    rows, probs = [], []
    for images, _, row in loader:
        x = (images.to(device, non_blocking=True).float() / 255 - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            probs.append(torch.softmax(model(x).float(), dim=1).cpu())
        rows.append(row)
    return torch.cat(rows).numpy(), torch.cat(probs).numpy()


def prediction_table(folds: pd.DataFrame, rows: np.ndarray, probs: np.ndarray, round_: int, partition: str,
                     classes: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"round": round_, "partition": partition,
                        "photo_id": folds["photo_id"].to_numpy()[rows],
                        "true_class": folds["class"].to_numpy()[rows],
                        "pred_class": np.array(classes)[probs.argmax(axis=1)]})
    for k, c in enumerate(classes):
        out[f"prob_{c}"] = probs[:, k]
    return out


def loaders(config: dict[str, Any], folds: pd.DataFrame, round_: int, roles: tuple[str, ...],
            ) -> dict[str, torch.utils.data.DataLoader]:
    w = config["wholeimage"]
    images, ids = photo_cache(config, w["size"], w["otsu"]), photo_ids(config)
    batch = w["batch_size"][w["size"]]
    out = {}
    for role in roles:
        data = PhotoDataset(folds, images, ids, round_, role, w["size"], config["classes"],
                            augmented=role == "train")
        out[role] = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=role == "train",
                                                num_workers=w["workers"], pin_memory=True,
                                                drop_last=role == "train", persistent_workers=True)
    return out


def train(config: dict[str, Any], lr: float, round_: int) -> dict[str, Any]:
    require_clean_code()
    w = config["wholeimage"]
    seed_everything(config["seed"], w["determinism"])
    folds, checksum = read_folds(config)
    name = run_name(config, lr, round_)
    recorder = RunRecorder(name, config={**config, "lr": lr, "round": round_}, split_checksum=checksum,
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    device = torch.device("cuda")
    data = loaders(config, folds, round_, ("train", "val"))
    labels = data["train"].dataset.labels[data["train"].dataset.rows]
    model = build_model(config, w["size"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(labels, len(config["classes"])).to(device),
                                  label_smoothing=config["loss"]["label_smoothing"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=w["weight_decay"])
    steps_per_epoch = len(data["train"])
    total, warmup = w["max_epochs"] * steps_per_epoch, w["warmup_epochs"] * steps_per_epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler()
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    best, best_epoch, best_probs, best_rows = -1.0, 0, None, None
    for epoch in range(1, w["max_epochs"] + 1):
        model.train()
        losses = []
        for images, y, _ in data["train"]:
            x = (images.to(device, non_blocking=True).float() / 255 - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y.to(device))
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            losses.append(loss.item())
        rows, probs = predict(model, data["val"], config, device)
        val = metrics.scores(folds["class"].to_numpy()[rows], np.array(config["classes"])[probs.argmax(1)],
                             config["classes"])
        recorder.log_epoch(epoch, {"train_loss": float(np.mean(losses)), "val_macro_f1": val["macro_f1"],
                                   "val_balanced_accuracy": val["balanced_accuracy"],
                                   "lr": schedule.get_last_lr()[0]})
        if val["macro_f1"] > best:
            best, best_epoch, best_rows, best_probs = val["macro_f1"], epoch, rows, probs
            torch.save(model.state_dict(), recorder.run_dir / "model.pt")
        elif epoch - best_epoch >= w["patience"]:
            break
    predictions = prediction_table(folds, best_rows, best_probs, round_, "val", config["classes"])
    recorder.save_predictions(predictions)
    result = {"best_epoch": best_epoch, "epochs_run": epoch,
              "val": metrics.scores(predictions["true_class"], predictions["pred_class"], config["classes"])}
    recorder.save_metrics(result)
    recorder.finish()
    del model
    torch.cuda.empty_cache()
    return result


def done(name: str) -> bool:
    return completed_run(name, ("model.pt", "metrics.json", "predictions.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr", type=float, help="one learning rate (default: every one in the config)")
    parser.add_argument("--round", type=int, help="one round (default: all)")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["wholeimage"]["size"] != 224 or config["wholeimage"]["otsu"]:
        if args.lr is None:
            parser.error("128/Otsu/high-resolution runs require --lr selected on 224px validation; "
                         "use src.stage5")
    lrs = [args.lr] if args.lr else config["wholeimage"]["lrs"]
    rounds = [args.round] if args.round else range(1, config["split"]["n_folds"] + 1)
    for lr in lrs:
        for r in rounds:
            name = run_name(config, lr, r)
            if done(name):
                continue
            shutil.rmtree(get_default_runs_dir() / name, ignore_errors=True)  # an unfinished run starts again
            result = train(config, lr, r)
            print(f"{name}: val macro-F1 {result['val']['macro_f1']:.3f} (epoch {result['best_epoch']})")


if __name__ == "__main__":
    main()
