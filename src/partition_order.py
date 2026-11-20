"""Stage 9: how much do testing mistakes inflate scores? (declared 2026-11-20, handbook §24)

Condition A is the correct protocol: the five existing session-blocked runs of each model, reused as they are.
B, C and D are DELIBERATELY WRONG demonstrations (src/partition_splits.py): B augments first and splits the copies
at random, C ignores near-duplicate groups and capture blocks, D trains and scores on every photo. They reuse
A's selected settings (no tuning), have their own split files, and write only runs named ``stage9_*``. Their
numbers feed no tuning, selection or other table.

    uv run python -m src.partition_order <action>

splits    write splits/partition_order/ once and print each condition's leak counter (no GPU)
declare   freeze runs/stage9_declared.json: the settings, the split files and the A runs (once; never deleted)
cache     build the 8 fixed augmented copies of every photo that condition B needs (CPU)
fit       train the 30 runs (2 models x B, C, D x 5); resumable; refuses once scoring has been frozen
score     freeze runs/stage9_evaluation_plan.json, then score each run once (score_started.json is never deleted)
report    results/tables/partition_order_inflation.csv and docs/reports/partition_order_report.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import augment, evaluate, metrics, stats
from src import train_mil as tm
from src import train_wholeimage as tw
from src.config import REPO_ROOT, load_config, output_root
from src.embed import completed_cache
from src.inventory import read_inventory
from src.inventory_report import md_table
from src.mil_models import MILModel
from src.partition_splits import (
    BOTH,
    FILENAMES,
    condition_tables,
    leak_counter,
    lock_tables,
    photo_keys,
    read_condition,
)
from src.photo_dataset import PhotoDataset, photo_cache, photo_ids
from src.records import RunRecorder, completed_run, get_default_runs_dir, require_clean_code, reusable_run
from src.seeding import seed_everything
from src.split import round_column

ROUNDS = range(1, 6)
DECLARED, PLAN = "stage9_declared.json", "stage9_evaluation_plan.json"
TABLE, REPORT = "results/tables/partition_order_inflation.csv", "docs/reports/partition_order_report.md"
ARTIFACTS = {"B": ("model.pt", "metrics.json", "predictions.csv"), "C": ("model.pt", "metrics.json", "predictions.csv"),
             "D": ("model.pt", "metrics.json")}   # D has no validation photos: it trains for A's best epoch
MEANING = {"A": "correct: grouped by capture block, augmentation on training photos only",
           "L": "reference: locked split (duplicates grouped, capture sessions not), its own selected settings",
           "B": "augment first, then split the copies at random",
           "B_clean": "as B, scored on the clean copy of each photo only",
           "C": "no grouping: photos split at random", "D": "train on every photo, score on every photo"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_once(path: Path, data: Any) -> None:
    """Create the file with ``data``; an existing file must hold the same data."""
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if read_json(path) != json.loads(text):
            raise ValueError(f"{path} exists with other content; it is frozen")
        return
    with path.open("x") as out:
        out.write(text)


def run_name(model: str, condition: str, r: int) -> str:
    return f"stage9_{model}_{condition}_r{r}"


def reference_runs(config: dict[str, Any], model: str, which: str) -> list[str]:
    """The five runs of ``model`` in condition A (``which="a"``) or the locked-split reference L (``"l"``)."""
    po = config["partition_order"]
    selection = read_json(get_default_runs_dir() / po[f"{which}_selection"])
    return selection["methods"][po[f"{which}_methods"][model]]["runs"]


def with_role(units: pd.DataFrame, r: int, role: str) -> pd.DataFrame:
    """Condition D's ``train_and_test`` rows act as ``role``; the other conditions are unchanged."""
    column = round_column(r)
    return units.assign(**{column: units[column].replace(BOTH, role)})


# ---------------------------------------------------------------- declaration

def split_hashes(config: dict[str, Any]) -> dict[str, str]:
    folder = REPO_ROOT / config["partition_order"]["split_dir"]
    return {c: evaluate.digest(folder / f) for c, f in FILENAMES.items()}


def protocol(config: dict[str, Any]) -> dict[str, Any]:
    """What the declaration freezes: the settings, the three split files, and the A and L runs by file hash."""
    root, runs = get_default_runs_dir(), {}
    for which in ("a", "l"):
        for model in config["partition_order"]["models"]:
            for name in reference_runs(config, model, which):
                runs[name] = {"record_sha256": evaluate.digest(root / name / "record.json"),
                              "test_predictions_sha256": evaluate.digest(root / name / "test_predictions.csv")}
    return {"partition_order": config["partition_order"], "split_files": split_hashes(config),
            "reference_runs": runs}


def declare(config: dict[str, Any]) -> None:
    write_once(get_default_runs_dir() / DECLARED, protocol(config))
    print(f"declared: {DECLARED}", flush=True)


def require_declared(config: dict[str, Any]) -> None:
    path = get_default_runs_dir() / DECLARED
    if not path.exists() or read_json(path) != json.loads(json.dumps(protocol(config))):
        raise RuntimeError(f"runs/{DECLARED} is missing or differs from the settings, split files or A runs")


# ---------------------------------------------------------------- data for condition B

def build_copies(images: np.ndarray, ids: list[str], folder: Path, config: dict[str, Any]) -> np.memmap:
    """Version 0 (clean) and 7 augmented copies of every photo at 224 px, made with the augmentation of the main
    protocol (``src.augment``), frozen before any split is drawn. Built once; a different identity refuses."""
    po = config["partition_order"]
    shape = (po["versions"], len(ids), 224, 224, 3)
    identity = {"ids": ids, "shape": list(shape), "seed": config["seed"],
                "augment_sha256": evaluate.digest(REPO_ROOT / "src/augment.py"),
                "photos": evaluate.digest(REPO_ROOT / "src/photo_dataset.py")}
    folder.mkdir(parents=True, exist_ok=True)
    write_once(folder / "spec.json", identity)
    data = folder / "copies.u8"
    if not (folder / "complete.json").exists():
        temporary = data.with_suffix(".tmp")
        out = np.memmap(temporary, np.uint8, "w+", shape=shape)
        for version in range(shape[0]):
            transform = augment.evaluation(224) if version == 0 else augment.training(224)
            for i, photo_id in enumerate(ids):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed((config["seed"] + version * 1000003 + zlib.crc32(photo_id.encode())) % 2**32)
                    x = torch.from_numpy(np.array(images[i])).permute(2, 0, 1)
                    out[version, i] = transform(x).permute(1, 2, 0).numpy()
        out.flush()
        del out
        temporary.replace(data)
        write_once(folder / "complete.json", identity)
    if read_json(folder / "complete.json") != json.loads(json.dumps(identity)) or data.stat().st_size != math.prod(shape):
        raise ValueError(f"{folder}: damaged or incompatible copy cache")
    return np.memmap(data, np.uint8, "r", shape=shape)


class CopyDataset(torch.utils.data.Dataset):
    """Fixed augmented copies (condition B): (uint8 CHW image, class index, row). Never augmented again."""

    def __init__(self, units: pd.DataFrame, copies: np.ndarray, image_ids: list[str], r: int, role: str,
                 classes: list[str]) -> None:
        self.rows = np.flatnonzero((units[round_column(r)] == role).to_numpy())
        self.labels = units["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        self.cache_row = pd.Index(image_ids).get_indexer(units["photo_id"])
        if (self.cache_row[self.rows] < 0).any():
            raise ValueError("some photos of this round are not in the copy cache")
        self.version, self.copies = units["version"].to_numpy(int), copies

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[i]
        image = np.array(self.copies[self.version[row], self.cache_row[row]])
        return torch.from_numpy(image).permute(2, 0, 1), int(self.labels[row]), int(row)


class Images:
    """The photo caches whole-image runs read: the resized photos, and the fixed copies when B is wanted."""

    def __init__(self, config: dict[str, Any], copies: bool) -> None:
        self.photos, self.ids = photo_cache(config, 224, False, config["wholeimage"]["workers"]), photo_ids(config)
        folder = output_root() / config["partition_order"]["copy_cache"]
        self.copies = build_copies(self.photos, self.ids, folder, config) if copies else None


def photo_data(config: dict[str, Any], units: pd.DataFrame, r: int, role: str, condition: str,
               images: Images) -> torch.utils.data.Dataset:
    units = with_role(units, r, role)
    if condition == "B":
        return CopyDataset(units, images.copies, images.ids, r, role, config["classes"])
    return PhotoDataset(units, images.photos, images.ids, r, role, 224, config["classes"], augmented=role == "train")


# ---------------------------------------------------------------- training loops

def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class CopyStore:
    """Condition B's bags are named by sample id ('photo|vK'); the version Bags asks for is ignored."""

    def __init__(self, store: Any) -> None:
        self.store, self.dim = store, store.dim

    def bag(self, sample: str, version: int) -> np.ndarray:
        photo, _, copy_ = sample.rpartition("|v")
        return self.store.bag(photo, int(copy_))


def mil_bags(store: Any, units: pd.DataFrame, r: int, role: str, condition: str, config: dict[str, Any],
             params: dict[str, Any], seed: int) -> tm.Bags:
    """The standard bags of ``train_mil``. B serves its fixed copies; the other conditions pick a random cached
    version of each training photo and read version 0 for scoring, exactly as in A."""
    mil, train = config["mil"], role == "train"
    units = with_role(units, r, role)
    if condition == "B":
        store, units = CopyStore(store), units.assign(photo_id=units["sample_id"])
    versions = mil["versions"] if train and condition != "B" else 1
    return tm.Bags(store, units, r, role, config["classes"], versions, params["tile_dropout"] if train else 0.0,
                   mil["batch"], seed)


def fit_mil(store: Any, units: pd.DataFrame, r: int, condition: str, config: dict[str, Any], seed: int,
            fixed_epochs: int | None = None, epoch_size: int | None = None,
            log: Callable[[int, dict[str, float]], None] | None = None) -> dict[str, Any]:
    """The loop of ``train_mil.fit`` (same loss, optimiser, schedule, early stopping on validation macro-F1), plus
    two options of the declared protocol: ``fixed_epochs`` (D: no validation photos, stop at A's best epoch) and
    ``epoch_size`` (B: each epoch draws this many training copies, so B takes as many steps per epoch as A)."""
    mil, classes, params = config["mil"], config["classes"], config["params"]
    seed_everything(seed, config["determinism"])
    torch.set_num_threads(mil.get("cpu_threads", 4))
    device = torch.device(mil.get("device", "cuda"))
    train = mil_bags(store, units, r, "train", condition, config, params, seed)
    val = None if fixed_epochs else mil_bags(store, units, r, "val", condition, config, params, seed)
    all_rows = train.rows
    model = MILModel(store.dim, params["width"], params["dropout"], mil["aggregator"], len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=tw.class_weights(train.labels[all_rows], len(classes)).to(device) if mil["class_weighting"] else None,
        label_smoothing=config["loss"]["label_smoothing"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"],
                                  eps=mil["adam_eps"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda e: 0.5 * (1 + math.cos(math.pi * e / mil["max_epochs"])))
    best: dict[str, Any] = {"f1": -1.0, "epoch": 0}
    for epoch in range(1, (fixed_epochs or mil["max_epochs"]) + 1):
        if epoch_size:
            train.rows = np.random.default_rng([seed, epoch]).choice(all_rows, epoch_size, replace=False)
        model.train()
        losses = []
        for tiles, mask, y, _ in train:
            logits, _ = model(tiles.to(device), mask.to(device))
            loss = loss_fn(logits, y.to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        schedule.step()
        if val is None:
            if log:
                log(epoch, {"train_loss": float(np.mean(losses))})
            best = {"f1": None, "epoch": epoch, "state": snapshot(model)}
            continue
        rows, probs, _ = tm.predict(model, val, device)
        f1 = metrics.scores(train.labels[rows], probs.argmax(1), list(range(len(classes))))["macro_f1"]
        if log:
            log(epoch, {"train_loss": float(np.mean(losses)), "val_macro_f1": f1})
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "state": snapshot(model), "rows": rows, "probs": probs}
        elif epoch - best["epoch"] >= mil["patience"]:
            break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def loader(data: torch.utils.data.Dataset, config: dict[str, Any], train: bool, epoch_size: int | None) -> Any:
    w = config["wholeimage"]
    sampler = torch.utils.data.RandomSampler(data, num_samples=epoch_size) if train and epoch_size else None
    return torch.utils.data.DataLoader(
        data, batch_size=w["batch_size"][224], shuffle=train and sampler is None, sampler=sampler,
        num_workers=w["workers"], pin_memory=torch.cuda.is_available(), drop_last=train,
        persistent_workers=w["workers"] > 0)


def fit_whole(config: dict[str, Any], units: pd.DataFrame, r: int, condition: str, images: Images, seed: int,
              fixed_epochs: int | None = None, epoch_size: int | None = None,
              log: Callable[[int, dict[str, float]], None] | None = None,
              make_model: Callable[[dict[str, Any], int], nn.Module] = tw.build_model) -> dict[str, Any]:
    """The loop of ``train_wholeimage.train`` (AdamW, warm-up then cosine, mixed precision, class-weighted loss with
    label smoothing, early stopping on validation macro-F1), with the same two options as ``fit_mil``."""
    w, classes = config["wholeimage"], config["classes"]
    device = torch.device(w.get("device", "cuda"))
    seed_everything(seed, w["determinism"])
    roles = ("train",) if fixed_epochs else ("train", "val")
    data = {role: loader(photo_data(config, units, r, role, condition, images), config, role == "train", epoch_size)
            for role in roles}
    model = make_model(config, 224).to(device)
    train = data["train"].dataset
    loss_fn = nn.CrossEntropyLoss(weight=tw.class_weights(train.labels[train.rows], len(classes)).to(device),
                                  label_smoothing=config["loss"]["label_smoothing"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=w["weight_decay"])
    steps, warmup = len(data["train"]), len(data["train"]) * w["warmup_epochs"]
    total = w["max_epochs"] * steps
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    mean = torch.tensor(config["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    best: dict[str, Any] = {"f1": -1.0, "epoch": 0}
    for epoch in range(1, (fixed_epochs or w["max_epochs"]) + 1):
        model.train()
        losses = []
        for batch, y, _ in data["train"]:
            x = (batch.to(device, non_blocking=True).float() / 255 - mean) / std
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y.to(device))
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            losses.append(loss.item())
        if not fixed_epochs:
            rows, probs = tw.predict(model, data["val"], config, device)
            f1 = metrics.scores(units["class"].to_numpy()[rows], np.array(classes)[probs.argmax(1)], classes)["macro_f1"]
            if log:
                log(epoch, {"train_loss": float(np.mean(losses)), "val_macro_f1": f1})
            if f1 > best["f1"]:
                best = {"f1": f1, "epoch": epoch, "state": snapshot(model), "rows": rows, "probs": probs}
            elif epoch - best["epoch"] >= w["patience"]:
                break
        else:
            if log:
                log(epoch, {"train_loss": float(np.mean(losses))})
            best = {"f1": None, "epoch": epoch, "state": snapshot(model)}
    del model, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


# ---------------------------------------------------------------- the runs

def predictions(units: pd.DataFrame, rows: np.ndarray, probs: np.ndarray, r: int, partition: str,
                classes: list[str]) -> pd.DataFrame:
    out = tw.prediction_table(units, rows, probs, r, partition, classes)
    out.insert(3, "sample_id", units["sample_id"].to_numpy()[rows])
    out.insert(4, "version", units["version"].to_numpy()[rows])
    return out


def run_config(config: dict[str, Any], a_record: dict[str, Any], model: str, condition: str, r: int) -> dict[str, Any]:
    """A's recorded settings with the condition's split file and seed: nothing is tuned."""
    po = config["partition_order"]
    c = evaluate.restore_size_keys(a_record["config"]) if model == "efficientnetv2s" else copy.deepcopy(a_record["config"])
    c.update(condition=condition, round=r, partition_order=po,
             run_seed=po["d_seeds"][r - 1] if condition == "D" else config["seed"],
             split_files={**c["split_files"], "folds": f"{po['split_dir']}/{FILENAMES[condition]}"})
    return c


def archive_partial(name: str) -> None:
    folder = get_default_runs_dir() / name
    if folder.exists():
        n = 1
        while folder.with_name(f"{name}_interrupted{n}").exists():
            n += 1
        folder.rename(folder.with_name(f"{name}_interrupted{n}"))


def fit_all(config: dict[str, Any]) -> None:
    require_declared(config)
    require_clean_code()
    root = get_default_runs_dir()
    if (root / PLAN).exists():
        raise RuntimeError(f"runs/{PLAN} is frozen: no run may be trained after scoring was declared")
    session_folds = pd.read_csv(REPO_ROOT / config["split_files"]["folds"], dtype=str)
    for model in config["partition_order"]["models"]:
        a_runs = reference_runs(config, model, "a")
        a_records = [read_json(root / name / "record.json") for name in a_runs]
        store = images = None
        for condition in config["partition_order"]["conditions"]:
            units = read_condition(condition)
            split_sha = evaluate.digest(REPO_ROOT / config["partition_order"]["split_dir"] / FILENAMES[condition])
            for r in ROUNDS:
                name, record = run_name(model, condition, r), a_records[r - 1]
                c = run_config(config, record, model, condition, r)
                fixed = read_json(root / a_runs[r - 1] / "metrics.json")["best_epoch"] if condition == "D" else None
                epoch = int((session_folds[round_column(r)] == "train").sum()) if condition == "B" else None
                c.update(fixed_epochs=fixed, epoch_size=epoch)
                if model == "attention":
                    c["cache_spec_sha256"] = completed_cache(c, c["mil"]["cache"])["spec_sha256"]
                    if c["cache_spec_sha256"] != record["config"]["cache_spec_sha256"]:
                        raise ValueError(f"{name}: the tile cache differs from the one A was trained on")
                if reusable_run(name, c, split_sha, ARTIFACTS[condition]):
                    continue
                archive_partial(name)
                recorder = RunRecorder(name, config=c, split_checksum=split_sha,
                                       data_version={"primary": c["datasets"]["primary"]["version"]})
                if model == "attention":
                    store = store or tm.BagStore(c["mil"]["cache"], c["params"]["descriptor"],
                                                 sorted(session_folds.photo_id), c["mil"]["versions"])
                    best = fit_mil(store, units, r, condition, c, c["run_seed"], fixed, epoch, recorder.log_epoch)
                else:
                    images = images or Images(c, "B" in config["partition_order"]["conditions"])
                    best = fit_whole(c, units, r, condition, images, c["run_seed"], fixed, epoch, recorder.log_epoch)
                torch.save(best["state"], recorder.run_dir / "model.pt")
                if "rows" in best:
                    recorder.save_predictions(predictions(units, best["rows"], best["probs"], r, "val", c["classes"]))
                recorder.save_metrics({"best_epoch": best["epoch"], "val_macro_f1": best["f1"], "partition": "val",
                                       "deliberately_flawed_protocol": True,
                                       "checkpoint_selection": "A's best epoch" if fixed else "validation"})
                recorder.finish()
                print(f"{name}: done", flush=True)


def all_runs(config: dict[str, Any]) -> list[tuple[str, str, str, int]]:
    po = config["partition_order"]
    return [(run_name(m, c, r), m, c, r) for m in po["models"] for c in po["conditions"] for r in ROUNDS]


def score_run(name: str, model: str, condition: str, r: int) -> pd.DataFrame:
    folder = get_default_runs_dir() / name
    c = read_json(folder / "record.json")["config"]
    if model == "efficientnetv2s":
        c = evaluate.restore_size_keys(c)
    units = read_condition(condition)
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if model == "attention":
        mil, params = c["mil"], c["params"]
        seed_everything(c["run_seed"], c["determinism"])
        store = tm.BagStore(mil["cache"], params["descriptor"], sorted(set(units["photo_id"])), mil["versions"])
        net = MILModel(store.dim, params["width"], params["dropout"], mil["aggregator"], len(c["classes"]))
        net.load_state_dict(state)
        device = torch.device(mil.get("device", "cuda"))
        bags = mil_bags(store, units, r, "test", condition, c, params, c["seed"])
        rows, probs, _ = tm.predict(net.to(device), bags, device)
    else:
        w = c["wholeimage"]
        seed_everything(c["run_seed"], w["determinism"])
        net, device = tw.build_model(c, 224), torch.device(w.get("device", "cuda"))
        net.load_state_dict(state)
        data = loader(photo_data(c, units, r, "test", condition, Images(c, condition == "B")), c, False, None)
        rows, probs = tw.predict(net.to(device), data, c, device)
    return predictions(units, rows, probs, r, "scored", c["classes"])


def score_all(config: dict[str, Any]) -> None:
    """Freeze every checkpoint, then score each run once. Nothing is scored while any run is still missing."""
    root, version = get_default_runs_dir(), require_clean_code()
    runs = all_runs(config)
    missing = [n for n, _, c, _ in runs if not completed_run(n, ARTIFACTS[c])]
    if missing:
        raise RuntimeError(f"{len(missing)} runs are not trained yet, e.g. {missing[0]}")
    frozen = {n: {"record_sha256": evaluate.digest(root / n / "record.json"),
                  "checkpoint_sha256": evaluate.digest(root / n / "model.pt")} for n, *_ in runs}
    if (root / PLAN).exists() and read_json(root / PLAN)["runs"] != frozen:
        raise ValueError("a run changed after the evaluation plan was frozen; stop and report to Ahsan")
    write_once(root / PLAN, {"runs": frozen})
    for name, model, condition, r in runs:
        folder = root / name
        if (folder / "score_metrics.json").exists():
            continue
        if (folder / "score_started.json").exists():
            raise RuntimeError(f"{name}: scoring started earlier and did not finish; stop and report to Ahsan")
        with (folder / "score_started.json").open("x") as out:
            json.dump({"scoring_code": version, "plan_sha256": evaluate.digest(root / PLAN)}, out)
        table = score_run(name, model, condition, r)
        table.to_csv(folder / "score_predictions.csv", index=False)
        (folder / "score_metrics.json").write_text(json.dumps(
            {"partition": "scored", "deliberately_flawed_protocol": True,
             **metrics.scores(table.true_class, table.pred_class, config["classes"])}, indent=2))
        print(f"{name}: scored", flush=True)   # scores are read by `report` only


# ---------------------------------------------------------------- inflation table

def confusion_by_photo(table: pd.DataFrame, photos: pd.Index, classes: list[str]) -> np.ndarray:
    """(photos, k*k) counts: for each photo, how many of its scored rows fall in each (true, predicted) cell."""
    k, code = len(classes), {c: i for i, c in enumerate(classes)}
    at = photos.get_indexer(table["photo_id"])
    if (at < 0).any():
        raise ValueError("scored rows of photos that are not in the reference set")
    counts = np.zeros((len(photos), k * k))
    np.add.at(counts, (at, table["true_class"].map(code).to_numpy() * k + table["pred_class"].map(code).to_numpy()), 1)
    return counts


def scores_from_confusion(counts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 (over all k classes, 0 where undefined) and balanced accuracy of confusion counts (..., k*k)."""
    c = counts.reshape(*counts.shape[:-1], k, k)
    tp, actual, predicted = np.diagonal(c, axis1=-2, axis2=-1), c.sum(-1), c.sum(-2)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros_like(tp), where=actual + predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    present = actual > 0
    return f1.mean(-1), (recall * present).sum(-1) / present.sum(-1)


def inflation_ranges(counts: dict[str, np.ndarray], photo_class: np.ndarray, settings: dict[str, Any], seed: int,
                     k: int, reference: str = "A") -> dict[str, dict[str, dict[str, float]]]:
    """Per label and metric: the pooled value with its range, and the difference to ``reference`` with its range.
    The ranges come from one set of bootstrap draws over source photos (resampled within class, all copies and
    repeats of a photo kept together), so every difference is paired."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(photo_class == c) for c in np.unique(photo_class)]
    draws = np.zeros((settings["resamples"], len(photo_class)))
    for i in range(settings["resamples"]):
        draws[i] = np.bincount(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]),
                               minlength=len(photo_class))
    point = {label: scores_from_confusion(c.sum(0), k) for label, c in counts.items()}
    boot = {label: scores_from_confusion(draws @ c, k) for label, c in counts.items()}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for label in counts:
        result[label] = {}
        for m, metric in enumerate(("macro_f1", "balanced_accuracy")):
            low, high = stats.interval(boot[label][m], settings["confidence"])
            d_low, d_high = stats.interval(boot[label][m] - boot[reference][m], settings["confidence"])
            result[label][metric] = {"value": float(point[label][m]), "low": low, "high": high,
                                     "difference": float(point[label][m] - point[reference][m]),
                                     "difference_low": d_low, "difference_high": d_high}
    return result


def round_scores(table: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Macro-F1 and balanced accuracy of each round (or repeat), indexed by round."""
    return pd.DataFrame({int(r): {m: s[m] for m in ("macro_f1", "balanced_accuracy")}
                         for r, g in table.groupby("round")
                         for s in [metrics.scores(g["true_class"], g["pred_class"], classes)]}).T


def reading(difference_low: float, difference_high: float, above: int, below: int, minimum: int) -> str:
    """The declared decision rule for 'higher than A': the range excludes 0 and at least
    ``minimum`` rounds agree. Otherwise there is no clear difference."""
    if difference_low > 0 and above >= minimum:
        return "clearly above A"
    if difference_high < 0 and below >= minimum:
        return "clearly below A"
    return "no clear difference"


def leak_shares(units: pd.DataFrame, keys: pd.DataFrame, scored_where: np.ndarray | None = None) -> dict[str, float]:
    """The leak counter averaged over the five rounds, weighted by the number of scored units."""
    rows = [leak_counter(units, r, keys, scored_where) for r in ROUNDS]
    weight = np.array([row["scored_units"] for row in rows], float)
    return {key: float(np.average([row[key] for row in rows], weights=weight)) for key in rows[0] if key != "scored_units"}


def read_scored(names: list[str], filename: str) -> pd.DataFrame:
    root = get_default_runs_dir()
    return pd.concat([pd.read_csv(root / n / filename) for n in names], ignore_index=True)


def inflation_table(config: dict[str, Any], tables: dict[str, pd.DataFrame], leaks: dict[str, dict[str, float]],
                    model: str) -> pd.DataFrame:
    """One row per condition for one model; ``tables`` maps a label to its scored predictions (A first)."""
    classes, boot = config["classes"], config["evaluation"]["bootstrap"]
    minimum = config["evaluation"]["comparison"]["min_round_wins"]
    photos = pd.Index(sorted(tables["A"]["photo_id"].unique()))
    counts = {label: confusion_by_photo(t, photos, classes) for label, t in tables.items()}
    for label, c in counts.items():
        if (c.sum(1) == 0).any():
            raise ValueError(f"{label}: some photos have no scored row; the table needs every photo in every condition")
    photo_class = tables["A"].drop_duplicates("photo_id").set_index("photo_id").loc[photos, "true_class"]
    ranges = inflation_ranges(counts, photo_class.map({c: i for i, c in enumerate(classes)}).to_numpy(), boot,
                              config["seed"], len(classes))
    per_round = {label: round_scores(t, classes) for label, t in tables.items()}
    rows = []
    for label, t in tables.items():
        row: dict[str, Any] = {"model": model, "condition": label, "meaning": MEANING[label], "scored_rows": len(t),
                               "photos": t["photo_id"].nunique(), "rounds": t["round"].nunique()}
        for metric in ("macro_f1", "balanced_accuracy"):
            mine, ref = per_round[label][metric], per_round["A"][metric]
            above, below = int((mine > ref.reindex(mine.index)).sum()), int((mine < ref.reindex(mine.index)).sum())
            r = ranges[label][metric]
            row.update({f"{metric}_pooled": r["value"], f"{metric}_low": r["low"], f"{metric}_high": r["high"],
                        f"{metric}_round_mean": mine.mean(), f"{metric}_round_sd": mine.std(ddof=1),
                        f"{metric}_inflation": r["difference"], f"{metric}_inflation_low": r["difference_low"],
                        f"{metric}_inflation_high": r["difference_high"], f"{metric}_rounds_above_A": above,
                        f"{metric}_reading": "reference" if label == "A" else
                        reading(r["difference_low"], r["difference_high"], above, below, minimum)})
        rows.append({**row, **{f"leak_{k}": v for k, v in leaks[label].items()}})
    return pd.DataFrame(rows)


def report(config: dict[str, Any]) -> None:
    root = get_default_runs_dir()
    missing = [n for n, *_ in all_runs(config) if not (root / n / "score_metrics.json").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} runs are not scored yet, e.g. {missing[0]}")
    session = pd.read_csv(REPO_ROOT / config["split_files"]["folds"], dtype=str)
    locked = pd.read_csv(REPO_ROOT / "splits/folds.csv", dtype=str)
    keys = photo_keys(session, locked, read_inventory(config["inventory"]["files"]["primary"]))
    b = read_condition("B")
    leaks = {"A": leak_shares(session, keys), "L": leak_shares(locked, keys), "B": leak_shares(b, keys),
             "B_clean": leak_shares(b, keys, (b["version"] == 0).to_numpy()),
             "C": leak_shares(read_condition("C"), keys), "D": leak_shares(read_condition("D"), keys)}
    frames = []
    for model in config["partition_order"]["models"]:
        scored = {c: read_scored([run_name(model, c, r) for r in ROUNDS], "score_predictions.csv") for c in "BCD"}
        tables = {"A": read_scored(reference_runs(config, model, "a"), "test_predictions.csv"),
                  "L": read_scored(reference_runs(config, model, "l"), "test_predictions.csv"),
                  "B": scored["B"], "B_clean": scored["B"][scored["B"]["version"] == 0],
                  "C": scored["C"], "D": scored["D"]}
        frames.append(inflation_table(config, tables, leaks, model))
    table = pd.concat(frames, ignore_index=True)
    (REPO_ROOT / TABLE).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPO_ROOT / TABLE, index=False)
    (REPO_ROOT / REPORT).write_text(report_text(table))
    print(f"report: {len(table)} rows", flush=True)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def report_text(table: pd.DataFrame) -> str:
    """The neutral report, built from the table only."""
    sections = []
    for model, title in (("efficientnetv2s", "EfficientNetV2-S, whole photo at 224 px"),
                         ("attention", "Main tile model (E1 tiles + gated-attention head)")):
        part = table[table["model"] == model]
        shown = pd.DataFrame({
            "condition": part["condition"], "what it does": part["meaning"],
            "scored rows": part["scored_rows"].map("{:,}".format),
            "macro-F1, pooled [95% range]": [f"{pct(v)} [{pct(lo)}, {pct(hi)}]" for v, lo, hi in
                                             zip(part.macro_f1_pooled, part.macro_f1_low, part.macro_f1_high)],
            "mean ± SD over rounds": [f"{pct(m)} ± {pct(s)}" for m, s in zip(part.macro_f1_round_mean, part.macro_f1_round_sd)],
            "inflation vs A (points) [95% range]": ["–" if c == "A" else f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                                                    for c, d, lo, hi in zip(part.condition, part.macro_f1_inflation,
                                                                            part.macro_f1_inflation_low, part.macro_f1_inflation_high)],
            "rounds above A": ["–" if c == "A" else f"{n}/{r}" for c, n, r in zip(part.condition, part.macro_f1_rounds_above_A, part.rounds)],
            "reading": part["macro_f1_reading"]})
        leak = pd.DataFrame({
            "condition": part["condition"], "same source photo": part["leak_source_photo"].map(pct),
            "same near-duplicate group": part["leak_near_duplicate_group"].map(pct),
            "same capture block": part["leak_capture_block"].map(pct),
            "same camera within 10 s": part["leak_same_camera_within_10s"].map(pct)})
        balanced = pd.DataFrame({
            "condition": part["condition"], "balanced accuracy, pooled": part["balanced_accuracy_pooled"].map(pct),
            "inflation vs A (points) [95% range]": ["–" if c == "A" else f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                                                    for c, d, lo, hi in zip(part.condition, part.balanced_accuracy_inflation,
                                                                            part.balanced_accuracy_inflation_low, part.balanced_accuracy_inflation_high)],
            "reading": part["balanced_accuracy_reading"]})
        sections += [f"## {title}", "", "Macro-F1 (percent, test photos of each condition):", "", md_table(shown), "",
                     "Balanced accuracy:", "", md_table(balanced), "",
                     "Leak counter: share of the scored rows (all five rounds) whose photo, group, block or neighbour in time "
                     "is also in training (percent):", "", md_table(leak), ""]
    return "\n".join([
        "# Partition-order experiment", "",
        "Built by `src/partition_order.py` from `runs/`; do not edit by hand. Declared on 2026-11-20 (handbook §24).", "",
        "**What this is.** A general lesson about testing practice. Testing mistakes are common across plant-disease "
        "recognition work (Xu et al., 2024). Here we repeat three of them on our own photos and models and measure how "
        "much each raises the score, compared with the correct protocol. No published study is assessed.", "",
        "**Conditions.** A is the correct protocol: the session-blocked split, augmentation on training photos only (the "
        "five headline runs of each model, reused as they are). B makes eight copies of every photo with the same "
        "augmentation, then splits the copies at random, so copies of one photo land on both sides. C splits photos at "
        "random and ignores near-duplicate groups and capture blocks. D trains on every photo and scores on every photo. "
        "L is a reference row: the locked split, which groups pixel-level copies but not capture sessions, with the "
        "settings selected on it (not A's).", "",
        "**How to read it.** B, C and D are deliberately wrong. Their scores are not estimates of how the models "
        "perform on new photos, and they are used nowhere else. All of them reuse A's selected settings: nothing is "
        "tuned. B takes as many training steps per epoch as A. B is scored on its test copies, which include augmented "
        "copies; `B_clean` keeps only the clean copy of each photo. D stops at A's best epoch and its five repeats use "
        "five seeds on the same photos. Inflation is the pooled score of a condition minus the pooled score of A. Its "
        "range is a bootstrap over source photos (resampled within class; all copies and repeats of a photo stay "
        "together). A condition is called higher than A only when the range excludes 0 and at least 4 of 5 rounds agree.",
        "One training seed per round and condition; the rounds are the repeats.", ""] + sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["splits", "declare", "cache", "fit", "score", "report"])
    parser.add_argument("--config", default="configs/partition_order.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "splits":
        session = pd.read_csv(REPO_ROOT / config["split_files"]["folds"], dtype=str)
        lock_tables(condition_tables(session, config), REPO_ROOT / config["partition_order"]["split_dir"])
        keys = photo_keys(session, pd.read_csv(REPO_ROOT / "splits/folds.csv", dtype=str),
                          read_inventory(config["inventory"]["files"]["primary"]))
        for condition in "BCD":
            print(condition, leak_shares(read_condition(condition), keys), flush=True)
        print("A", leak_shares(session, keys), flush=True)
    elif args.action == "cache":
        Images(config, copies=True)
    else:
        {"declare": declare, "fit": fit_all, "score": score_all, "report": report}[args.action](config)
    print(f"STAGE9 {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
