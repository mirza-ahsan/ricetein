"""Stage 9: how much do testing mistakes inflate scores? (declared 2026-11-20)

Condition A is the correct protocol: the five existing session-blocked runs of each model, reused as they are.
B, C and D are DELIBERATELY WRONG demonstrations (src/partition_splits.py): B augments first and splits the copies
at random, C ignores near-duplicate groups and capture blocks, D trains and scores on every photo. They reuse
A's selected settings (no tuning), have their own split files, and write only runs named ``stage9_*``. Their
numbers feed no tuning, selection or other table.

    uv run python -m src.partition_order <action>

splits    write splits/partition_order/ once (no GPU; the leak counter is `python -m src.partition_report leaks`)
declare   freeze runs/stage9_declared.json: the settings, the split files and the A runs (once; never deleted)
cache     build the 8 fixed augmented copies of every photo that condition B needs (CPU)
fit       train the 30 runs (2 models x B, C, D x 5); resumable; refuses once scoring has been frozen
score     freeze runs/stage9_evaluation_plan.json, then score each run once (score_started.json is never deleted)

The inflation table and the report are built by ``python -m src.partition_report`` (a separate module, so that a fix to
the report text never makes a finished run look changed; ``records.code_paths`` hashes this module and what it imports).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import augment, metrics
from src import train_mil as tm
from src import train_wholeimage as tw
from src.config import REPO_ROOT, load_config, output_root
from src.embed import completed_cache
from src.mil_models import MILModel
from src.partition_splits import BOTH, FILENAMES, SPLIT_DIR, condition_tables, lock_tables, read_condition
from src.photo_dataset import PhotoDataset, photo_cache, photo_ids
from src.records import RunRecorder, completed_run, get_default_runs_dir, require_clean_code, reusable_run
from src.seeding import seed_everything
from src.split import read_folds, round_column, sha256

ROUNDS = range(1, 6)
DECLARED, PLAN = "stage9_declared.json", "stage9_evaluation_plan.json"
FULL = ("model.pt", "metrics.json", "predictions.csv")
ARTIFACTS = {"B": FULL, "C": FULL, "D": FULL[:2]}   # D has no validation photos: it trains for A's best epoch
TRAINING_KEYS = ("versions", "split_seed_offset", "d_seeds", "copy_cache")   # the settings that shape a run


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_once(path: Path, data: Any) -> None:
    """Create the file with ``data``, all or nothing; an existing file must hold the same data."""
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if not path.exists():
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        try:
            os.link(temporary, path)          # fails if another call created the file first
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    if read_json(path) != json.loads(text):
        raise ValueError(f"{path} exists with other content; it is frozen")


def int_size_keys(config: dict[str, Any]) -> dict[str, Any]:
    """A copy of a recorded whole-image config whose size tables are keyed by integers again (JSON turns them
    into text; a Swin-style ``size_kwargs`` lookup by integer would silently find nothing)."""
    config = copy.deepcopy(config)
    whole = config["wholeimage"]
    whole["batch_size"] = {int(size): batch for size, batch in whole["batch_size"].items()}
    for spec in whole["models"].values():
        if "size_kwargs" in spec:
            spec["size_kwargs"] = {int(size): kwargs for size, kwargs in spec["size_kwargs"].items()}
    return config


def run_name(model: str, condition: str, r: int) -> str:
    return f"stage9_{model}_{condition}_r{r}"


def reference_runs(config: dict[str, Any], model: str, which: str) -> list[str]:
    """The five runs of ``model`` in condition A (``which="a"``) or the locked-split reference L (``"l"``)."""
    po = config["partition_order"]
    selection = read_json(get_default_runs_dir() / po[f"{which}_selection"])
    return selection["methods"][po[f"{which}_methods"][model]]["runs"]


def with_role(units: pd.DataFrame, r: int, role: str) -> pd.DataFrame:
    """Condition D's ``train_and_test`` rows act as ``role`` (train or test only); the others are unchanged."""
    column = round_column(r)
    if role not in ("train", "test") and (units[column] == BOTH).any():
        raise ValueError(f"condition D has no {role} photos")
    return units.assign(**{column: units[column].replace(BOTH, role)})


# ---------------------------------------------------------------- declaration

def split_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {c: sha256(REPO_ROOT / SPLIT_DIR / f) for c, f in FILENAMES.items()}


def protocol(config: dict[str, Any]) -> dict[str, Any]:
    """What the declaration freezes: the settings, the three split files, and the A and L runs by file hash."""
    root, runs = get_default_runs_dir(), {}
    for which in ("a", "l"):
        for model in config["partition_order"]["models"]:
            for name in reference_runs(config, model, which):
                runs[name] = {"record_sha256": sha256(root / name / "record.json"),
                              "test_predictions_sha256": sha256(root / name / "test_predictions.csv")}
                if which == "a":   # condition D trains for A's best epoch, read from metrics.json
                    runs[name]["metrics_sha256"] = sha256(root / name / "metrics.json")
    return {"partition_order": config["partition_order"], "split_files": split_hashes(config),
            "session_folds_sha256": read_folds(config)[1],
            "seed": config["seed"], "evaluation": config["evaluation"], "reference_runs": runs}


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
    shape = (config["partition_order"]["versions"], len(ids), 224, 224, 3)
    identity = {"ids": ids, "shape": list(shape), "seed": config["seed"],
                "augment_sha256": sha256(REPO_ROOT / "src/augment.py"),
                "photos": sha256(REPO_ROOT / "src/photo_dataset.py")}
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
    if data.stat().st_size != math.prod(shape):
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
        self.copies = None
        if copies:
            folder = output_root() / config["partition_order"]["copy_cache"]
            self.copies = build_copies(self.photos, self.ids, folder, config)


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
    """Condition B's bags are named by sample id; Bags asks for ``bag(sample_id, 0)`` and gets that copy's bag."""

    def __init__(self, store: Any, units: pd.DataFrame) -> None:
        self.store, self.dim = store, store.dim
        self.copy_of = dict(zip(units["sample_id"], zip(units["photo_id"], units["version"].astype(int))))

    def bag(self, sample: str, version: int) -> np.ndarray:
        assert version == 0, "condition B reads one fixed copy per sample id"
        return self.store.bag(*self.copy_of[sample])


def mil_bags(store: Any, units: pd.DataFrame, r: int, role: str, condition: str, config: dict[str, Any],
             params: dict[str, Any], seed: int) -> tm.Bags:
    """The standard bags of ``train_mil``. B serves its fixed copies; the other conditions pick a random cached
    version of each training photo and read version 0 for scoring, exactly as in A."""
    mil, train = config["mil"], role == "train"
    units = with_role(units, r, role)
    if condition == "B":
        store, units = CopyStore(store, units), units.assign(photo_id=units["sample_id"])
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
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda e: 0.5 * (1 + math.cos(math.pi * e / mil["max_epochs"])))
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
        entry = {"train_loss": float(np.mean(losses))}
        if fixed_epochs:
            if log:
                log(epoch, entry)
            continue
        rows, probs, _ = tm.predict(model, val, device)
        f1 = metrics.scores(train.labels[rows], probs.argmax(1), list(range(len(classes))))["macro_f1"]
        if log:
            log(epoch, {**entry, "val_macro_f1": f1})
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "state": snapshot(model), "rows": rows, "probs": probs}
        elif epoch - best["epoch"] >= mil["patience"]:
            break
    if fixed_epochs:
        best = {"f1": None, "epoch": fixed_epochs, "state": snapshot(model)}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def loader(data: torch.utils.data.Dataset, config: dict[str, Any], train: bool, epoch_size: int | None,
           persistent: bool = True) -> Any:
    """One pass per epoch for training and validation (workers stay alive); scoring passes ``persistent=False``."""
    w = config["wholeimage"]
    sampler = torch.utils.data.RandomSampler(data, num_samples=epoch_size) if train and epoch_size else None
    return torch.utils.data.DataLoader(
        data, batch_size=w["batch_size"][224], shuffle=train and sampler is None, sampler=sampler,
        num_workers=w["workers"], pin_memory=torch.cuda.is_available(), drop_last=train,
        persistent_workers=persistent and w["workers"] > 0)


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
        optimiser, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
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
        entry = {"train_loss": float(np.mean(losses))}
        if fixed_epochs:
            if log:
                log(epoch, entry)
            continue
        rows, probs = tw.predict(model, data["val"], config, device)
        true = units["class"].to_numpy()[rows]
        f1 = metrics.scores(true, np.array(classes)[probs.argmax(1)], classes)["macro_f1"]
        if log:
            log(epoch, {**entry, "val_macro_f1": f1})
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "state": snapshot(model), "rows": rows, "probs": probs}
        elif epoch - best["epoch"] >= w["patience"]:
            break
    if fixed_epochs:
        best = {"f1": None, "epoch": fixed_epochs, "state": snapshot(model)}
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
    c = int_size_keys(a_record["config"]) if model == "efficientnetv2s" else copy.deepcopy(a_record["config"])
    c.update(condition=condition, round=r, partition_order={k: po[k] for k in TRAINING_KEYS},
             run_seed=po["d_seeds"][r - 1] if condition == "D" else config["seed"],
             split_files={**c["split_files"], "folds": f"{SPLIT_DIR}/{FILENAMES[condition]}"})
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
    root, po = get_default_runs_dir(), config["partition_order"]
    if (root / PLAN).exists():
        raise RuntimeError(f"runs/{PLAN} is frozen: no run may be trained after scoring was declared")
    session_folds, session_sha = read_folds(config)
    for model in po["models"]:
        a_runs = reference_runs(config, model, "a")
        a_records = [read_json(root / name / "record.json") for name in a_runs]
        for r, record in enumerate(a_records, start=1):
            if record["config"]["round"] != r or record["split_checksum"] != session_sha:
                raise ValueError(f"{a_runs[r - 1]} is not round {r} of the declared session split")
        if model == "attention" and a_records[0]["config"]["mil"]["versions"] != po["versions"]:
            raise ValueError("partition_order.versions must equal the number of cached versions")
        store = images = None
        for condition in po["conditions"]:
            units = read_condition(condition)
            split_sha = sha256(REPO_ROOT / SPLIT_DIR / FILENAMES[condition])
            for r in ROUNDS:
                name, record = run_name(model, condition, r), a_records[r - 1]
                c = run_config(config, record, model, condition, r)
                fixed = read_json(root / a_runs[r - 1] / "metrics.json")["best_epoch"] if condition == "D" else None
                epoch_size = int((session_folds[round_column(r)] == "train").sum()) if condition == "B" else None
                c.update(fixed_epochs=fixed, epoch_size=epoch_size)
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
                    best = fit_mil(store, units, r, condition, c, c["run_seed"], fixed, epoch_size, recorder.log_epoch)
                else:
                    images = images or Images(c, "B" in po["conditions"])
                    best = fit_whole(c, units, r, condition, images, c["run_seed"], fixed, epoch_size,
                                     recorder.log_epoch)
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


def score_run(name: str, model: str, condition: str, r: int) -> Callable[[], pd.DataFrame]:
    """Everything that can fail for lack of a file, a cache or GPU memory happens here, before the started marker is
    written; the returned function only runs the model on the scored photos."""
    folder = get_default_runs_dir() / name
    c = read_json(folder / "record.json")["config"]
    units = read_condition(condition)
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if model == "attention":
        mil, params = c["mil"], c["params"]
        seed_everything(c["run_seed"], c["determinism"])
        store = tm.BagStore(mil["cache"], params["descriptor"], sorted(set(units["photo_id"])), mil["versions"])
        net = MILModel(store.dim, params["width"], params["dropout"], mil["aggregator"], len(c["classes"]))
        net.load_state_dict(state)
        device = torch.device(mil.get("device", "cuda"))
        net, bags = net.to(device), mil_bags(store, units, r, "test", condition, c, params, c["seed"])

        def run() -> pd.DataFrame:
            rows, probs, _ = tm.predict(net, bags, device)
            return predictions(units, rows, probs, r, "scored", c["classes"])
    else:
        c = int_size_keys(c)
        w = c["wholeimage"]
        seed_everything(c["run_seed"], w["determinism"])
        net, device = tw.build_model(c, 224), torch.device(w.get("device", "cuda"))
        net.load_state_dict(state)
        net = net.to(device).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            net(torch.zeros(w["batch_size"][224], 3, 224, 224, device=device))   # a full batch: out-of-memory shows here
        data = loader(photo_data(c, units, r, "test", condition, Images(c, condition == "B")), c, False, None,
                      persistent=False)

        def run() -> pd.DataFrame:
            rows, probs = tw.predict(net, data, c, device)
            return predictions(units, rows, probs, r, "scored", c["classes"])
    return run


def score_all(config: dict[str, Any]) -> None:
    """Freeze every checkpoint and split file, then score each run once. Nothing is scored while any run is missing."""
    require_declared(config)
    root, version = get_default_runs_dir(), require_clean_code()
    runs = all_runs(config)
    missing = [n for n, _, c, _ in runs if not completed_run(n, ARTIFACTS[c])]
    if missing:
        raise RuntimeError(f"{len(missing)} runs are not trained yet, e.g. {missing[0]}")
    splits = split_hashes(config)
    for n, _, condition, _ in runs:
        if read_json(root / n / "record.json")["split_checksum"] != splits[condition]:
            raise ValueError(f"{n}: trained on another split file than the declared one")
    frozen = {"split_files": splits,
              "runs": {n: {"record_sha256": sha256(root / n / "record.json"),
                           "checkpoint_sha256": sha256(root / n / "model.pt")} for n, *_ in runs}}
    if (root / PLAN).exists() and read_json(root / PLAN) != frozen:
        raise ValueError("a run or a split file changed after the evaluation plan was frozen; stop and report to Ahsan")
    write_once(root / PLAN, frozen)
    plan_sha = sha256(root / PLAN)
    for name, model, condition, r in runs:
        folder = root / name
        if (folder / "score_metrics.json").exists():
            continue
        if (folder / "score_started.json").exists():
            raise RuntimeError(f"{name}: scoring started earlier and did not finish; stop and report to Ahsan")
        run = score_run(name, model, condition, r)
        with (folder / "score_started.json").open("x") as out:
            json.dump({"scoring_code": version, "plan_sha256": plan_sha}, out)
        table = run()
        del run                      # frees the model and its loader before the next run is prepared
        table.to_csv(folder / "score_predictions.csv", index=False)
        (folder / "score_metrics.json").write_text(json.dumps(
            {"partition": "scored", "deliberately_flawed_protocol": True,
             **metrics.scores(table.true_class, table.pred_class, config["classes"])}, indent=2))
        print(f"{name}: scored", flush=True)   # scores are read by `partition_report` only


def make_splits(config: dict[str, Any]) -> None:
    session, _ = read_folds(config)
    lock_tables(condition_tables(session, config), REPO_ROOT / SPLIT_DIR)
    print("written:", split_hashes(config), flush=True)   # the leak counter: python -m src.partition_report leaks


def build_cache(config: dict[str, Any]) -> None:
    Images(config, copies=True)


ACTIONS = {"splits": make_splits, "declare": declare, "cache": build_cache, "fit": fit_all, "score": score_all}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--config", default="configs/partition_order.yaml")
    args = parser.parse_args()
    ACTIONS[args.action](load_config(args.config))
    print(f"STAGE9 {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
