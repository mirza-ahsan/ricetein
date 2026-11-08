"""Stage 7: masked MIL training, persistent equal-budget tuning and three-seed final runs.

    uv run python -m src.train_mil --config configs/mil/attention.yaml tune
    uv run python -m src.train_mil --config configs/mil/attention.yaml final

Only validation scores are used here. Test evaluation is a separate, frozen-selection step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import metrics
from src.config import cache_folder, load_config
from src.embed import completed_cache, photo_key
from src.mil_models import MILModel
from src.records import RunRecorder, get_default_runs_dir, require_clean_code, reusable_run
from src.seeding import seed_everything
from src.split import read_folds, round_column
from src.train_wholeimage import class_weights, prediction_table


class BagStore:
    """Read descriptors photo by photo through memory maps, never loading an entire cache into RAM."""

    def __init__(self, cache: str, descriptor: str, photo_ids: list[str], versions: int) -> None:
        self.parts = {"c": ["cls"], "m": ["mean"], "cm": ["cls", "mean"]}[descriptor]
        folder = cache_folder(cache)
        complete = json.loads((folder / "complete.json").read_text())
        if complete["versions"] < versions:
            raise ValueError(f"{cache}: requires {versions} versions")
        packed = cache_folder(f"packed/{cache}", write=True)
        packed.mkdir(parents=True, exist_ok=True)
        self.arrays, self.index = {}, None
        for key in self.parts:
            array, index = pack(folder, packed, key, photo_ids, versions, complete["spec_sha256"])
            if self.index is not None and index != self.index:
                raise ValueError("class-token and patch-mean bag indices differ")
            self.arrays[key], self.index = array, index
        self.dim = sum(a.shape[1] for a in self.arrays.values())

    def bag(self, photo_id: str, version: int) -> np.ndarray:
        start, stop = self.index[f"{photo_id}|{version}"]
        return np.concatenate([self.arrays[k][start:stop] for k in self.parts], axis=1)


def pack(
    folder: Path, target: Path, key: str, photo_ids: list[str], versions: int, fingerprint: str
) -> tuple[np.memmap, dict]:
    identity = {"photos": photo_ids, "versions": versions, "cache_spec": fingerprint, "key": key}
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    data, meta_path = target / f"{key}_{tag}.f16", target / f"{key}_{tag}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["identity"] != identity or data.stat().st_size != math.prod(meta["shape"]) * 2:
            raise ValueError(f"{meta_path}: damaged packed cache; archive it before rebuilding")
    else:
        index, rows, dim = {}, 0, None
        temporary = data.with_suffix(".tmp")
        with temporary.open("wb") as out:
            for version in range(versions):
                for photo_id in photo_ids:
                    with np.load(folder / f"v{version}" / f"{photo_key(photo_id)}.npz") as saved:
                        values = saved[key]
                        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
                            raise ValueError(f"{photo_id} version {version}: invalid {key} descriptors")
                        if dim is not None and values.shape[1] != dim:
                            raise ValueError("descriptor dimensions differ")
                        values = values.astype(np.float16)
                    index[f"{photo_id}|{version}"] = (rows, rows + len(values))
                    rows, dim = rows + len(values), values.shape[1]
                    out.write(values.tobytes())
        meta = {"identity": identity, "shape": [rows, dim], "index": index}
        os.replace(temporary, data)
        temporary_meta = meta_path.with_suffix(".tmp")
        temporary_meta.write_text(json.dumps(meta))
        os.replace(temporary_meta, meta_path)
    return np.memmap(data, np.float16, "r", shape=tuple(meta["shape"])), meta["index"]


class Bags:
    """One role in one round. Validation/test refuse augmented versions and tile dropout."""

    def __init__(
        self,
        store: BagStore,
        folds: pd.DataFrame,
        round_: int,
        role: str,
        classes: list[str],
        versions: int,
        tile_dropout: float,
        batch: int,
        seed: int,
    ) -> None:
        if role not in ("train", "val", "test"):
            raise ValueError(f"unknown split role {role}")
        if role != "train" and (versions != 1 or tile_dropout != 0):
            raise ValueError(f"{role} photos require version 0 with all tiles")
        if versions < 1 or batch < 1 or not 0 <= tile_dropout <= 1:
            raise ValueError("invalid versions, batch size or tile dropout")
        self.rows = np.flatnonzero((folds[round_column(round_)] == role).to_numpy())
        if not len(self.rows):
            raise ValueError(f"round {round_} has no {role} photos")
        self.photo_ids = folds["photo_id"].to_numpy()
        self.labels = folds["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        self.store, self.role, self.versions = store, role, versions
        self.tile_dropout, self.batch = tile_dropout, batch
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = self.rng.permutation(self.rows) if self.role == "train" else self.rows
        for start in range(0, len(order), self.batch):
            rows = order[start : start + self.batch]
            bags = [self.store.bag(self.photo_ids[r], int(self.rng.integers(self.versions))) for r in rows]
            if self.tile_dropout > 0:
                kept = []
                for bag in bags:
                    mask = self.rng.random(len(bag)) >= self.tile_dropout
                    if not mask.any():
                        mask[self.rng.integers(len(bag))] = True
                    kept.append(bag[mask])
                bags = kept
            size = max(len(b) for b in bags)
            tiles = np.zeros((len(bags), size, self.store.dim), np.float32)
            mask = np.zeros((len(bags), size), bool)
            for i, bag in enumerate(bags):
                tiles[i, : len(bag)], mask[i, : len(bag)] = bag, True
            yield torch.from_numpy(tiles), torch.from_numpy(mask), torch.from_numpy(self.labels[rows]), rows


@torch.inference_mode()
def predict(model: MILModel, bags: Bags, device: torch.device) -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    rows, probs, weights = [], [], []
    for tiles, mask, _, r in bags:
        logits, attention = model(tiles.to(device), mask.to(device))
        rows.append(r)
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        n = mask.sum(1).tolist()
        weights += (
            [None] * len(r) if attention is None else [a[:k] for a, k in zip(attention.cpu().numpy(), n)]
        )
    return np.concatenate(rows), np.concatenate(probs), weights


def fit(
    store: BagStore,
    folds: pd.DataFrame,
    round_: int,
    params: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    log=None,
) -> dict[str, Any]:
    mil, classes = config["mil"], config["classes"]
    seed_everything(seed, config["determinism"])
    torch.set_num_threads(mil.get("cpu_threads", 4))
    device = torch.device(mil.get("device", "cuda"))
    train = Bags(
        store, folds, round_, "train", classes, mil["versions"], params["tile_dropout"], mil["batch"], seed
    )
    val = Bags(store, folds, round_, "val", classes, 1, 0.0, mil["batch"], seed)
    model = MILModel(store.dim, params["width"], params["dropout"], mil["aggregator"], len(classes)).to(
        device
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(train.labels[train.rows], len(classes)).to(device)
        if mil["class_weighting"]
        else None,
        label_smoothing=config["loss"]["label_smoothing"],
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], eps=mil["adam_eps"]
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda e: 0.5 * (1 + math.cos(math.pi * e / mil["max_epochs"]))
    )
    best = {"f1": -1.0}
    for epoch in range(1, mil["max_epochs"] + 1):
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
        rows, probs, weights = predict(model, val, device)
        f1 = metrics.scores(train.labels[rows], probs.argmax(1), list(range(len(classes))))["macro_f1"]
        if log:
            log(epoch, {"train_loss": float(np.mean(losses)), "val_macro_f1": f1})
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "rows": rows,
                "probs": probs,
                "weights": weights,
            }
        elif epoch - best["epoch"] >= mil["patience"]:
            break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def store_for(config: dict[str, Any], folds: pd.DataFrame, descriptor: str) -> BagStore:
    return BagStore(config["mil"]["cache"], descriptor, sorted(folds.photo_id), config["mil"]["versions"])


def tune(config: dict[str, Any]) -> dict[str, Any]:
    """Persist trial parameters and five fold scores; a completed fold is never repeated on resume."""
    import optuna

    version = require_clean_code()
    config = {**config, "cache_spec_sha256": completed_cache(config, config["mil"]["cache"])["spec_sha256"]}
    mil, space = config["mil"], config["search"]
    folds, checksum = read_folds(config)
    name = f"stage7_tune_{mil['name']}"
    folder = get_default_runs_dir() / name
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "split_checksum": checksum, "code_version": version}
    identity_path = folder / "study_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(f"{name}: cannot resume a study with different code, settings or split")
    identity_path.write_text(json.dumps(identity, indent=2))
    if reusable_run(name, config, checksum, ("metrics.json",)):
        return json.loads((folder / "metrics.json").read_text())
    sampler_path = folder / "sampler.pkl"
    if sampler_path.exists():
        with sampler_path.open("rb") as handle:
            sampler = pickle.load(handle)
    else:
        sampler = optuna.samplers.TPESampler(seed=space["seed"])
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{folder / 'study.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    # Recover the exact interrupted parameter set. Its cached fold results remain usable.
    recovered = {t.user_attrs.get("recovered_from") for t in study.trials}
    for trial in study.trials:
        running = trial.state == optuna.trial.TrialState.RUNNING
        if running:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        if (
            (running or trial.state == optuna.trial.TrialState.FAIL)
            and trial.user_attrs.get("parameters_complete")
            and trial.number not in recovered
        ):
            study.enqueue_trial(trial.params, user_attrs={"recovered_from": trial.number})
    stores: dict[str, BagStore] = {}
    recorder = RunRecorder(
        name + "_record",
        config=config,
        split_checksum=checksum,
        data_version={"primary": config["datasets"]["primary"]["version"]},
        overwrite=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", *space["lr"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
            **{
                key: trial.suggest_categorical(key, space[key])
                for key in ("width", "dropout", "tile_dropout", "descriptor")
            },
        }
        trial.set_user_attr("parameters_complete", True)
        with sampler_path.with_suffix(".tmp").open("wb") as handle:
            pickle.dump(study.sampler, handle)
        os.replace(sampler_path.with_suffix(".tmp"), sampler_path)
        if params["descriptor"] not in stores:
            stores[params["descriptor"]] = store_for(config, folds, params["descriptor"])
        tag = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        scores = []
        for r in range(1, config["split"]["n_folds"] + 1):
            path = folder / f"trial_{tag}_r{r}.json"
            if path.exists():
                f1 = json.loads(path.read_text())["val_macro_f1"]
            else:
                best = fit(stores[params["descriptor"]], folds, r, params, config, config["seed"])
                f1 = best["f1"]
                path.write_text(
                    json.dumps(
                        {"params": params, "round": r, "val_macro_f1": f1, "best_epoch": best["epoch"]}
                    )
                )
            scores.append(f1)
            trial.set_user_attr(f"val_macro_f1_r{r}", f1)
            print(f"{name} trial {trial.number} round {r}: validation F1 {f1:.4f}", flush=True)
        recorder.log_epoch(trial.number, {**params, "mean_val_macro_f1": float(np.mean(scores))})
        return float(np.mean(scores))

    remaining = space["trials"] - len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    try:
        study.optimize(objective, n_trials=max(0, remaining))
    except BaseException:
        record = recorder.finish(status="interrupted")
        (folder / f"session_{record['start_time'].replace(':', '')}.json").write_text(
            json.dumps(record, indent=2)
        )
        raise
    result = {
        "best_params": study.best_params,
        "best_mean_val_macro_f1": study.best_value,
        "trials": len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))),
        "partition": "val",
    }
    recorder.save_metrics(result)
    record = recorder.finish()
    record["run_name"] = name
    record["timing_scope"] = "this tuning invocation; earlier resumed invocations have separate records"
    session = folder / f"session_{record['start_time'].replace(':', '')}.json"
    session.write_text(json.dumps(record, indent=2))
    (folder / "metrics.json").write_text(json.dumps(result, indent=2))
    (folder / "record.json").write_text(json.dumps(record, indent=2))
    for filename in ("run.log", "history.csv"):
        if (recorder.run_dir / filename).exists():
            shutil.copy2(recorder.run_dir / filename, folder / filename)
    study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
    return result


def final(config: dict[str, Any], seeds: list[int]) -> None:
    require_clean_code()
    config = {**config, "cache_spec_sha256": completed_cache(config, config["mil"]["cache"])["spec_sha256"]}
    mil, classes = config["mil"], config["classes"]
    tune_name = f"stage7_tune_{mil['tuned_from']}"
    folds, checksum = read_folds(config)
    tuning_config = load_config(f"configs/mil/{mil['tuned_from']}.yaml")
    tuning_config["cache_spec_sha256"] = completed_cache(tuning_config, tuning_config["mil"]["cache"])[
        "spec_sha256"
    ]
    if not reusable_run(tune_name, tuning_config, checksum, ("metrics.json",)):
        raise ValueError("finish all declared tuning trials before final training")
    tuned = json.loads((get_default_runs_dir() / tune_name / "metrics.json").read_text())
    if tuned["trials"] != config["search"]["trials"]:
        raise ValueError("incomplete tuning budget")
    params = {**tuned["best_params"], **mil.get("override", {})}
    store = store_for(config, folds, params["descriptor"])
    for seed in seeds:
        for r in range(1, config["split"]["n_folds"] + 1):
            name = f"stage7_{mil['name']}{'' if seed == config['seed'] else f'_s{seed}'}_r{r}"
            if reusable_run(
                name,
                {**config, "params": params, "round": r, "run_seed": seed},
                checksum,
                ("model.pt", "metrics.json", "predictions.csv"),
            ):
                continue
            recorder = RunRecorder(
                name,
                config={**config, "params": params, "round": r, "run_seed": seed},
                split_checksum=checksum,
                overwrite=True,
                data_version={"primary": config["datasets"]["primary"]["version"]},
            )
            best = fit(store, folds, r, params, config, seed, log=recorder.log_epoch)
            torch.save(best["state"], recorder.run_dir / "model.pt")
            recorder.save_predictions(prediction_table(folds, best["rows"], best["probs"], r, "val", classes))
            if best["weights"][0] is not None:
                np.savez_compressed(
                    recorder.run_dir / "attention_val.npz",
                    **{folds.photo_id.iloc[row]: w for row, w in zip(best["rows"], best["weights"])},
                )
            recorder.save_metrics(
                {
                    "best_epoch": best["epoch"],
                    "val_macro_f1": best["f1"],
                    "params": params,
                    "partition": "val",
                }
            )
            recorder.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=["tune", "final"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "tune":
        print(tune(config))
    else:
        final(config, [config["seed"], *config["mil"]["stability_seeds"]])


if __name__ == "__main__":
    main()
