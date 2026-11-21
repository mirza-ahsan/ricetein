"""Stage 13 distillation (declared in handbook §24, 2026-11-21).

    uv run python -m src.distill

A small student (the E4 attention head, with the settings of the E4 ablation and nothing tuned) is trained, in each of
the five rounds, to imitate the soft answers of a teacher committee. **Training photos only:** the teacher answers on
the clean version of each training photo of the round; the student sees the cached versions of that photo (augmented
copies stay with their source photo) and the same answer. Validation photos stop the training and are scored;
test photos are never read. Loss: cross-entropy between the teacher's probabilities and the student's, temperature
from ``configs/cost.yaml``, no hard labels, no class weights.

The teacher is the Stage 11 committee: its answers on the training photos come from ``distill.teacher.soft_answers_file``
when Stage 11 saved them, else they are computed here as the mean probability of ``distill.teacher.members``
(selected tile heads on E1, probes, or fine-tuned whole-image models) with each member's own round-model. A
whole-image member is scored on the clean training photos of the round only (evaluation preprocessing, no
augmentation); its answers, like the heads', are in-sample. The temperature softens the teacher's answers
only. Run all five rounds in one go: ``reusable_run`` compares every file under ``src/`` and ``configs/``, so a resume
after an unrelated commit asks for the unfinished rounds to be archived.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import metrics, train_mil
from src.compress import cached_bags, head_probs, load_head, load_probe, role_rows, selected_runs
from src.config import REPO_ROOT, load_config
from src.followup import ROUNDS, read_json
from src.mil_models import MILModel
from src.records import RunRecorder, get_default_runs_dir, require_clean_code, reusable_run
from src.seeding import seed_everything
from src.split import read_folds
from src.train_wholeimage import prediction_table

STUDENT_ABLATION = "stage7_session_ablate_encoder_e4_r1"    # its record holds the settings the student reuses


def soften(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature on probabilities, renormalised (temperature 1 changes nothing)."""
    powered = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / temperature)
    return powered / powered.sum(axis=1, keepdims=True)


def scatter(folds: pd.DataFrame, rows: np.ndarray, probs: np.ndarray, temperature: float) -> np.ndarray:
    """(photos, classes): the answers on ``rows`` (softened), NaN for every other photo."""
    targets = np.full((len(folds), probs.shape[1]), np.nan, np.float32)
    targets[rows] = soften(probs, temperature)
    return targets


def teacher_targets(folds: pd.DataFrame, round_: int, members: list[Callable[[np.ndarray], np.ndarray]],
                    temperature: float) -> np.ndarray:
    """The members' mean probability on the TRAINING photos of the round, NaN for every other photo. A member is
    asked about training rows only."""
    rows = role_rows(folds, "train")[round_]
    return scatter(folds, rows, np.mean([member(rows) for member in members], axis=0), temperature)


def file_targets(path, folds: pd.DataFrame, round_: int, classes: list[str], temperature: float) -> np.ndarray:
    """The same, from a CSV of Stage 11 (photo_id, round, prob_<class>). Every training photo of the round must be
    there, and no photo that is not a training photo of the round may be."""
    table = pd.read_csv(path, dtype={"photo_id": str})
    table = table[table["round"] == round_].set_index("photo_id")
    rows = role_rows(folds, "train")[round_]
    wanted = folds["photo_id"].to_numpy()[rows]
    if set(table.index) != set(wanted):
        raise ValueError(f"{path}: round {round_} must hold exactly the training photos of the round")
    return scatter(folds, rows, table.loc[wanted, [f"prob_{c}" for c in classes]].to_numpy(float), temperature)


def probe_member(method: str, round_: int, folds: pd.DataFrame, shared: dict) -> Callable[[np.ndarray], np.ndarray]:
    from src import linear_probe

    if method not in shared:      # the whole-photo descriptions do not depend on the round: read them once
        config = load_config(f"configs/session/probe_e1_{method.split('_')[1]}.yaml")   # probe_224 -> size 224
        shared[method] = linear_probe.descriptors(config, folds, 1)[0]
    x, model = shared[method], load_probe(method, round_)
    order = np.argsort(model.classes_)
    return lambda rows: model.predict_proba(x[rows])[:, order]


CNN_MEMBERS = ("inceptionv3_224", "efficientnetv2s_224", "convnext_tiny_224", "swin_tiny_224", "convnext_tiny_448")


def cnn_answers(model, config: dict[str, Any], folds: pd.DataFrame, round_: int, images: np.ndarray, ids: list[str],
                device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(rows, probabilities) of one round's whole-image model on the CLEAN TRAINING photos of the round: the
    evaluation preprocessing (resize only), no augmentation. No validation or test photo is in the dataset."""
    from src import photo_dataset
    from src import train_wholeimage as tw

    w = config["wholeimage"]
    data = photo_dataset.PhotoDataset(folds, images, ids, round_, "train", w["size"], config["classes"], augmented=False)
    loader = torch.utils.data.DataLoader(data, batch_size=w["batch_size"][w["size"]], shuffle=False,
                                         num_workers=w["workers"])
    return tw.predict(model.to(device), loader, config, device)


def cnn_member(method: str, round_: int, folds: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
    """A fine-tuned whole-image member: its round-r checkpoint, asked about the training photos of the round only."""
    from src import evaluate, photo_dataset
    from src import train_wholeimage as tw

    def member(rows: np.ndarray) -> np.ndarray:
        if not np.array_equal(rows, role_rows(folds, "train")[round_]):
            raise ValueError(f"{method}: asked about photos that are not exactly the training photos of round {round_}")
        run = selected_runs(method)[round_ - 1]
        config = evaluate.restore_size_keys(read_json(get_default_runs_dir() / run / "record.json")["config"])
        w, device = config["wholeimage"], torch.device("cuda")
        model = tw.build_model(config, w["size"])
        model.load_state_dict(torch.load(get_default_runs_dir() / run / "model.pt", map_location="cpu", weights_only=True))
        scored, probs = cnn_answers(model, config, folds, round_, photo_dataset.photo_cache(config, w["size"], w["otsu"]),
                                    photo_dataset.photo_ids(config), device)
        del model
        torch.cuda.empty_cache()
        if not np.array_equal(scored, rows):
            raise RuntimeError(f"{method}: the scored photos differ from the training photos of round {round_}")
        return probs
    return member


def tile_member(method: str, round_: int, config: dict[str, Any], folds: pd.DataFrame, shared: dict) -> Callable:
    model, params = load_head(selected_runs(method)[round_ - 1])
    bag_of, ids = cached_bags(config, "E1", params["descriptor"], folds, shared), folds["photo_id"].to_numpy()
    return lambda rows: np.array([head_probs(model, bag_of(ids[r])) for r in rows])


def build_members(names: list[str], round_: int, config: dict[str, Any], folds: pd.DataFrame,
                  shared: dict) -> list[Callable]:
    """``shared`` keeps what does not depend on the round (descriptions, packed stores) across the five rounds."""
    return [probe_member(n, round_, folds, shared) if n.startswith("probe")
            else cnn_member(n, round_, folds) if n in CNN_MEMBERS else tile_member(n, round_, config, folds, shared)
            for n in names]


def student_loss(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between the teacher's probabilities and the student's."""
    return -(soft * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def fit_student(store, folds: pd.DataFrame, round_: int, params: dict[str, Any], config: dict[str, Any],
                targets: np.ndarray, seed: int, log=None) -> dict[str, Any]:
    """Train one round's student. Same loop, optimiser, schedule and stopping as ``train_mil.fit``, with the soft
    loss. ``targets`` must be finite for every training photo of the round."""
    mil, classes = config["mil"], config["classes"]
    seed_everything(seed, config["determinism"])
    torch.set_num_threads(mil.get("cpu_threads", 4))
    device = torch.device(mil.get("device", "cuda"))
    train = train_mil.Bags(store, folds, round_, "train", classes, mil["versions"], params["tile_dropout"],
                           mil["batch"], seed)
    val = train_mil.Bags(store, folds, round_, "val", classes, 1, 0.0, mil["batch"], seed)
    if not np.isfinite(targets[train.rows]).all():
        raise ValueError("the teacher gave no answer for some training photos")
    soft = torch.from_numpy(targets)
    model = MILModel(store.dim, params["width"], params["dropout"], "attention", len(classes)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"],
                                  eps=mil["adam_eps"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda e: 0.5 * (1 + math.cos(math.pi * e / mil["max_epochs"])))
    best = {"f1": -1.0}
    for epoch in range(1, mil["max_epochs"] + 1):
        model.train()
        losses = []
        for tiles, mask, _, rows in train:
            loss = student_loss(model(tiles.to(device), mask.to(device))[0], soft[rows].to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        schedule.step()
        rows, probs, _ = train_mil.predict(model, val, device)
        f1 = metrics.scores(train.labels[rows], probs.argmax(1), list(range(len(classes))))["macro_f1"]
        if log:
            log(epoch, {"train_loss": float(np.mean(losses)), "val_macro_f1": f1})
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "rows": rows, "probs": probs,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= mil["patience"]:
            break
    return best


def run(config: dict[str, Any]) -> None:
    from src.session_ablations import pack_one_key_at_a_time, variant_config

    require_clean_code()
    root = get_default_runs_dir()
    base = read_json(root / STUDENT_ABLATION / "record.json")["config"]
    params = base.pop("params")
    for key in ("round", "run_seed"):
        base.pop(key)
    d = config["distill"]
    folds, checksum = read_folds(base)
    pack_one_key_at_a_time(variant_config("encoder_e4"))
    store = train_mil.store_for(base, folds, params["descriptor"])
    classes, shared = base["classes"], {}
    for r in ROUNDS:
        name = f"cost_distill_e4_r{r}"
        run_config = {**base, "params": params, "round": r, "run_seed": base["seed"], "distill": d}
        if reusable_run(name, run_config, checksum, ("model.pt", "metrics.json", "predictions.csv")):
            continue
        if d["teacher"]["soft_answers_file"]:
            targets = file_targets(REPO_ROOT / d["teacher"]["soft_answers_file"], folds, r, classes, d["temperature"])
        else:
            members = build_members(d["teacher"]["members"], r, base, folds, shared)
            targets = teacher_targets(folds, r, members, d["temperature"])
        recorder = RunRecorder(name, config=run_config, split_checksum=checksum, overwrite=True,
                               data_version={"primary": base["datasets"]["primary"]["version"]})
        best = fit_student(store, folds, r, params, base, targets, base["seed"], log=recorder.log_epoch)
        torch.save(best["state"], recorder.run_dir / "model.pt")
        recorder.save_predictions(prediction_table(folds, best["rows"], best["probs"], r, "val", classes))
        recorder.save_metrics({"best_epoch": best["epoch"], "val_macro_f1": best["f1"], "params": params,
                               "partition": "val", "teacher": d["teacher"]})
        recorder.finish()
        print(f"{name}: validation macro-F1 {best['f1']:.3f}", flush=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    run(load_config("configs/cost.yaml"))
    print("DISTILL DONE", flush=True)


if __name__ == "__main__":
    main()
