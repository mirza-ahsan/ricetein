"""Stage 10: score every trained model on Paddy Doctor, once (plan: docs/reports/paddy_pre_declaration.md).

    uv run python -m src.paddy_zero_shot predict-heads   # mean, max, attention at three zooms; whole-photo heads and probes
    uv run python -m src.paddy_zero_shot predict-cnn     # the whole-image models
    uv run python -m src.paddy_zero_shot report          # tables and docs/reports/paddy_report.md

Nothing is trained, tuned or selected on Paddy Doctor. Each (run, zoom level) is scored once: a
``paddy_started.json`` marker is written before the predictions, and a run that has a marker or predictions is
refused. If a run crashed after its marker, stop and report; do not delete the marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import evaluate, metrics, stats
from src import train_wholeimage as tw
from src.config import REPO_ROOT, cache_folder, load_config
from src.embed import photo_key
from src.mil_models import MILModel
from src.paddy_embed import description_caches, photo_array, photo_ids
from src.photo_dataset import PhotoDataset
from src.records import get_default_runs_dir, require_clean_code
from src.split import round_column

CONFIG = "configs/paddy_zero_shot.yaml"
PARTS = {"c": ["cls"], "m": ["mean"], "cm": ["cls", "mean"]}
TILE_METHODS = ("mean", "max", "attention")
WHOLE_HEADS = ("whole224", "whole512")
PROBES = ("probe_224", "probe_512")
FINE_TUNED = ("inceptionv3_224", "efficientnetv2s_224", "convnext_tiny_224", "swin_tiny_224")
PLAN_FILES = ("docs/reports/paddy_pre_declaration.md", CONFIG)


def read_json(path) -> Any:
    return json.loads(path.read_text())


def methods(config: dict[str, Any]) -> dict[str, list[str]]:
    """Method -> its five run names: the 11 headline methods, then the Stage 5 conditions once they are trained."""
    runs = get_default_runs_dir()
    found = {name: entry["runs"] for name, entry in read_json(runs / "session_selection.json")["methods"].items()}
    conditions = runs / "session_conditions_selection.json"
    if conditions.exists():
        found.update(read_json(conditions)["configurations"])
    return found


def zooms(method: str, config: dict[str, Any]) -> list[str | None]:
    return list(config["paddy"]["zoom_factors"]) if method in TILE_METHODS else [None]


def truth(config: dict[str, Any]) -> pd.DataFrame:
    split = pd.read_csv(REPO_ROOT / config["paddy"]["split_file"]).set_index("photo_id")
    return split.loc[photo_ids(config), ["class"]].reset_index()


def paths(run: str, zoom: str | None):
    stem = "paddy_predictions" + (f"_{zoom}" if zoom else "")
    folder = get_default_runs_dir() / run
    return folder / f"{stem}.csv", folder / f"{stem}_started.json"


def save_once(config: dict[str, Any], run: str, zoom: str | None, round_: int, probs: np.ndarray) -> None:
    """Write the marker, then the predictions; a run that was started before is refused."""
    table_path, marker = paths(run, zoom)
    if table_path.exists() or marker.exists():
        raise ValueError(f"{run} ({zoom}): already scored on Paddy Doctor or started; no silent retry")
    plan = hashlib.sha256(b"".join((REPO_ROOT / p).read_bytes() for p in PLAN_FILES)).hexdigest()
    marker.write_text(json.dumps({"code_version": require_clean_code(), "plan_sha256": plan}, indent=2))
    folds = truth(config)
    tw.prediction_table(folds, np.arange(len(folds)), probs, round_, "paddy", config["classes"]).to_csv(
        table_path, index=False)


def record(run: str) -> dict[str, Any]:
    return read_json(get_default_runs_dir() / run / "record.json")["config"]


# ------------------------------------------------------------------------------------------- heads and probes
def load_head(run: str, dim: int, device: torch.device) -> tuple[MILModel, list[str]]:
    config = record(run)
    parts = PARTS[config["params"]["descriptor"]]
    model = MILModel(dim * len(parts), config["params"]["width"], config["params"]["dropout"],
                     config["mil"]["aggregator"], len(config["classes"]))
    model.load_state_dict(torch.load(get_default_runs_dir() / run / "model.pt", map_location="cpu", weights_only=True))
    return model.eval().to(device), parts


@torch.inference_mode()
def predict_tile_heads(config: dict[str, Any], zoom: str, runs: list[str], device: torch.device,
                       batch: int = 8) -> dict[str, np.ndarray]:
    """Probabilities of every tile head for every Paddy photo at one zoom level, in one pass over the cache."""
    folder = cache_folder(config["paddy"]["caches"]["tiles"][zoom])
    dim = config["encoders"][config["paddy"]["encoder"]]["dim"]
    heads = {run: load_head(run, dim, device) for run in runs}
    ids, probs = photo_ids(config), {run: [] for run in runs}
    for start in range(0, len(ids), batch):
        bags = []
        for photo_id in ids[start:start + batch]:
            with np.load(folder / "v0" / f"{photo_key(photo_id)}.npz") as saved:
                bags.append({key: saved[key].astype(np.float32) for key in ("cls", "mean")})
        size = max(len(bag["cls"]) for bag in bags)
        padded = {key: torch.zeros(len(bags), size, dim) for key in ("cls", "mean")}
        mask = torch.zeros(len(bags), size, dtype=torch.bool)
        for i, bag in enumerate(bags):
            for key in padded:
                padded[key][i, :len(bag[key])] = torch.from_numpy(bag[key])
            mask[i, :len(bag["cls"])] = True
        for run, (model, parts) in heads.items():
            logits, _ = model(torch.cat([padded[k] for k in parts], -1).to(device), mask.to(device))
            probs[run].append(torch.softmax(logits, 1).cpu().numpy())
    return {run: np.concatenate(p) for run, p in probs.items()}


def whole_descriptors(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """{"cls": (n, sizes, dim), "mean": ...}: the whole-photo descriptions of every Paddy photo."""
    folder = cache_folder(config["paddy"]["caches"]["whole"])
    arrays: dict[str, list[np.ndarray]] = {"cls": [], "mean": []}
    for photo_id in photo_ids(config):
        with np.load(folder / "v0" / f"{photo_key(photo_id)}.npz") as saved:
            for key in arrays:
                arrays[key].append(saved[key].astype(np.float32))
    return {key: np.stack(value) for key, value in arrays.items()}


@torch.inference_mode()
def predict_whole(config: dict[str, Any], runs: list[str], device: torch.device) -> dict[str, np.ndarray]:
    """Probabilities of the whole-photo heads (one-tile bags) and probes (logistic regression on the class token)."""
    descriptors, sizes = whole_descriptors(config), config["paddy"]["caches"]["whole_sizes"]
    dim = config["encoders"][config["paddy"]["encoder"]]["dim"]
    out = {}
    for run in runs:
        cfg = record(run)
        if "probe" in cfg:
            x = descriptors[cfg["probe"]["descriptor"]][:, sizes.index(cfg["probe"]["size"])]
            with (get_default_runs_dir() / run / "model.pkl").open("rb") as f:
                model = pickle.load(f)  # our own local, checksum-verified checkpoint
            out[run] = model.predict_proba(x)[:, np.argsort(model.classes_)]
        else:
            model, parts = load_head(run, dim, device)
            index = sizes.index(cfg["whole_bag"]["size"])
            x = torch.from_numpy(np.concatenate([descriptors[k][:, index] for k in parts], -1))[:, None, :]
            logits, _ = model(x.to(device), torch.ones(len(x), 1, dtype=torch.bool, device=device))
            out[run] = torch.softmax(logits, 1).cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------- whole-image models
def predict_cnn(config: dict[str, Any], run: str, device: torch.device) -> tuple[int, np.ndarray]:
    cfg = evaluate.restore_size_keys(record(run))
    w, round_ = cfg["wholeimage"], cfg["round"]
    ids, folds = photo_ids(config), truth(config)
    folds[round_column(round_)] = "test"          # every Paddy photo is served as one unseen set
    data = PhotoDataset(folds, photo_array(config, w["size"], w["otsu"]), ids, round_, "test", w["size"],
                        cfg["classes"], augmented=False)
    loader = torch.utils.data.DataLoader(data, batch_size=w["batch_size"][w["size"]], num_workers=4)
    model = tw.build_model(cfg, w["size"])
    model.load_state_dict(torch.load(get_default_runs_dir() / run / "model.pt", map_location="cpu", weights_only=True))
    rows, probs = tw.predict(model.to(device), loader, cfg, device)
    return round_, probs[np.argsort(rows)]


def predict_heads(config: dict[str, Any]) -> None:
    device, found = torch.device("cuda"), methods(config)
    for zoom in config["paddy"]["zoom_factors"]:
        runs = [run for m in TILE_METHODS for run in found[m]]
        probs = predict_tile_heads(config, zoom, runs, device)
        for run in runs:
            save_once(config, run, zoom, record(run)["round"], probs[run])
        print(f"tile heads at {zoom}: {len(runs)} runs scored", flush=True)
    runs = [run for m in (*WHOLE_HEADS, *PROBES) for run in found[m]]
    for run, p in predict_whole(config, runs, device).items():
        save_once(config, run, None, record(run)["round"], p)
    print(f"whole-photo heads and probes: {len(runs)} runs scored", flush=True)


def predict_cnns(config: dict[str, Any]) -> None:
    device, found = torch.device("cuda"), methods(config)
    runs = [run for name, names in found.items() if name not in (*TILE_METHODS, *WHOLE_HEADS, *PROBES) for run in names]
    for run in runs:
        round_, probs = predict_cnn(config, run, device)
        save_once(config, run, None, round_, probs)
        print(f"{run}: scored", flush=True)


# ------------------------------------------------------------------------------------------- report
def load_frames(config: dict[str, Any]) -> dict[tuple[str, str | None], list[pd.DataFrame]]:
    frames = {}
    for method, runs in methods(config).items():
        for zoom in zooms(method, config):
            files = [paths(run, zoom)[0] for run in runs]
            if all(f.exists() for f in files):
                frames[(method, zoom)] = [pd.read_csv(f) for f in files]
    return frames


def committee(frames: list[pd.DataFrame], classes: list[str]) -> pd.DataFrame:
    """The average of the round-models' probabilities, and the class it picks, for every photo."""
    out = frames[0][["photo_id", "true_class"]].copy()
    mean = np.mean([metrics.probabilities(f, classes) for f in frames], axis=0)
    out["pred_class"] = np.array(classes)[mean.argmax(axis=1)]
    return out


def session_test_f1(runs: list[str]) -> list[float | None]:
    """The in-domain test macro-F1 of each round-model on the session-blocked split (for the drop)."""
    files = [get_default_runs_dir() / run / "test_metrics.json" for run in runs]
    return [read_json(f)["macro_f1"] if f.exists() else None for f in files]


def result_rows(config: dict[str, Any], frames: dict) -> list[dict[str, Any]]:
    classes, found, rows = config["classes"], methods(config), []
    for (method, zoom), group in frames.items():
        rounds = [metrics.scores(f["true_class"], f["pred_class"], classes) for f in group]
        pooled = committee(group, classes)
        together = metrics.scores(pooled["true_class"], pooled["pred_class"], classes)
        f1 = np.array([r["macro_f1"] for r in rounds])
        before = session_test_f1(found[method])
        drops = [b - a for b, a in zip(before, f1) if b is not None]
        rows.append({"method": method, "zoom": zoom or "-", "headline": zoom in (None, config["paddy"]["headline_zoom"]),
                     "macro_f1_mean": f1.mean(), "macro_f1_sd": f1.std(ddof=1),
                     "balanced_accuracy_mean": np.mean([r["balanced_accuracy"] for r in rounds]),
                     "committee_macro_f1": together["macro_f1"], "committee_balanced_accuracy": together["balanced_accuracy"],
                     **{f"committee_recall_{c}": together["per_class"][c]["recall"] for c in classes},
                     "session_test_macro_f1_mean": np.mean([b for b in before if b is not None]) if drops else np.nan,
                     "drop_mean": np.mean(drops) if drops else np.nan})
    return rows


def paired_rows(config: dict[str, Any], frames: dict) -> list[dict[str, Any]]:
    """Attention at the headline zoom against every other method (at its headline setting)."""
    classes, rule = config["classes"], config["paddy"]["comparison"]
    metric, settings = stats.macro_f1(classes), config["evaluation"]["bootstrap"]
    a_key = ("attention", config["paddy"]["headline_zoom"])
    if a_key not in frames:
        return []
    a_frames, rows = frames[a_key], []
    a_pooled = committee(a_frames, classes)
    for (method, zoom), group in frames.items():
        if method == "attention" or zoom not in (None, config["paddy"]["headline_zoom"]):
            continue
        pooled = committee(group, classes)
        result = stats.paired(a_pooled["true_class"], a_pooled["pred_class"], pooled["pred_class"], metric, settings)
        wins = sum(metric(a["true_class"], a["pred_class"]) > metric(b["true_class"], b["pred_class"])
                   for a, b in zip(a_frames, group))
        better = result["low"] > 0 and wins >= rule["min_round_wins"]
        rows.append({"B": method, "difference": result["difference"], "low": result["low"], "high": result["high"],
                     "round_wins": wins, "verdict": "A beats B" if better else "no clear difference"})
    return rows


def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.1f}"


def report(config: dict[str, Any]) -> None:
    frames = load_frames(config)
    results, paired = pd.DataFrame(result_rows(config, frames)), pd.DataFrame(paired_rows(config, frames))
    tables = REPO_ROOT / "results/tables"
    results.to_csv(tables / "paddy_results.csv", index=False)
    paired.to_csv(tables / "paddy_paired.csv", index=False)
    classes, head = config["classes"], config["paddy"]["headline_zoom"]
    lines = ["# Paddy Doctor zero-shot test", "",
             "Built by `src/paddy_zero_shot.py` from `runs/*/paddy_predictions*.csv`; do not edit by hand. Plan: "
             "`docs/reports/paddy_pre_declaration.md` (frozen before the run). All scores are percentages on the "
             f"{len(truth(config))} Paddy Doctor photos of the four tested classes; nothing was trained or tuned on them.", "",
             "## 1. Results (mean ± SD over the five round-models, and the 5-model committee)", "",
             "| method | zoom | macro-F1 | committee macro-F1 | committee balanced acc. | "
             + " | ".join(f"recall {c}" for c in classes) + " | session test macro-F1 | drop |",
             "|---|---|---|---|---|" + "---|" * len(classes) + "---|---|"]
    for r in results.itertuples():
        recalls = " | ".join(pct(getattr(r, f"committee_recall_{c}")) for c in classes)
        mark = " (headline)" if r.headline and r.zoom != "-" else ""
        lines.append(f"| {r.method} | {r.zoom}{mark} | {pct(r.macro_f1_mean)} ± {pct(r.macro_f1_sd)} | "
                     f"{pct(r.committee_macro_f1)} | {pct(r.committee_balanced_accuracy)} | {recalls} | "
                     f"{pct(r.session_test_macro_f1_mean)} | {pct(r.drop_mean)} |")
    lines += ["", f"## 2. Attention at the headline zoom ({head}) against every other method (committees, photo bootstrap)", ""]
    if paired.empty:
        lines.append("Attention was not scored.")
    else:
        lines += ["| B | difference (points) | photo range | rounds won | verdict |", "|---|---|---|---|---|"]
        lines += [f"| {r.B} | {100 * r.difference:.1f} | [{100 * r.low:.1f}, {100 * r.high:.1f}] | {r.round_wins}/5 | {r.verdict} |"
                  for r in paired.itertuples()]
        beaten = [r.verdict == "A beats B" for r in paired.itertuples() if r.B in FINE_TUNED]
        lines += ["", "**H3 reading rule (pre-declaration, item 5):** attention at the native zoom beats each of the four "
                  f"fine-tuned 224 px models: {'yes, H3 is supported' if len(beaten) == 4 and all(beaten) else 'no, H3 is not shown by this rule'}."]
    (REPO_ROOT / config["paddy"]["report"]["markdown"]).write_text("\n".join(lines) + "\n")
    print(f"report: {len(results)} rows, {len(paired)} comparisons")


ACTIONS = {"predict-heads": predict_heads, "predict-cnn": predict_cnns, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=ACTIONS)
    args = parser.parse_args()
    ACTIONS[args.action](load_config(CONFIG))
    print(f"PADDY {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
