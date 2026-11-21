"""Stage 10: score every trained model on Paddy Doctor, once (plan: docs/reports/paddy_pre_declaration.md).

    uv run python -m src.paddy_zero_shot predict-heads   # mean, max, attention at three zooms; whole-photo heads and probes
    uv run python -m src.paddy_zero_shot predict-cnn     # the whole-image models
    uv run python -m src.paddy_zero_shot report          # tables and docs/reports/paddy_report.md

Nothing is trained, tuned or selected on Paddy Doctor. Each (run, zoom level) is scored once: a
``paddy_started.json`` marker is created before the predictions are written, a run that already has predictions
is skipped, and a run that has a marker but no predictions is refused (it crashed: stop and report, and do not
delete the marker). The methods, the zoom levels and the rules come from the frozen ``configs/paddy_zero_shot.yaml``.
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
from src import followup_report as fr
from src import train_wholeimage as tw
from src.config import REPO_ROOT, cache_folder, load_config
from src.embed import photo_key
from src.followup import read_json
from src.mil_models import MILModel
from src.paddy_embed import photo_array, photo_ids, verify
from src.photo_dataset import PhotoDataset
from src.records import get_default_runs_dir, require_clean_code
from src.seeding import seed_everything
from src.split import round_column

CONFIG = "configs/paddy_zero_shot.yaml"
PARTS = {"c": ["cls"], "m": ["mean"], "cm": ["cls", "mean"]}
PLAN_FILES = ("docs/reports/paddy_pre_declaration.md", CONFIG)


# ------------------------------------------------------------------------------------------- what is scored
def methods(config: dict[str, Any]) -> dict[str, list[str]]:
    """Method -> its five run names: the headline methods, then the Stage 5 conditions once they are trained."""
    files = config["paddy"]["methods"]
    found = {name: entry["runs"] for name, entry in read_json(REPO_ROOT / files["session_selection"])["methods"].items()}
    conditions = REPO_ROOT / files["condition_selection"]
    if conditions.exists():
        extra = read_json(conditions)["configurations"]
        if not isinstance(extra, dict) or set(extra) & set(found) or any(len(runs) != 5 for runs in extra.values()):
            raise ValueError(f"{conditions}: expected {{configuration: five runs}} with names not used by a headline method")
        found.update(extra)
    return found


def families(config: dict[str, Any]) -> dict[str, list[str]]:
    """The scored method names by how they read a photo: tile heads, whole-photo methods, whole-image models."""
    inputs, found = config["paddy"]["inputs"], methods(config)
    fixed = [*inputs["tile_methods"], *inputs["whole_photo_methods"]]
    return {"tile": list(inputs["tile_methods"]), "whole": list(inputs["whole_photo_methods"]),
            "cnn": [m for m in found if m not in fixed]}


def zooms(method: str, config: dict[str, Any]) -> list[str | None]:
    return list(config["paddy"]["zoom_factors"]) if method in config["paddy"]["inputs"]["tile_methods"] else [None]


def is_headline(zoom: str | None, config: dict[str, Any]) -> bool:
    return zoom in (None, config["paddy"]["headline_zoom"])


def truth(config: dict[str, Any]) -> pd.DataFrame:
    """The Paddy photos in the order of every description cache and prediction table, with their class."""
    split = pd.read_csv(REPO_ROOT / config["paddy"]["split_file"]).set_index("photo_id")
    return split.loc[photo_ids(config), ["class"]].reset_index()


def record(run: str) -> dict[str, Any]:
    """The settings a run was trained with; refuses a run that did not finish."""
    saved = read_json(get_default_runs_dir() / run / "record.json")
    if saved["status"] != "completed":
        raise ValueError(f"{run}: training did not complete")
    return saved["config"]


# ------------------------------------------------------------------------------------------- run-once guard
def paths(run: str, zoom: str | None):
    stem = "paddy_predictions" + (f"_{zoom}" if zoom else "")
    folder = get_default_runs_dir() / run
    return folder / f"{stem}.csv", folder / f"{stem}_started.json"


def pending(runs: list[str], zoom: str | None) -> list[str]:
    """The runs still to score. A run that has a marker but no predictions crashed and is refused."""
    todo = []
    for run in runs:
        table, marker = paths(run, zoom)
        if table.exists():
            continue
        if marker.exists():
            raise ValueError(f"{run} ({zoom}): started but has no predictions; stop and report, do not retry")
        todo.append(run)
    return todo


def save_once(config: dict[str, Any], folds: pd.DataFrame, run: str, zoom: str | None, round_: int,
              probs: np.ndarray, plan_sha256: str | None = None) -> None:
    """Create the marker (it cannot exist yet), then write the predictions in one step. The marker records the
    Stage 10 plan files unless a later stage passes the checksum of its own plan."""
    table_path, marker = paths(run, zoom)
    plan = plan_sha256 or hashlib.sha256(b"".join((REPO_ROOT / p).read_bytes() for p in PLAN_FILES)).hexdigest()
    with marker.open("x") as out:
        json.dump({"code_version": require_clean_code(), "plan_sha256": plan}, out, indent=2)
    table = tw.prediction_table(folds, np.arange(len(folds)), probs, round_, "paddy", config["classes"])
    partial = table_path.with_name(table_path.name + ".tmp")
    table.to_csv(partial, index=False)
    partial.replace(table_path)


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
    keys = sorted({key for _, parts in heads.values() for key in parts})   # a ConvNeXt (E2) cache has no "mean"
    for start in range(0, len(ids), batch):
        bags = []
        for photo_id in ids[start:start + batch]:
            with np.load(folder / "v0" / f"{photo_key(photo_id)}.npz") as saved:
                bags.append({key: saved[key].astype(np.float32) for key in keys})
        size = max(len(bag["cls"]) for bag in bags)
        padded = {key: torch.zeros(len(bags), size, dim) for key in keys}
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
def predict_cnn(config: dict[str, Any], folds: pd.DataFrame, run: str, device: torch.device) -> tuple[int, np.ndarray]:
    cfg = evaluate.restore_size_keys(record(run))
    w, round_ = cfg["wholeimage"], cfg["round"]
    seed_everything(cfg["seed"], w["determinism"])
    folds = folds.assign(**{round_column(round_): "test"})   # every Paddy photo is served as one unseen set
    data = PhotoDataset(folds, photo_array(config, w["size"], w["otsu"]), photo_ids(config), round_, "test",
                        w["size"], cfg["classes"], augmented=False)
    loader = torch.utils.data.DataLoader(data, batch_size=w["batch_size"][w["size"]], num_workers=4)
    model = tw.build_model(cfg, w["size"])
    model.load_state_dict(torch.load(get_default_runs_dir() / run / "model.pt", map_location="cpu", weights_only=True))
    rows, probs = tw.predict(model.to(device), loader, cfg, device)
    return round_, probs[np.argsort(rows)]


def predict_heads(config: dict[str, Any]) -> None:
    require_clean_code()
    seed_everything(config["seed"], config["determinism"])
    device, found, kinds, folds = torch.device("cuda"), methods(config), families(config), truth(config)
    for zoom in config["paddy"]["zoom_factors"]:
        runs = pending([run for m in kinds["tile"] for run in found[m]], zoom)
        if runs:
            verify(config, config["paddy"]["caches"]["tiles"][zoom])
            probs = predict_tile_heads(config, zoom, runs, device)
            for run in runs:
                save_once(config, folds, run, zoom, record(run)["round"], probs[run])
        print(f"tile heads at {zoom}: {len(runs)} runs scored now", flush=True)
    runs = pending([run for m in kinds["whole"] for run in found[m]], None)
    if runs:
        verify(config, config["paddy"]["caches"]["whole"])
        for run, p in predict_whole(config, runs, device).items():
            save_once(config, folds, run, None, record(run)["round"], p)
    print(f"whole-photo heads and probes: {len(runs)} runs scored now", flush=True)


def predict_cnns(config: dict[str, Any]) -> None:
    require_clean_code()
    device, found, folds = torch.device("cuda"), methods(config), truth(config)
    runs = pending([run for m in families(config)["cnn"] for run in found[m]], None)
    for run in runs:
        round_, probs = predict_cnn(config, folds, run, device)
        save_once(config, folds, run, None, round_, probs)
        print(f"{run}: scored", flush=True)
    print(f"whole-image models: {len(runs)} runs scored now", flush=True)


# ------------------------------------------------------------------------------------------- report
def load_frames(config: dict[str, Any]) -> tuple[dict[tuple[str, str | None], list[pd.DataFrame]], list[str]]:
    """The five round tables of every (method, zoom) that is fully scored, and the names of those that are not."""
    ids, frames, missing = photo_ids(config), {}, []
    for method, runs in methods(config).items():
        for zoom in zooms(method, config):
            files = [paths(run, zoom)[0] for run in runs]
            if not all(f.exists() for f in files):
                missing.append(f"{method}" + (f" ({zoom})" if zoom else ""))
                continue
            group = sorted((pd.read_csv(f) for f in files), key=lambda f: f["round"].iloc[0])
            if [int(f["round"].iloc[0]) for f in group] != [1, 2, 3, 4, 5] or any(f["photo_id"].tolist() != ids for f in group):
                raise ValueError(f"{method} ({zoom}): the tables do not cover rounds 1 to 5 on the Paddy photos in order")
            frames[(method, zoom)] = group
    return frames, missing


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


def round_rows(config: dict[str, Any], frames: dict, found: dict[str, list[str]]) -> list[dict[str, Any]]:
    """One row per (method, zoom, round): the scores of that round-model, and its drop from the session test."""
    classes, rows = config["classes"], []
    for (method, zoom), group in frames.items():
        for f, before in zip(group, session_test_f1(found[method])):
            s = metrics.scores(f["true_class"], f["pred_class"], classes)
            rows.append({"method": method, "zoom": zoom or "-", "round": int(f["round"].iloc[0]),
                         "macro_f1": s["macro_f1"], "balanced_accuracy": s["balanced_accuracy"],
                         **{f"recall_{c}": s["per_class"][c]["recall"] for c in classes},
                         "session_test_macro_f1": np.nan if before is None else before,
                         "drop": np.nan if before is None else before - s["macro_f1"]})
    return rows


def result_rows(config: dict[str, Any], frames: dict, rounds: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """One row per (method, zoom): mean ± SD over the round-models and the committee; and the committee's confusion."""
    classes, rows, confusion = config["classes"], [], []
    for (method, zoom), group in frames.items():
        mine = rounds[(rounds["method"] == method) & (rounds["zoom"] == (zoom or "-"))]
        pooled = committee(group, classes)
        together = metrics.scores(pooled["true_class"], pooled["pred_class"], classes)
        row = {"method": method, "zoom": zoom or "-", "headline": is_headline(zoom, config)}
        for key in ("macro_f1", "balanced_accuracy", *(f"recall_{c}" for c in classes), "drop"):
            row[f"{key}_mean"], row[f"{key}_sd"] = mine[key].mean(), mine[key].std(ddof=1)
        row.update(session_test_macro_f1_mean=mine["session_test_macro_f1"].mean(),
                   committee_macro_f1=together["macro_f1"], committee_balanced_accuracy=together["balanced_accuracy"],
                   **{f"committee_recall_{c}": together["per_class"][c]["recall"] for c in classes})
        rows.append(row)
        for c, counts in zip(classes, together["confusion"]):
            confusion.append({"method": method, "zoom": zoom or "-", "true_class": c,
                              **{f"pred_{p}": n for p, n in zip(classes, counts)}})
    return rows, confusion


def paired_rows(config: dict[str, Any], frames: dict) -> list[dict[str, Any]]:
    """Attention at the headline zoom against every other method (at its headline setting)."""
    classes, rule = config["classes"], config["paddy"]["comparison"]
    metric, settings = stats.macro_f1(classes), config["evaluation"]["bootstrap"]
    a_key = ("attention", config["paddy"]["headline_zoom"])
    if a_key not in frames:
        return []
    a_frames, rows = frames[a_key], []
    a_pooled, a_scores = committee(a_frames, classes), [metric(f["true_class"], f["pred_class"]) for f in a_frames]
    for (method, zoom), group in frames.items():
        if method == "attention" or not is_headline(zoom, config):
            continue
        pooled = committee(group, classes)
        result = stats.paired(a_pooled["true_class"], a_pooled["pred_class"], pooled["pred_class"], metric, settings)
        wins = sum(a > metric(b["true_class"], b["pred_class"]) for a, b in zip(a_scores, group))
        rows.append({"B": method, "difference": result["difference"], "low": result["low"], "high": result["high"],
                     "round_wins": wins, "verdict": fr.verdict(result["low"], wins, rule)})
    return rows


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.1f}"


def h3_sentence(config: dict[str, Any], paired: pd.DataFrame) -> str:
    """The pre-declared reading rule: attention beats each fine-tuned 224 px model, or H3 is not shown."""
    models = config["paddy"]["inputs"]["whole_image_models"]
    verdicts = {r.B: r.verdict for r in paired.itertuples()} if not paired.empty else {}
    left = [m for m in models if m not in verdicts]
    head = f"attention at the {config['paddy']['headline_zoom']} zoom beats each of the {len(models)} fine-tuned 224 px models"
    if left:
        return f"**H3 reading rule (pre-declaration, item 5):** not evaluable yet; no scores for {', '.join(left)}."
    ok = all(verdicts[m] == "A beats B" for m in models)
    return f"**H3 reading rule (pre-declaration, item 5):** {head}: {'yes, H3 is supported' if ok else 'no, H3 is not shown by this rule'}."


def report(config: dict[str, Any]) -> None:
    frames, missing = load_frames(config)
    found, classes, head = methods(config), config["classes"], config["paddy"]["headline_zoom"]
    rounds = pd.DataFrame(round_rows(config, frames, found))
    summary, confusion = result_rows(config, frames, rounds)
    results, paired = pd.DataFrame(summary), pd.DataFrame(paired_rows(config, frames))
    tables = REPO_ROOT / config["paddy"]["report"]["tables"].rsplit("/", 1)[0]
    tables.mkdir(parents=True, exist_ok=True)
    for name, table in (("results", results), ("rounds", rounds), ("confusion", pd.DataFrame(confusion)), ("paired", paired)):
        table.to_csv(tables / f"paddy_{name}.csv", index=False)
    lines = ["# Paddy Doctor zero-shot test", "",
             "Built by `src/paddy_zero_shot.py` from `runs/*/paddy_predictions*.csv`; do not edit by hand. Plan: "
             "`docs/reports/paddy_pre_declaration.md` (frozen before the run). All scores are percentages on the "
             f"{len(truth(config))} Paddy Doctor photos of the four tested classes; nothing was trained or tuned on them. "
             "Per-round scores, per-class recall of every round and the confusion matrices are in "
             "`results/tables/paddy_rounds.csv` and `paddy_confusion.csv`.", "",
             "## 1. Results (mean ± SD over the five round-models, and the 5-model committee)", "",
             "| method | zoom | macro-F1 | balanced acc. | committee macro-F1 | "
             + " | ".join(f"committee recall {c}" for c in classes) + " | session test macro-F1 | drop |",
             "|---|---|---|---|---|" + "---|" * len(classes) + "---|---|"]
    for r in results.itertuples():
        recalls = " | ".join(pct(getattr(r, f"committee_recall_{c}")) for c in classes)
        lines.append(f"| {r.method} | {r.zoom}{' (headline)' if r.headline and r.zoom != '-' else ''} | "
                     f"{pct(r.macro_f1_mean)} ± {pct(r.macro_f1_sd)} | {pct(r.balanced_accuracy_mean)} ± "
                     f"{pct(r.balanced_accuracy_sd)} | {pct(r.committee_macro_f1)} | {recalls} | "
                     f"{pct(r.session_test_macro_f1_mean)} | {pct(r.drop_mean)} ± {pct(r.drop_sd)} |")
    if missing:
        lines += ["", f"Not scored yet: {', '.join(missing)}."]
    lines += ["", f"## 2. Attention at the headline zoom ({head}) against every other method (committees, photo bootstrap)", ""]
    if paired.empty:
        lines.append("Attention was not scored.")
    else:
        lines += ["| B | difference (points) | photo range | rounds won | verdict |", "|---|---|---|---|---|"]
        lines += [f"| {r.B} | {100 * r.difference:.1f} | [{100 * r.low:.1f}, {100 * r.high:.1f}] | {r.round_wins}/5 | {r.verdict} |"
                  for r in paired.itertuples()]
    lines += ["", h3_sentence(config, paired)]
    (REPO_ROOT / config["paddy"]["report"]["markdown"]).write_text("\n".join(lines) + "\n")
    print(f"report: {len(results)} rows, {len(paired)} comparisons, {len(missing)} not scored")


ACTIONS = {"predict-heads": predict_heads, "predict-cnn": predict_cnns, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    ACTIONS[args.action](load_config(args.config))
    print(f"PADDY {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
