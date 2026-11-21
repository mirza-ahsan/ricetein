"""Stage 13 compression variants (declared 2026-11-21).

    uv run python -m src.compress fewer-tiles    # validation macro-F1 of every tile variant (needs distilled runs)
    uv run python -m src.compress int8-describe  # describe the 8-bit subset on the CPU (long, resumable)
    uv run python -m src.compress int8-score     # float against 8-bit descriptions on that subset
    uv run python -m src.compress export         # ONNX files of the final small model and the parity check

Every accuracy here is **validation** macro-F1 (session split, mean over the five rounds). The round-models are
applied as they are: nothing is retrained or tuned. The tile rule reads pixels only, never a label.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from multiprocessing import Pool
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from src import embed, metrics, otsu, tiling, train_mil
from src.config import REPO_ROOT, cache_folder, dataset_dir, load_config
from src.encoders import describe, load_encoder
from src.followup import ROUNDS, read_json
from src.leaf_width import rescale_factor
from src.mil_models import MILModel
from src.records import RunRecorder, completed_run, get_default_runs_dir, require_clean_code
from src.session_ablations import PARTS, pack_one_key_at_a_time
from src.split import read_folds, round_column, sha256
from src.train_wholeimage import prediction_table

TILE = 224
RULE_VERSION = 1
NEUTRAL = embed.version_settings({}, 0, 0, "")           # version 0: the photo as it is
CACHES = {"E1": "e1_dinov3_vitb16", "E4": "e4_dinov3_vits16"}
ENCODER = {"mean": "E1", "max": "E1", "attention": "E1", "e4": "E4", "distilled_e4": "E4"}
# variant -> (head, which tiles). The first is the reference; the others are measured; "even16" is a control.
VARIANTS = {"attention": ("attention", "all"), "e1_top16": ("attention", "top16"), "e1_even16": ("attention", "even16"),
            "e4": ("e4", "all"), "e4_top16": ("e4", "top16"), "e4_even16": ("e4", "even16"),
            "distilled_e4": ("distilled_e4", "all"), "distilled_e4_top16": ("distilled_e4", "top16")}
# the candidates for the final small model: those that run from cached descriptions (the 8-bit encoder is not one)
CANDIDATES = ["e1_top16", "e4", "e4_top16", "distilled_e4", "distilled_e4_top16"]
Policy = Callable[[str, int], "np.ndarray | None"]   # (photo, number of tiles) -> indices of the tiles kept, or None


# ---------------------------------------------------------------------------------------------- which runs
def selected_runs(method: str) -> list[str]:
    """The five run names of a main method, from the frozen validation-only selections."""
    root = get_default_runs_dir()
    if method == "convnext_tiny_448":
        return read_json(root / "session_conditions_selection.json")["configurations"][method]
    return read_json(root / "session_selection.json")["methods"][method]["runs"]


def head_runs(head: str) -> list[str]:
    """The five round-models of a tile head: a selected E1 head, the E4 ablation, or the distilled E4 head."""
    if head == "e4":
        return [f"stage7_session_ablate_encoder_e4_r{r}" for r in ROUNDS]
    if head == "distilled_e4":
        return [f"cost_distill_e4_r{r}" for r in ROUNDS]
    return selected_runs(head)


def load_head(run: str) -> tuple[MILModel, dict[str, Any]]:
    """A trained tile head from its run folder, in eval mode on the CPU, and the settings it was trained with."""
    folder = get_default_runs_dir() / run
    config = read_json(folder / "record.json")["config"]
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    params = config["params"]
    model = MILModel(state["project.0.weight"].shape[1], params["width"], params["dropout"],
                     config["mil"]["aggregator"], len(config["classes"]))
    model.load_state_dict(state)
    return model.eval(), params


# ---------------------------------------------------------------------------------------------- the tile rule
def leaf_scale_photo(path: str, decision: dict[str, Any]) -> np.ndarray:
    """Open a photo and bring it to the common leaf scale (the Stage 6 step, version 0)."""
    with Image.open(path) as image:
        short = min(image.size)
    return embed.load_version(path, float(rescale_factor(short, decision)), NEUTRAL)


def grid_of(rgb: np.ndarray) -> np.ndarray:
    """The tile boxes ``tiling.cut`` would use for this photo."""
    h, w = rgb.shape[:2]
    return tiling.boxes(max(w, TILE), max(h, TILE), TILE)


def top_tiles(rgb: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    """Indices (ascending) of the tiles to describe, from a shrunk view of the photo only.

    Shrink by ``shrink`` (area), find the leaf with Otsu, and score each tile of the normal grid by the mean colour
    distance of its leaf pixels from the photo's median leaf colour. Tiles with less than ``min_leaf_share`` leaf
    are not eligible. The ``tiles`` highest scores win (ties by tile order); if fewer are eligible, the rest are
    filled by leaf share. No label and no model is used.
    """
    grid, shrink = grid_of(rgb), rule["shrink"]
    if len(grid) <= rule["tiles"]:
        return np.arange(len(grid))
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, w // shrink), max(1, h // shrink)), interpolation=cv2.INTER_AREA)
    leaf = otsu.leaf_mask(small)
    pixels = small[leaf] if leaf.any() else small.reshape(-1, 3)
    distance = np.linalg.norm(small.astype(np.float32) - np.median(pixels, axis=0), axis=2) * leaf
    score, share = np.zeros(len(grid)), np.zeros(len(grid))
    for i, (x, y, _, _) in enumerate(grid):
        window = (slice(y // shrink, (y + TILE) // shrink), slice(x // shrink, (x + TILE) // shrink))
        count = leaf[window].sum()
        share[i] = leaf[window].mean() if leaf[window].size else 0.0
        score[i] = distance[window].sum() / max(count, 1)
    eligible = np.flatnonzero(share >= rule["min_leaf_share"])
    first = eligible[np.lexsort((eligible, -score[eligible]))][:rule["tiles"]]
    if len(first) < rule["tiles"]:
        rest = np.setdiff1d(np.arange(len(grid)), first)
        first = np.concatenate([first, rest[np.lexsort((rest, -share[rest]))][:rule["tiles"] - len(first)]])
    return np.sort(first)


def even_tiles(n: int, k: int) -> np.ndarray:
    """The control: ``k`` evenly spaced tiles of ``n`` (all of them if there are no more than ``k``)."""
    return np.arange(n) if n <= k else np.unique(np.linspace(0, n - 1, k).round().astype(int))


def cut_selected(rgb: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The chosen tiles only (same pixels as ``tiling.cut``)."""
    if min(rgb.shape[:2]) < TILE:
        return tiling.cut(rgb, TILE)[0][indices]
    return np.stack([rgb[y:y + TILE, x:x + TILE] for x, y, _, _ in grid_of(rgb)[indices]])


def rule_of(config: dict[str, Any]) -> dict[str, Any]:
    c = config["compress"]
    return {"version": RULE_VERSION, "tile": TILE, "tiles": c["tiles"], "shrink": c["shrink"],
            "min_leaf_share": c["min_leaf_share"]}


def _select_one(job: tuple[str, str, dict, dict]) -> tuple[str, int, list[int]]:
    photo_id, path, decision, rule = job
    rgb = leaf_scale_photo(path, decision)
    return photo_id, len(grid_of(rgb)), top_tiles(rgb, rule).tolist()


def tile_selection(config: dict[str, Any], photo_ids: list[str], workers: int = 8) -> dict[str, dict[str, Any]]:
    """The tiles kept for each photo, cached in cache/compress/top_tiles.json. Only the photos asked for are
    opened, so a validation run never reads a test photo."""
    sources = {f: sha256(REPO_ROOT / f)
               for f in ("src/compress.py", "src/otsu.py", "src/embed.py", config["pixel_scale"]["decision_file"])}
    rule, path = {**rule_of(config), "sources": sources}, cache_folder("compress", write=True) / "top_tiles.json"
    cached = read_json(path) if path.exists() else {"rule": rule, "photos": {}}
    if cached["rule"] != rule:
        raise ValueError(f"{path}: made with another rule; archive it before rebuilding")
    todo = [p for p in photo_ids if p not in cached["photos"]]
    if todo:
        decision = read_json(REPO_ROOT / config["pixel_scale"]["decision_file"])
        root = dataset_dir(config, "primary")
        jobs = [(p, str(root / p), decision, rule_of(config)) for p in todo]
        if workers == 1:
            results = list(map(_select_one, jobs))
        else:
            with Pool(workers) as pool:
                results = list(pool.imap(_select_one, jobs, chunksize=8))
        for photo_id, n, tiles in results:
            cached["photos"][photo_id] = {"n": n, "tiles": tiles}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached))
    return {p: cached["photos"][p] for p in photo_ids}


def photo_tiles(path: str, decision: dict[str, Any], rule: dict[str, Any], top16: bool) -> np.ndarray:
    """Open a photo, bring it to the leaf scale and cut the tiles to describe: the 16 chosen, or all of them."""
    rgb = leaf_scale_photo(path, decision)
    return cut_selected(rgb, top_tiles(rgb, rule)) if top16 else tiling.cut(rgb, TILE)[0]


def describe_batches(encoder: nn.Module, tiles: np.ndarray, spec: dict[str, Any], config: dict[str, Any],
                     device: torch.device, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Class tokens and patch averages of all tiles, ``batch`` tiles per encoder pass."""
    parts = [describe(encoder, tiles[i:i + batch], spec, config, device) for i in range(0, len(tiles), batch)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def load_probe(method: str, round_: int):
    """The fitted probe (a scikit-learn pipeline) of one round."""
    import pickle

    with (get_default_runs_dir() / selected_runs(method)[round_ - 1] / "model.pkl").open("rb") as f:
        return pickle.load(f)   # our own local, checksum-verified checkpoint


def all_tiles(photo: str, n: int) -> None:
    return None


def even_policy(k: int) -> Policy:
    return lambda photo, n: even_tiles(n, k)


def selected_policy(selection: dict[str, dict[str, Any]]) -> Policy:
    def policy(photo: str, n: int) -> np.ndarray:
        entry = selection[photo]
        if entry["n"] != n:
            raise ValueError(f"{photo}: the tile grid has {n} tiles in the cache but {entry['n']} here")
        return np.array(entry["tiles"])
    return policy


# ---------------------------------------------------------------------------------------------- accuracy
@torch.inference_mode()
def head_probs(model: MILModel, bag: np.ndarray) -> np.ndarray:
    """Class probabilities of one photo from its tile descriptions (n, d)."""
    tiles = torch.from_numpy(np.asarray(bag, np.float32))[None]
    logits, _ = model(tiles, torch.ones(1, len(bag), dtype=torch.bool))
    return torch.softmax(logits, 1)[0].numpy()


def evaluate_rows(heads: list[MILModel], bag_of: Callable[[str], np.ndarray], folds: pd.DataFrame,
                  round_rows: dict[int, np.ndarray], policy: Policy, classes: list[str],
                  role: str) -> tuple[list[float], pd.DataFrame, float]:
    """Macro-F1 of each round, the predictions, and the mean number of tiles used, for the given rows. Every row
    must have ``role`` in its round: validation work cannot touch test photos and the other way round."""
    scores, tables, used, ids = [], [], [], folds["photo_id"].to_numpy()
    for r, model in zip(ROUNDS, heads):
        rows = round_rows[r]
        if not (folds[round_column(r)].to_numpy()[rows] == role).all():
            raise ValueError(f"round {r}: some of these photos are not {role} photos")
        probs = []
        for row in rows:
            photo = ids[row]
            bag = bag_of(photo)
            keep = policy(photo, len(bag))
            bag = bag if keep is None else bag[keep]
            used.append(len(bag))
            probs.append(head_probs(model, bag))
        table = prediction_table(folds, rows, np.array(probs), r, role, classes)
        scores.append(metrics.scores(table["true_class"], table["pred_class"], classes)["macro_f1"])
        tables.append(table)
    return scores, pd.concat(tables, ignore_index=True), float(np.mean(used))


def role_rows(folds: pd.DataFrame, role: str) -> dict[int, np.ndarray]:
    return {r: np.flatnonzero((folds[round_column(r)] == role).to_numpy()) for r in ROUNDS}


def cached_bags(config: dict[str, Any], encoder: str, descriptor: str, folds: pd.DataFrame,
                shared: dict | None = None) -> Callable:
    """Clean (version 0) tile descriptions of the cache of ``encoder`` as a function photo -> (n, d). Pass the same
    ``shared`` dict to every call of one job and each (encoder, descriptor) store is built once."""
    key = (encoder, descriptor)
    if shared is not None and key in shared:
        return shared[key]
    cache = CACHES[encoder]
    packed = {**config, "mil": {**config["mil"], "cache": cache, "versions": 1,
                                "override": {"descriptor": descriptor}}}
    pack_one_key_at_a_time(packed)    # builds the packed files one kind at a time
    store = train_mil.BagStore(cache, descriptor, sorted(folds["photo_id"]), 1)

    def bag_of(photo: str) -> np.ndarray:
        return store.bag(photo, 0)

    if shared is not None:
        shared[key] = bag_of
    return bag_of


class Variant:
    """Everything a variant needs before any photo is scored: its five heads, its descriptions and its tile policy."""

    def __init__(self, config: dict[str, Any], name: str, folds: pd.DataFrame, selection: dict | None,
                 shared: dict | None = None) -> None:
        head, kind = VARIANTS[name]
        loaded = [load_head(run) for run in head_runs(head)]
        self.models = [m for m, _ in loaded]
        self.bag_of = cached_bags(config, ENCODER[head], loaded[0][1]["descriptor"], folds, shared)
        if kind == "all":
            self.policy = all_tiles
        elif kind == "even16":
            self.policy = even_policy(config["compress"]["tiles"])
        elif selection is None:
            raise ValueError(f"{name} needs the tile selection")
        else:
            self.policy = selected_policy(selection)


def score_variant(variant: Variant, folds: pd.DataFrame, rows: dict[int, np.ndarray], classes: list[str],
                  role: str) -> dict[str, Any]:
    scores, table, tiles = evaluate_rows(variant.models, variant.bag_of, folds, rows, variant.policy, classes, role)
    return {"scores": scores, "predictions": table, "tiles_used": tiles}


def agreement_with_runs(table: pd.DataFrame, runs: list[str]) -> float:
    """Share of photos on which these predictions equal the validation predictions recorded in the runs."""
    recorded = pd.concat([pd.read_csv(get_default_runs_dir() / run / "predictions.csv") for run in runs])
    merged = table.merge(recorded, on=["round", "photo_id"], suffixes=("", "_recorded"))
    if len(merged) != len(table):
        raise ValueError("the recorded validation predictions do not cover these photos")
    return float((merged["pred_class"] == merged["pred_class_recorded"]).mean())


def fewer_tiles(config: dict[str, Any]) -> None:
    waiting = [n for n, (head, _) in VARIANTS.items()
               if not all(completed_run(run, ("metrics.json", "model.pt")) for run in head_runs(head))]
    if waiting:
        raise SystemExit(f"NOT DONE: no complete runs yet for {', '.join(waiting)}")
    require_clean_code()
    folds, checksum = read_folds(config)
    rows = role_rows(folds, "val")
    union = sorted({folds["photo_id"].iloc[r] for rs in rows.values() for r in rs})
    selection, shared = tile_selection(config, union), {}
    result = {name: score_variant(Variant(config, name, folds, selection, shared), folds, rows, config["classes"], "val")
              for name in VARIANTS}
    recorder = RunRecorder("cost_fewer_tiles", config=config, split_checksum=checksum, overwrite=True,
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    recorded = [read_json(get_default_runs_dir() / run / "metrics.json")["val_macro_f1"] for run in head_runs("attention")]
    gap = float(np.max(np.abs(np.array(recorded) - np.array(result["attention"]["scores"]))))
    agree = agreement_with_runs(result["attention"]["predictions"], head_runs("attention"))
    if gap > 0.005 or agree < 0.99:    # the same heads on the same descriptions must reproduce the recorded predictions
        raise RuntimeError(f"all tiles, attention: macro-F1 {gap:.3f} away from the runs, {100 * agree:.1f}% same predictions")
    summary = {}
    for name, entry in result.items():
        entry["predictions"].to_csv(recorder.run_dir / f"predictions_{name}.csv", index=False)
        scores = np.array(entry["scores"])
        summary[name] = {"val_macro_f1_rounds": scores.tolist(), "val_macro_f1_mean": float(scores.mean()),
                         "val_macro_f1_sd": float(scores.std(ddof=1)), "tiles_used": entry["tiles_used"]}
    recorder.save_metrics({"partition": "val", "variants": summary, "check_gap_attention_all_tiles": gap, "check_agreement_attention_all_tiles": agree,
                           "rule": rule_of(config)})
    recorder.finish()


# ---------------------------------------------------------------------------------------------- 8-bit encoder
def quantise(encoder: nn.Module) -> nn.Module:
    """Dynamic 8-bit weights for every linear layer (CPU); the input encoder is left as it is."""
    return torch.ao.quantization.quantize_dynamic(encoder, {nn.Linear}, dtype=torch.qint8)


def int8_subset(folds: pd.DataFrame, n: int, seed: int) -> list[str]:
    """A seeded, class-proportional sample of the photos that are validation photos in at least one round."""
    union = folds[(folds[[round_column(r) for r in ROUNDS]] == "val").any(axis=1)]
    sample = union.groupby("class", group_keys=False).sample(frac=n / len(union), random_state=seed)
    return sorted(sample["photo_id"])


def int8_describe(config: dict[str, Any]) -> None:
    from tqdm import tqdm

    require_clean_code()
    embed_config = load_config("configs/embed/e1_dinov3_vitb16.yaml")
    folds, _ = read_folds(config)
    settings = config["compress"]["int8"]
    subset = int8_subset(folds, settings["photos"], settings["seed"])
    folder = cache_folder(f"compress/{settings['cache']}", write=True)
    batch = config["cost"]["timing"]["tile_batch"]["cpu"]
    todo = [(p, 0) for p in subset if not (folder / f"{embed.photo_key(p)}.npz").exists()]
    spec = embed_config["encoders"]["E1"]
    model = quantise(load_encoder(embed_config, "E1", torch.device("cpu")))
    loader = torch.utils.data.DataLoader(embed.Jobs(embed_config, todo, embed.scale_factors(embed_config, subset)),
                                         batch_size=None, num_workers=4)
    for photo_id, _, tiles, boxes in tqdm(loader, total=len(todo), desc="int8"):
        cls, mean = describe_batches(model, tiles, spec, embed_config, torch.device("cpu"), batch)
        embed.save(folder / f"{embed.photo_key(photo_id)}.npz", cls.float(), mean.float(), np.asarray(boxes))


def int8_bags(folder, descriptor: str) -> Callable[[str], np.ndarray]:
    def bag(photo: str) -> np.ndarray:
        with np.load(folder / f"{embed.photo_key(photo)}.npz") as saved:
            return np.concatenate([saved[k].astype(np.float32) for k in PARTS[descriptor]], axis=1)
    return bag


def description_cosine(float_bag: Callable[[str], np.ndarray], int8_bag: Callable[[str], np.ndarray],
                       photos: list[str]) -> float:
    """Mean cosine similarity between the float and the 8-bit description of each tile (1.0: no damage)."""
    values = []
    for photo in photos:
        a, b = float_bag(photo).astype(np.float32), int8_bag(photo).astype(np.float32)
        values.append((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    return float(np.concatenate(values).mean())


def int8_score(config: dict[str, Any]) -> None:
    """The five attention round-models on the subset: with the cached float descriptions, and with 8-bit ones.
    A photo is scored only by the rounds in which it is a validation photo."""
    require_clean_code()
    folds, checksum = read_folds(config)
    settings = config["compress"]["int8"]
    subset = set(int8_subset(folds, settings["photos"], settings["seed"]))
    models = [load_head(run) for run in head_runs("attention")]
    descriptor = models[0][1]["descriptor"]
    rows = {r: np.array([i for i in rs if folds["photo_id"].iloc[i] in subset]) for r, rs in role_rows(folds, "val").items()}
    folder = cache_folder(f"compress/{settings['cache']}")
    heads = [m for m, _ in models]
    classes = config["classes"]
    float_bags, quantised_bags = cached_bags(config, "E1", descriptor, folds), int8_bags(folder, descriptor)
    float_scores, float_table, _ = evaluate_rows(heads, float_bags, folds, rows, all_tiles, classes, "val")
    int8_scores, int8_table, _ = evaluate_rows(heads, quantised_bags, folds, rows, all_tiles, classes, "val")
    recorder = RunRecorder("cost_int8", config=config, split_checksum=checksum, overwrite=True,
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    recorder.save_predictions(int8_table)
    float_table.to_csv(recorder.run_dir / "predictions_float.csv", index=False)
    recorder.save_metrics({"partition": "val", "photos": len(subset), "photo_rounds": len(int8_table),
                           "float_macro_f1_rounds": float_scores, "int8_macro_f1_rounds": int8_scores,
                           "float_macro_f1_mean": float(np.mean(float_scores)),
                           "int8_macro_f1_mean": float(np.mean(int8_scores)),
                           "description_cosine": description_cosine(float_bags, quantised_bags, sorted(subset)),
                           "agreement": float((float_table["pred_class"].to_numpy()
                                               == int8_table["pred_class"].to_numpy()).mean())})
    recorder.finish()


# ---------------------------------------------------------------------------------------------- ONNX
class DescribeModule(nn.Module):
    """Uint8-scaled tiles (N, 3, 224, 224) in [0, 1] -> descriptions (N, d): normalisation inside, then the encoder's
    class token and/or patch average (register tokens skipped) as ``describe`` does."""

    def __init__(self, encoder: nn.Module, mean: list[float], std: list[float], first_patch: int, descriptor: str):
        super().__init__()
        self.encoder, self.first_patch, self.descriptor = encoder, first_patch, descriptor
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(pixel_values=(pixels - self.mean) / self.std).last_hidden_state
        parts = {"cls": tokens[:, 0], "mean": tokens[:, self.first_patch:].mean(1)}
        return torch.cat([parts[k] for k in PARTS[self.descriptor]], dim=1)


class HeadModule(nn.Module):
    """A tile head as one graph: (1, N, d) descriptions and a (1, N) mask -> class probabilities."""

    def __init__(self, head: MILModel):
        super().__init__()
        self.head = head

    def forward(self, tiles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(tiles, mask)[0], dim=1)


def export_onnx(module: nn.Module, example: tuple, path, input_names: list[str], dynamic_axes: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module.eval(), example, str(path), input_names=input_names, output_names=["output"],
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False)


def onnx_session(path, threads: int):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def max_difference(module: nn.Module, session, inputs: dict[str, np.ndarray]) -> float:
    """Largest absolute difference between torch and onnxruntime outputs on the same inputs."""
    with torch.inference_mode():
        expected = module.eval()(*[torch.from_numpy(v) for v in inputs.values()]).numpy()
    return float(np.abs(expected - session.run(None, inputs)[0]).max())


def export(config: dict[str, Any]) -> None:
    """ONNX files of the final small model (round 1) in cache/compress/onnx/, and the torch-against-onnx check."""
    require_clean_code()
    selection = read_json(get_default_runs_dir() / "cost_selection.json")
    head, _ = VARIANTS[selection["selected"]]
    model, params = load_head(head_runs(head)[0])
    code, folder = ENCODER[head], cache_folder("compress/onnx", write=True)
    spec = config["encoders"][code]
    encoder = DescribeModule(load_encoder(config, code, torch.device("cpu")), config["normalisation"]["mean"],
                             config["normalisation"]["std"], 1 + spec["register_tokens"], params["descriptor"])
    pixels = np.random.default_rng(0).random((4, 3, TILE, TILE), dtype=np.float32)
    names = {"encoder": folder / f"{selection['selected']}_encoder.onnx", "head": folder / f"{selection['selected']}_head.onnx"}
    export_onnx(encoder, (torch.from_numpy(pixels),), names["encoder"], ["pixels"], {"pixels": {0: "tiles"}, "output": {0: "tiles"}})
    dim = model.project[0].in_features
    tiles = np.random.default_rng(1).standard_normal((1, 7, dim)).astype(np.float32)
    mask = np.ones((1, 7), bool)
    head_module = HeadModule(model)
    export_onnx(head_module, (torch.from_numpy(tiles), torch.from_numpy(mask)), names["head"], ["tiles", "mask"],
                {"tiles": {1: "n"}, "mask": {1: "n"}})
    threads = config["cost"]["timing"]["cpu_threads"]
    sessions = {k: onnx_session(p, threads) for k, p in names.items()}
    rng = np.random.default_rng(2)
    more = rng.random((16, 3, TILE, TILE), dtype=np.float32)         # the batch the timing uses on the CPU
    many = rng.standard_normal((1, 79, dim)).astype(np.float32)      # a whole photo, with a partly empty mask
    partial = np.ones((1, 79), bool)
    partial[0, 60:] = False
    gaps = {"encoder": max(max_difference(encoder, sessions["encoder"], {"pixels": x}) for x in (pixels, more)),
            "head": max(max_difference(head_module, sessions["head"], inputs)
                        for inputs in ({"tiles": tiles, "mask": mask}, {"tiles": many, "mask": partial}))}
    recorder = RunRecorder("cost_export", config=config, overwrite=True,
                           data_version={"primary": config["datasets"]["primary"]["version"]})
    recorder.save_metrics({"partition": "none", "candidate": selection["selected"], "max_abs_difference": gaps,
                           "bytes": {k: p.stat().st_size for k, p in names.items()}, "opset": 17})
    recorder.finish()
    if max(gaps.values()) > 1e-3:
        raise RuntimeError(f"the ONNX files differ from torch by {gaps}")


ACTIONS = {"fewer-tiles": fewer_tiles, "int8-describe": int8_describe, "int8-score": int8_score, "export": export}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=ACTIONS)
    args = parser.parse_args()
    ACTIONS[args.action](load_config("configs/cost.yaml"))
    print(f"COMPRESS {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
