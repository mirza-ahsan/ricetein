"""Stage 12: does the attention head look at real symptoms? (handbook Stage 12; declared in §24, 2026-11-20)

    uv run python -m src.explain select   [--partition val|test|paddy]  # rule-based example list (CSV first)
    uv run python -m src.explain heatmaps [--partition val|test|paddy]  # labelled and blind heat maps
    uv run python -m src.explain plan                                   # freeze the declaration (once)
    uv run python -m src.explain deletion   --partition test|val|paddy  # remove top-k / random-k tiles
    uv run python -m src.explain background --partition test|val|paddy  # attention on background tiles
    uv run python -m src.explain report                                 # tables, figure and report

Models: the five session-split attention runs (``stage7_session_attention_r1..r5``, tested once) and, on
validation photos only, their two stability seeds. Every action is resumable and writes only under
``results/``, ``docs/reports/``, ``data/heatmap_ratings.csv`` (header only, created once) and (the
declaration and the run-once marker) ``runs/``. Test photos are read
by ``deletion`` and ``background`` only, after ``plan`` froze what is measured.

``--partition paddy`` (extension declared in handbook §24, 2026-11-21) treats Paddy Doctor photos like test
photos: the same five models on the native-zoom cache ``paddy_e1_native``, after ``plan``, under run-once
markers ``runs/explain_paddy_<action>_started.json``. Paddy has no capture blocks: its verdict uses the photo
range only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import followup_report, metrics, otsu, stats, tiling
from src import paddy_zero_shot as pz
from src import train_mil as tm
from src.config import REPO_ROOT, cache_folder, dataset_dir, load_config
from src.embed import load_version, photo_key, scale_factors, version_settings
from src.evaluate import checkpoint, digest
from src.inventory import read_inventory
from src.inventory_report import md_table
from src.mil_models import MILModel
from src.paddy_embed import TILE as PADDY_TILE
from src.paddy_embed import verify as verify_paddy_cache
from src.records import get_default_runs_dir, require_clean_code
from src.seeding import seed_everything
from src.split import read_folds

EMBED_CONFIG = "configs/embed/e1_dinov3_vitb16.yaml"
SPLIT_CONFIG = "configs/mil/session_attention.yaml"  # for the session-blocked fold file
OUT = REPO_ROOT / "results/explain"
TABLES = REPO_ROOT / "results/tables"
FIGURES = REPO_ROOT / "results/figures"
REPORT = REPO_ROOT / "docs/reports/explain_report.md"
RATINGS = REPO_ROOT / "data/heatmap_ratings.csv"
RATING_COLUMNS = ["item_id", "rater", "rater_kind", "rating", "note"]  # rating: yes / partly / no
ANSWERS = ("yes", "partly", "no")
RATER_KINDS = ("model", "human")

PADDY_CONFIG = "configs/paddy_zero_shot.yaml"
PADDY_ZOOM = "native"  # the Stage 10 headline zoom; the only one used here
PADDY_TOLERANCE = 1e-3  # no-removal probabilities must match the saved Stage 10 ones to this
SEED_TAGS = {0: "", 1: "_s1", 2: "_s2"}  # model training seeds; only seed 0 was tested
ROUNDS = (1, 2, 3, 4, 5)
KS = (1, 2, 4, 8, 16)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
PER_CELL = 3  # examples per class and category
EXTRA_ITEMS = 14  # 4 classes x 3 categories x 3 = 36 rule examples, plus 14 random ones = 50 rated heat maps
SELECT_SEED = 0
BACKGROUND_SHARE = 0.9  # a tile is background when more than this share of its pixels is
CATEGORIES = ("correct", "wrong", "random")
HEATMAP_LONG_SIDE = 900  # pixels of the longer side of a drawn heat map
# A leaf mask covering a share of the photo outside this range is only flagged, never changed: a thin leaf
# on a plain background can be a few per cent, but an inverted side guess gives a leaf share near 1 - real.
MASK_SANE = (0.02, 0.60)


def run_name(seed: int, round_: int) -> str:
    return f"stage7_session_attention{SEED_TAGS[seed]}_r{round_}"


def runs_for(partition: str) -> list[str]:
    """Test and Paddy: only the five runs that were tested. Validation: all three training seeds."""
    seeds = (0,) if partition in ("test", "paddy") else tuple(SEED_TAGS)
    return [run_name(s, r) for s in seeds for r in ROUNDS]


def write_once(path: Path, content: dict) -> None:
    """Create ``path`` whole or not at all; never replace an existing one (markers are never rewritten)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(content, indent=2))
    try:
        os.link(partial, path)  # fails if ``path`` exists
    finally:
        partial.unlink()


def atomic(path: Path, write) -> None:
    """``write(partial_path)`` beside ``path``, then rename: a killed run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp" + path.suffix)
    write(partial)
    os.replace(partial, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic(path, lambda partial: frame.to_csv(partial, index=False))


# ---------------------------------------------------------------- example selection (pure)


def rank_examples(predictions: pd.DataFrame, classes: list[str], per_cell: int, seed: int) -> pd.DataFrame:
    """Per true class: the most confident correct, the most confident wrong, and random photos.

    ``predictions`` has photo_id, round, true_class, pred_class and prob_<class>. A photo that appears in
    several rounds is taken once (its first pick wins); confidence is the probability of the predicted class.
    """
    table = predictions.copy()
    probs = table[[f"prob_{c}" for c in classes]].to_numpy()
    true_index = table["true_class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    table["confidence"] = probs.max(axis=1)
    table["prob_true"] = probs[np.arange(len(table)), true_index]
    table = table.sort_values(["confidence", "photo_id", "round"], ascending=[False, True, True])
    first = table.sort_values(["photo_id", "round"]).drop_duplicates("photo_id")
    chosen: list[pd.DataFrame] = []
    taken: set[str] = set()
    for k, name in enumerate(classes):
        rows = table[table["true_class"] == name]
        for category, pool in (
            ("correct", rows[rows["pred_class"] == name]),
            ("wrong", rows[rows["pred_class"] != name]),
        ):
            picked = pool[~pool["photo_id"].isin(taken)].drop_duplicates("photo_id").head(per_cell)
            chosen.append(picked.assign(category=category))
            taken |= set(picked["photo_id"])
        pool = first[(first["true_class"] == name) & ~first["photo_id"].isin(taken)].sort_values("photo_id")
        rng = np.random.default_rng([seed, k])
        picked = pool.iloc[np.sort(rng.choice(len(pool), min(per_cell, len(pool)), replace=False))]
        chosen.append(picked.assign(category="random"))
        taken |= set(picked["photo_id"])
    out = pd.concat(chosen, ignore_index=True)
    out["rank"] = out.groupby(["true_class", "category"]).cumcount() + 1
    return out[
        ["true_class", "category", "rank", "photo_id", "round", "pred_class", "confidence", "prob_true"]
    ]


def add_extras(examples: pd.DataFrame, predictions: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Add ``n`` random photos (not yet chosen) and give all items a shuffled, blind ``item_id``."""
    first = predictions.sort_values(["photo_id", "round"]).drop_duplicates("photo_id")
    pool = first[~first["photo_id"].isin(examples["photo_id"])].reset_index(drop=True)
    rng = np.random.default_rng([seed, 99])
    extra = pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))].assign(category="extra", rank=0)
    out = pd.concat(
        [examples, extra[["true_class", "category", "rank", "photo_id", "round", "pred_class"]]],
        ignore_index=True,
    )
    out["item_id"] = [f"H{i:02d}" for i in rng.permutation(len(out)) + 1]
    return out


# ---------------------------------------------------------------- deletion (pure on tensors)


def top_k_mask(weights: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """The real-tile mask with the ``k`` most-attended real tiles switched off."""
    check_removable(mask, k)
    top = weights.masked_fill(~mask, -1.0).topk(k, dim=1).indices
    return mask.clone().scatter_(1, top, False)


def random_k_mask(mask: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """The mask with ``k`` randomly chosen real tiles switched off (padding is never chosen)."""
    check_removable(mask, k)
    noise = torch.rand(mask.shape, generator=generator).to(mask.device).masked_fill(~mask, -1.0)
    return mask.clone().scatter_(1, noise.topk(k, dim=1).indices, False)


def check_removable(mask: torch.Tensor, k: int) -> None:
    if (mask.sum(1) <= k).any():
        raise ValueError(f"cannot remove {k} tiles from a bag with {int(mask.sum(1).min())} real tiles")


@torch.inference_mode()
def delete_and_score(
    model: MILModel,
    bags,
    device: torch.device,
    stored: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """One row per photo, condition (none/top/random), k and random seed: the prediction and true-class
    probability after the tiles are removed. ``stored`` (saved attention) must match the recomputed one."""
    model.eval()
    generators = {s: torch.Generator().manual_seed(s) for s in RANDOM_SEEDS}
    parts = []

    def record(condition, k, seed, ids, y, mask, tiles):
        probs = torch.softmax(model(tiles, mask)[0], 1).cpu().numpy()
        parts.append(
            pd.DataFrame(
                {
                    "photo_id": ids,
                    "condition": condition,
                    "k": k,
                    "seed": seed,
                    "true": y,
                    "pred": probs.argmax(1),
                    "p_true": probs[np.arange(len(y)), y],
                }
            )
        )

    for tiles, mask, labels, rows in bags:
        tiles, mask, y = tiles.to(device), mask.to(device), labels.numpy()
        ids = bags.photo_ids[rows]
        _, weights = model(tiles, mask)
        if stored is not None:
            for i, photo_id in enumerate(ids):
                n = int(mask[i].sum())
                if not np.allclose(weights[i, :n].cpu().numpy(), stored[photo_id], atol=1e-4, rtol=1e-2):
                    raise ValueError(f"{photo_id}: recomputed attention differs from the saved attention")
        record("none", 0, -1, ids, y, mask, tiles)
        for k in KS:
            record("top", k, -1, ids, y, top_k_mask(weights, mask, k), tiles)
            for seed in RANDOM_SEEDS:
                record("random", k, seed, ids, y, random_k_mask(mask, k, generators[seed]), tiles)
    return pd.concat(parts, ignore_index=True)


def score_conditions(long: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Accuracy, macro-F1 and mean true-class probability per condition, k and random seed."""
    rows = []
    for (condition, k, seed), g in long.groupby(["condition", "k", "seed"]):
        s = metrics.scores(np.array(classes)[g["true"]], np.array(classes)[g["pred"]], classes)
        rows.append(
            {
                "condition": condition,
                "k": k,
                "seed": seed,
                "n": s["n"],
                "accuracy": s["accuracy"],
                "macro_f1": s["macro_f1"],
                "p_true": g["p_true"].mean(),
            }
        )
    return pd.DataFrame(rows)


def photo_table(long: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Per photo and k: true-class probability after top-k removal, and after random-k removal (seed mean)."""
    full = long[long["condition"] == "none"].set_index("photo_id")
    top = long[long["condition"] == "top"].set_index(["photo_id", "k"])
    random = long[long["condition"] == "random"].groupby(["photo_id", "k"])["p_true"].mean()
    out = top[["p_true"]].rename(columns={"p_true": "p_top"}).join(random.rename("p_random"))
    out = out.reset_index()
    out["p_full"] = full.loc[out["photo_id"], "p_true"].to_numpy()
    out["true_class"] = np.array(classes)[full.loc[out["photo_id"], "true"].to_numpy()]
    return out


# ---------------------------------------------------------------- background (pure)


def tile_background_fraction(leaf: np.ndarray, boxes: np.ndarray, tile: int) -> np.ndarray:
    """Share of each tile's pixels that are background (leaf mask False); padding counts as background."""
    height, width = leaf.shape
    total = np.pad((~leaf).astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = np.minimum(x0 + tile, width), np.minimum(y0 + tile, height)
    background = total[y1, x1] - total[y0, x1] - total[y1, x0] + total[y0, x0]
    outside = tile * tile - (x1 - x0) * (y1 - y0)
    return (background + outside) / (tile * tile)


def background_share(weights: np.ndarray, fraction: np.ndarray) -> dict:
    """Share of attention, and share of tiles, on tiles that are more than 90% background."""
    is_background = fraction > BACKGROUND_SHARE
    return {
        "attention_share": float(weights[is_background].sum() / weights.sum()),
        "tile_share": float(is_background.mean()),
    }


# ---------------------------------------------------------------- heat maps


def paint(
    boxes: np.ndarray, weights: np.ndarray, shape: tuple[int, int], tile: int, scale: float
) -> np.ndarray:
    """Attention laid back onto the tile grid: each pixel is the mean weight of the tiles covering it,
    scaled so the hottest pixel is 1. ``shape`` is the (height, width) of the photo; ``scale`` shrinks it."""
    height, width = round(shape[0] * scale), round(shape[1] * scale)
    heat, count = np.zeros((height, width)), np.zeros((height, width))
    for (x, y, _, _), w in zip(boxes, weights):
        ys, xs = (
            slice(round(y * scale), round((y + tile) * scale)),
            slice(round(x * scale), round((x + tile) * scale)),
        )
        heat[ys, xs] += w
        count[ys, xs] += 1
    heat = heat / np.maximum(count, 1)
    return heat / heat.max()


def check_grid(rgb: np.ndarray, boxes: np.ndarray, tile: int) -> None:
    if not np.array_equal(tiling.boxes(rgb.shape[1], rgb.shape[0], tile), boxes):
        raise ValueError("the rescaled photo does not give the cache's tile grid")


def draw_heatmap(
    rgb: np.ndarray,
    boxes: np.ndarray,
    weights: np.ndarray,
    tile: int,
    path: Path,
    title: str,
) -> None:
    """Photo and overlay side by side. Works for any photo, boxes and weights (primary or Paddy Doctor)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    scale = HEATMAP_LONG_SIDE / max(rgb.shape[:2])
    small = np.asarray(
        Image.fromarray(rgb).resize((round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
    )
    heat = paint(boxes, weights, rgb.shape[:2], tile, scale)[: small.shape[0], : small.shape[1]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 5 * small.shape[0] / small.shape[1] + 0.6))
    axes[0].imshow(small)
    axes[1].imshow(small)
    image = axes[1].imshow(np.ma.masked_less(heat, 0.02), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    for axis in axes:
        axis.axis("off")
    figure.colorbar(image, ax=axes[1], fraction=0.04, label="attention (hottest = 1)")
    figure.suptitle(title, fontsize=9)
    atomic(path, lambda partial: figure.savefig(partial, dpi=110, bbox_inches="tight"))
    plt.close(figure)


# ---------------------------------------------------------------- files of the runs


def read_run_predictions(partition: str) -> pd.DataFrame:
    """Seed-0 predictions of the five rounds on ``partition`` (Paddy: Stage 10 ones, native zoom)."""
    name = {"test": "test_predictions.csv", "paddy": f"paddy_predictions_{PADDY_ZOOM}.csv"}.get(
        partition, "predictions.csv"
    )
    root = get_default_runs_dir()
    return pd.concat([pd.read_csv(root / run_name(0, r) / name) for r in ROUNDS], ignore_index=True)


def read_attention(run: str, partition: str) -> dict[str, np.ndarray]:
    path = get_default_runs_dir() / run / ("attention.npz" if partition == "test" else "attention_val.npz")
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def require_plan() -> Path:
    path = get_default_runs_dir() / "explain_plan.json"
    if not path.exists():
        raise RuntimeError(
            "run `python -m src.explain plan` first: test photos are read only after the declaration"
        )
    return path


def declaration_for(partition: str) -> dict:
    """The declaration, required before any test or Paddy photo is read; Paddy needs its extension in it."""
    declaration = json.loads(require_plan().read_text())
    if partition == "paddy" and "paddy" not in declaration:
        raise RuntimeError(
            "the declaration has no Paddy extension (handbook §24, 2026-11-21): it was written before it; "
            "a new dated declaration is needed, because the plan file is never rewritten"
        )
    return declaration


class Photos:
    """Photo pixels at the common leaf scale (as the cache saw them) and the cache's tile boxes.

    ``Photos("paddy")``: Paddy Doctor photos at the native zoom, from ``paddy_e1_native``."""

    def __init__(self, partition: str = "primary") -> None:
        if partition == "paddy":
            self.config = load_config(PADDY_CONFIG)
            self.root = dataset_dir(self.config, "paddy")
            self.tile = PADDY_TILE
            self.cache = cache_folder(self.config["paddy"]["caches"]["tiles"][PADDY_ZOOM])
            self.factors = dict.fromkeys(
                pz.photo_ids(self.config), self.config["paddy"]["zoom_factors"][PADDY_ZOOM]
            )
            return
        self.config = load_config(EMBED_CONFIG)
        self.root = dataset_dir(self.config, "primary")
        self.tile = self.config["embed"]["tile"]
        self.cache = cache_folder(self.config["embed"]["name"])
        inventory = read_inventory(self.config["inventory"]["files"]["primary"])
        self.factors = scale_factors(self.config, sorted(inventory.loc[inventory["opened_ok"], "photo_id"]))

    def rgb(self, photo_id: str) -> np.ndarray:
        change = version_settings({}, self.config["seed"], 0, photo_id)
        return load_version(str(self.root / photo_id), self.factors[photo_id], change)

    def boxes(self, photo_id: str) -> np.ndarray:
        with np.load(self.cache / "v0" / f"{photo_key(photo_id)}.npz") as saved:
            return saved["boxes"]


# ---------------------------------------------------------------- Paddy Doctor photos


class PaddyBags:
    """Paddy photos' tile descriptions from the native-zoom cache, in the layout ``delete_and_score`` reads:
    iterating gives (tiles, mask, labels, rows) and ``photo_ids[rows]`` names the photos."""

    def __init__(self, config: dict, parts: list[str], ids: list[str] | None = None, batch: int = 8) -> None:
        self.folder = cache_folder(config["paddy"]["caches"]["tiles"][PADDY_ZOOM])
        truth = pz.truth(config).set_index("photo_id")["class"]
        self.photo_ids = np.array(pz.photo_ids(config) if ids is None else ids)
        self.labels = torch.tensor([config["classes"].index(truth[p]) for p in self.photo_ids])
        self.parts, self.batch = parts, batch

    def __iter__(self):
        for start in range(0, len(self.photo_ids), self.batch):
            rows = np.arange(start, min(start + self.batch, len(self.photo_ids)))
            bags = []
            for photo_id in self.photo_ids[rows]:
                with np.load(self.folder / "v0" / f"{photo_key(photo_id)}.npz") as saved:
                    bags.append(np.concatenate([saved[k].astype(np.float32) for k in self.parts], axis=1))
            tiles = torch.zeros(len(bags), max(map(len, bags)), bags[0].shape[1])
            mask = torch.zeros(tiles.shape[:2], dtype=torch.bool)
            for i, bag in enumerate(bags):
                tiles[i, : len(bag)], mask[i, : len(bag)] = torch.from_numpy(bag), True
            yield tiles, mask, self.labels[rows], rows


def check_paddy_predictions(declaration: dict) -> None:
    """The saved Stage 10 predictions (they pick examples, check deletion) are the declared ones."""
    for name, saved in declaration["paddy"]["predictions_sha256"].items():
        if digest(get_default_runs_dir() / name / f"paddy_predictions_{PADDY_ZOOM}.csv") != saved:
            raise ValueError(f"{name}: Paddy predictions changed after the declaration")


def paddy_setup(declaration: dict) -> dict:
    """The Paddy settings, after checking that the cache and the saved predictions are the declared ones."""
    check_paddy_predictions(declaration)
    config = load_config(PADDY_CONFIG)
    manifest = verify_paddy_cache(config, config["paddy"]["caches"]["tiles"][PADDY_ZOOM])
    if manifest["spec_sha256"] != declaration["paddy"]["cache_spec_sha256"]:
        raise ValueError("the Paddy description cache is not the one that was declared")
    return config


def paddy_run(name: str, declaration: dict, config: dict, device: torch.device, ids: list[str] | None = None):
    """One declared model with its Paddy bags (``ids`` limits the photos)."""
    check_declared(name, declaration)
    model, parts = pz.load_head(name, config["encoders"][config["paddy"]["encoder"]]["dim"], device)
    return model, PaddyBags(config, parts, ids)


@torch.inference_mode()
def attention_of(model: MILModel, bags, device: torch.device) -> dict[str, np.ndarray]:
    """Attention weights over the real tiles of every photo of ``bags``."""
    out = {}
    for tiles, mask, _, rows in bags:
        weights = model(tiles.to(device), mask.to(device))[1].cpu()
        for i, row in enumerate(rows):
            out[bags.photo_ids[row]] = weights[i, : int(mask[i].sum())].numpy()
    return out


def check_against_stage10(long: pd.DataFrame, name: str, classes: list[str]) -> None:
    """With nothing removed, each photo's true-class probability must be the one Stage 10 saved."""
    saved = pd.read_csv(get_default_runs_dir() / name / f"paddy_predictions_{PADDY_ZOOM}.csv").set_index(
        "photo_id"
    )
    none = long[long["condition"] == "none"].set_index("photo_id")
    probs = saved.loc[none.index, [f"prob_{c}" for c in classes]].to_numpy()
    if not np.allclose(
        none["p_true"].to_numpy(), probs[np.arange(len(none)), none["true"].to_numpy()], atol=PADDY_TOLERANCE
    ):
        raise ValueError(
            f"{name}: probabilities with nothing removed differ from the saved Stage 10 predictions"
        )


# ---------------------------------------------------------------- actions


def select(partition: str) -> None:
    if partition in ("test", "paddy"):
        declaration = declaration_for(partition)
    config = load_config(EMBED_CONFIG)
    classes = config["classes"]
    if partition == "paddy":
        check_paddy_predictions(declaration)
    predictions = read_run_predictions(partition)
    extra = 0 if partition == "paddy" else EXTRA_ITEMS  # Paddy: the 36 labelled examples only, no blind sheet
    examples = rank_examples(predictions, classes, PER_CELL, SELECT_SEED)
    if partition != "paddy":
        examples = add_extras(examples, predictions, extra, SELECT_SEED)
    path = OUT / f"examples_{partition}.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), examples, check_dtype=False, atol=1e-6)  # same rule
    else:
        write_csv(examples, path)
    if not RATINGS.exists():
        write_csv(pd.DataFrame(columns=RATING_COLUMNS), RATINGS)
    wanted = len(classes) * len(CATEGORIES) * PER_CELL + extra
    if len(examples) != wanted:
        print(f"WARNING: {len(examples)} items instead of {wanted} (a class has too few wrong photos?)")
    print(f"{len(examples)} examples ({len(examples) - extra} by rule, {extra} extra) -> {path}")


def heatmaps(partition: str) -> None:
    paddy = partition == "paddy"
    if partition in ("test", "paddy"):
        declaration = declaration_for(partition)
    if paddy:
        select(partition)  # the list on disk must be the one the rule gives from the declared predictions
    examples = pd.read_csv(OUT / f"examples_{partition}.csv")
    photos = Photos(partition)
    if paddy:  # no saved attention for Paddy photos: the declared model is run on the chosen photos' bags
        config, device, attention = (
            paddy_setup(declaration),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            {},
        )
        for r, chosen in examples.groupby("round"):
            model, bags = paddy_run(run_name(0, r), declaration, config, device, list(chosen["photo_id"]))
            attention[r] = attention_of(model, bags, device)
    else:
        attention = {r: read_attention(run_name(0, r), partition) for r in ROUNDS}
    for row in examples.itertuples():
        labelled = (
            OUT
            / "heatmaps"
            / partition
            / f"{row.true_class}_{row.category}_{row.rank}_{photo_key(row.photo_id)}.png"
        )
        blind = None if paddy else OUT / "blind" / partition / f"{row.item_id}.png"
        if (blind is None or blind.exists()) and (labelled.exists() or row.category == "extra"):
            continue
        weights, boxes, rgb = (
            attention[row.round][row.photo_id],
            photos.boxes(row.photo_id),
            photos.rgb(row.photo_id),
        )
        check_grid(rgb, boxes, photos.tile)
        if blind is not None:
            draw_heatmap(rgb, boxes, weights, photos.tile, blind, row.item_id)
        if row.category != "extra":
            title = (
                f"{row.true_class}, {row.category} #{row.rank}: predicted {row.pred_class} "
                f"({row.confidence:.2f}); {row.photo_id}, round {row.round}; peak tile "
                f"{weights.max() * len(weights):.1f}x uniform"
            )
            draw_heatmap(rgb, boxes, weights, photos.tile, labelled, title)
    print(
        f"heat maps -> {OUT / 'heatmaps' / partition}"
        + ("" if paddy else f" and {OUT / 'blind' / partition}")
    )


def plan() -> None:
    version = require_clean_code()
    base = load_config("configs/base.yaml")
    root = get_default_runs_dir()
    path = root / "explain_plan.json"
    if path.exists():
        print(f"{path} already declared; not changed")
        return
    paddy_config = load_config(PADDY_CONFIG)
    manifest = verify_paddy_cache(paddy_config, paddy_config["paddy"]["caches"]["tiles"][PADDY_ZOOM])
    runs = {}
    for name in runs_for("val"):
        tested = (root / name / "test_predictions.csv").exists()
        runs[name] = {
            "tested": tested,
            "record_sha256": digest(root / name / "record.json"),
            "checkpoint_sha256": digest(checkpoint(root / name)),
        }
    declaration = {
        "declared": time.strftime("%Y-%m-%d"),
        "code_version": version,
        "deletion": {
            "ks": list(KS),
            "random_seeds": list(RANDOM_SEEDS),
            "partitions": ["test", "val", "paddy"],
            "ranking": "the model's own attention on the full bag, taken once",
            "removal": "switch the tiles off in the real-tile mask (equal to removing them)",
            "scores": ["accuracy", "macro_f1", "true-class probability"],
        },
        "background": {
            "threshold": BACKGROUND_SHARE,
            "mask": "Otsu leaf mask of the photo at the common leaf scale, a measuring tool only",
            "mask_flag_outside": list(MASK_SANE),
            "partitions": ["test", "val", "paddy"],
        },
        "verdict": {
            "rule": "top-k matters more than random-k only if the photo-bootstrap and the capture-block-"
            "bootstrap ranges of the true-class-probability gap are both above 0 and the gap is positive "
            "in at least min_round_wins rounds",
            "bootstrap": base["evaluation"]["bootstrap"],
            "comparison": base["evaluation"]["comparison"],
        },
        "examples": {"per_cell": PER_CELL, "extra": EXTRA_ITEMS, "seed": SELECT_SEED},
        "paddy": {  # extension declared in handbook §24, 2026-11-21: same models and measures
            "cache": paddy_config["paddy"]["caches"]["tiles"][PADDY_ZOOM],
            "zoom": PADDY_ZOOM,
            "cache_spec_sha256": manifest["spec_sha256"],
            "photos": manifest["photos"],
            "split_sha256": digest(REPO_ROOT / paddy_config["paddy"]["split_file"]),
            "bootstrap": "photo only: no capture blocks; a photo's gap is averaged over the five models",
            "predictions_sha256": {
                name: digest(root / name / f"paddy_predictions_{PADDY_ZOOM}.csv")
                for name in runs_for("paddy")
            },
        },
        "explain_sha256": digest(Path(__file__)),
        "test_runs": [n for n, v in runs.items() if v["tested"]],
        "runs": runs,
    }
    write_once(path, declaration)
    print(f"declared {path}")


def start_marker(partition: str, action: str) -> dict:
    """The declaration, and (test, Paddy) the run-once marker: a later call may resume, never restart."""
    plan_path = require_plan()
    declaration = declaration_for(partition)
    version = require_clean_code()
    if partition in ("test", "paddy"):
        stem = f"explain_{action}" if partition == "test" else f"explain_paddy_{action}"
        marker, plan_sha = get_default_runs_dir() / f"{stem}_started.json", digest(plan_path)
        if not marker.exists():
            write_once(marker, {"evaluation_code": version, "plan_sha256": plan_sha})
        elif json.loads(marker.read_text())["plan_sha256"] != plan_sha:
            raise ValueError(f"{marker}: started under another plan")
    return declaration


def names_for(partition: str, declaration: dict) -> list[str]:
    """Test and Paddy photos: only the runs declared as tested. Validation: all three training seeds."""
    return declaration["test_runs"] if partition in ("test", "paddy") else runs_for("val")


def check_declared(name: str, declaration: dict) -> None:
    root, expected = get_default_runs_dir(), declaration["runs"][name]
    if (
        digest(root / name / "record.json") != expected["record_sha256"]
        or digest(checkpoint(root / name)) != expected["checkpoint_sha256"]
    ):
        raise ValueError(f"{name}: changed after the declaration")


def load_run(name: str, declaration: dict):
    """Model, folds and config of one run, after checking the files are the ones that were declared."""
    root = get_default_runs_dir()
    check_declared(name, declaration)
    record = json.loads((root / name / "record.json").read_text())
    config = record["config"]
    seed_everything(config.get("run_seed", config["seed"]), config["determinism"])
    folds, _ = read_folds(config)
    mil, params = config["mil"], config["params"]
    store = tm.store_for({**config, "mil": {**mil, "versions": 1}}, folds, params["descriptor"])
    model = MILModel(store.dim, params["width"], params["dropout"], mil["aggregator"], len(config["classes"]))
    model.load_state_dict(torch.load(root / name / "model.pt", map_location="cpu", weights_only=True))
    return model, store, folds, config


def deletion(partition: str) -> None:
    declaration = start_marker(partition, "deletion")
    names = names_for(partition, declaration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paddy = partition == "paddy"
    paddy_config = paddy_setup(declaration) if paddy else None
    for name in names:
        scores_path, photos_path = (
            OUT / f"deletion_{partition}_{name}.csv",
            OUT / f"deletion_photos_{partition}_{name}.csv",
        )
        if scores_path.exists() and photos_path.exists():
            continue
        if paddy:
            model, bags = paddy_run(name, declaration, paddy_config, device)
            r, classes = pz.record(name)["round"], paddy_config["classes"]
            long = delete_and_score(model, bags, device)  # no saved attention to compare with
            check_against_stage10(long, name, classes)
        else:
            model, store, folds, config = load_run(name, declaration)
            r, classes = config["round"], config["classes"]
            bags = tm.Bags(
                store, folds, r, partition, classes, 1, 0.0, config["mil"]["batch"], config["seed"]
            )
            stored = read_attention(name, partition)
            long = delete_and_score(model.to(device), bags, device, stored)
        write_csv(photo_table(long, classes).assign(run=name, round=r), photos_path)
        write_csv(score_conditions(long, classes).assign(run=name, round=r), scores_path)
        print(f"{name}: deletion test on {partition} photos done", flush=True)


def photo_background(photos: Photos, photo_id: str, folder: Path) -> tuple[np.ndarray, float]:
    """Background share of every tile of one photo, and the leaf share of the photo; cached per photo."""
    path = folder / f"{photo_key(photo_id)}.npz"
    if not path.exists():
        rgb, boxes = photos.rgb(photo_id), photos.boxes(photo_id)
        check_grid(rgb, boxes, photos.tile)
        leaf = otsu.leaf_mask(rgb)
        fraction = tile_background_fraction(leaf, boxes, photos.tile).astype(np.float32)
        atomic(path, lambda partial: np.savez(partial, fraction=fraction, leaf_share=leaf.mean()))
    with np.load(path) as saved:
        return saved["fraction"], float(saved["leaf_share"])


def background(partition: str) -> None:
    declaration = start_marker(partition, "background")
    names = names_for(partition, declaration)
    paddy = partition == "paddy"
    photos, folder = (
        Photos(partition),
        OUT / ("background_fraction_paddy" if paddy else "background_fraction"),
    )
    if paddy:
        device, paddy_config, attention = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            paddy_setup(declaration),
            {},
        )
        for name in names:
            model, bags = paddy_run(name, declaration, paddy_config, device)
            attention[name] = attention_of(model, bags, device)
    else:
        attention = {name: read_attention(name, partition) for name in names}
    wanted = sorted({photo_id for weights in attention.values() for photo_id in weights})
    with ThreadPoolExecutor(max_workers=8) as pool:  # decoding and the Otsu mask release the GIL
        measured = dict(
            zip(wanted, pool.map(lambda photo_id: photo_background(photos, photo_id, folder), wanted))
        )
    classes = (pz.truth(paddy_config) if paddy else read_folds(load_config(SPLIT_CONFIG))[0]).set_index(
        "photo_id"
    )["class"]
    rows = []
    for name in names:
        for photo_id, weights in attention[name].items():
            fraction, leaf_share = measured[photo_id]
            rows.append(
                {
                    "run": name,
                    "round": int(name.rsplit("_r", 1)[1]),
                    "photo_id": photo_id,
                    "true_class": classes[photo_id],
                    **background_share(weights, fraction),
                    "n_tiles": len(weights),
                    "leaf_share": leaf_share,
                    "mask_suspect": not MASK_SANE[0] < leaf_share < MASK_SANE[1],
                }
            )
    write_csv(pd.DataFrame(rows), OUT / f"background_{partition}.csv")
    print(f"background score on {partition} photos done: {len(wanted)} photos, {len(names)} runs")


# ---------------------------------------------------------------- report


def seed0(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of the seed-0 runs (the headline models); the `_s1`, `_s2` runs are stability checks."""
    return frame[~frame["run"].str.contains("_s[12]_")]


def read_all(pattern: str) -> pd.DataFrame | None:
    files = sorted(OUT.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def bootstrap_range(
    values: np.ndarray, index_sets: list[np.ndarray], confidence: float
) -> tuple[float, float]:
    """Percentile range of the mean of ``values`` over the resampled index sets."""
    return stats.interval(np.array([values[i].mean() for i in index_sets]), confidence)


def deletion_summary(
    scores: pd.DataFrame, photos: pd.DataFrame, blocks: pd.Series | None, settings: dict, comparison: dict
) -> pd.DataFrame:
    """Per k on the pooled photos of the runs given: top-k against random-k, and the verdict of rule 9.

    ``blocks`` maps photo_id to capture block. The range is given twice (decision D9): resampling photos
    and resampling whole capture blocks; the verdict needs both above 0 and the round wins. All five
    rounds must be present. Probabilities are means over photos, like the gap.

    ``blocks=None`` (Paddy Doctor has no capture blocks): the five round-models score the same photos, so
    each photo's values are averaged over them first; the photos are resampled; the verdict needs the photo
    range above 0 and the round wins (handbook §24, 2026-11-21).
    """
    if scores["round"].nunique() != len(ROUNDS):
        raise ValueError(f"deletion results of {scores['round'].nunique()} rounds, expected {len(ROUNDS)}")
    random = scores[scores["condition"] == "random"].groupby(["run", "round", "k"]).mean(numeric_only=True)
    top = scores[scores["condition"] == "top"].set_index(["run", "round", "k"])
    full_f1 = scores[scores["condition"] == "none"]["macro_f1"].mean()
    rows = []
    for k in sorted(top.index.get_level_values("k").unique()):
        t, rnd = top.xs(k, level="k"), random.xs(k, level="k")
        p = photos[photos["k"] == k]
        if blocks is None:
            p = p.groupby("photo_id", as_index=False).agg(
                true_class=("true_class", "first"),
                p_full=("p_full", "mean"),
                p_top=("p_top", "mean"),
                p_random=("p_random", "mean"),
            )
        gap = (p["p_random"] - p["p_top"]).to_numpy()
        photo_sets = stats.resample_indices(
            p["true_class"].to_numpy(), settings["resamples"], settings["stratified"], 0
        )
        low, high = bootstrap_range(gap, photo_sets, settings["confidence"])
        ranges = {"gap_low": low, "gap_high": high}
        if blocks is not None:
            block_sets = followup_report.block_samples(
                p["photo_id"].map(blocks).to_numpy(), settings["resamples"], 0
            )
            block_low, block_high = bootstrap_range(gap, block_sets, settings["confidence"])
            ranges |= {"gap_block_low": block_low, "gap_block_high": block_high}
        wins = int((rnd["p_true"] > t["p_true"]).sum())
        better = (
            ranges["gap_low"] > 0
            and ranges.get("gap_block_low", 1) > 0
            and wins >= comparison["min_round_wins"]
        )
        rows.append(
            {
                "k": k,
                "full_macro_f1": full_f1,
                "top_macro_f1": t["macro_f1"].mean(),
                "random_macro_f1": rnd["macro_f1"].mean(),
                "full_p_true": p["p_full"].mean(),
                "top_p_true": p["p_top"].mean(),
                "random_p_true": p["p_random"].mean(),
                "gap_p_true": gap.mean(),
                **ranges,
                "round_wins": wins,
                "rounds": len(t),
                "verdict": ("top-k matters more than random-k" if better else "no clear difference")
                + ("" if blocks is not None else " (photo range only)"),
            }
        )
    return pd.DataFrame(rows)


def gap_by_class(photos: pd.DataFrame) -> pd.DataFrame:
    """Mean true-class-probability gap (random-k minus top-k) per true class and k, in the photos' own units.
    Healthy photos have no symptoms to find, so read their row apart from the disease classes."""
    table = photos.assign(gap=photos["p_random"] - photos["p_top"])
    return table.pivot_table(index="true_class", columns="k", values="gap", aggfunc="mean").reset_index()


def deletion_figure(summary: pd.DataFrame, path: Path, partition: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, key, label in zip(
        axes, ("macro_f1", "p_true"), (f"{partition} macro-F1", f"{partition} true-class probability")
    ):
        axis.plot(summary["k"], summary[f"top_{key}"], "o-", label="top-k attended removed")
        axis.plot(summary["k"], summary[f"random_{key}"], "s--", label="random k removed")
        axis.axhline(summary[f"full_{key}"].iloc[0], color="grey", lw=0.8, label="nothing removed")
        axis.set(xscale="log", xlabel="tiles removed (k)", ylabel=label)
        axis.set_xticks(list(summary["k"]), [str(k) for k in summary["k"]])
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    atomic(path, lambda partial: figure.savefig(partial, dpi=130))
    plt.close(figure)


def photos_label(partition: str) -> str:
    return "Paddy Doctor photos at native zoom" if partition == "paddy" else f"{partition} photos"


def deletion_section(partition: str, settings: dict, comparison: dict) -> list[str]:
    scores = read_all(f"deletion_{partition}_*.csv")
    photos = read_all(f"deletion_photos_{partition}_*.csv")
    title = f"## Deletion test, {photos_label(partition)} (seed-0 models)"
    if scores is None:
        return [title, "", "Not run yet.", ""]
    if seed0(scores)["round"].nunique() != len(ROUNDS):
        return [
            title,
            "",
            f"Incomplete: {seed0(scores)['run'].nunique()} of {len(ROUNDS)} runs finished.",
            "",
        ]
    blocks = None if partition == "paddy" else followup_report.block_of()
    summary = deletion_summary(seed0(scores), seed0(photos), blocks, settings, comparison)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / f"explain_deletion_{partition}.csv", index=False)
    scores.to_csv(TABLES / f"explain_deletion_rounds_{partition}.csv", index=False)
    by_class = gap_by_class(seed0(photos))
    by_class.to_csv(TABLES / f"explain_deletion_by_class_{partition}.csv", index=False)
    deletion_figure(summary, FIGURES / f"explain_deletion_{partition}.png", partition)
    percent = [c for c in summary.columns if c.startswith(("full_", "top_", "random_", "gap_"))]
    by_class[[c for c in by_class.columns if c != "true_class"]] *= 100
    return [
        title,
        "",
        "Percentages. `gap` = true-class probability with random-k removed minus with top-k removed; "
        "ranges = "
        + (
            "95% paired bootstrap over photos only (Paddy Doctor has no capture blocks; each photo's five "
            "round-model values are averaged first; near-duplicate photos are not grouped, so the range is "
            "narrower than it would be with groups). Rule: top-k matters more than random-k only if the "
            "photo range is above 0 and the gap is positive in at least "
            if partition == "paddy"
            else "95% paired bootstrap over photos and over whole capture blocks (decision D9). Rule: top-k "
            "matters more than random-k only if both ranges are above 0 and the gap is positive in at least "
        )
        + f"{comparison['min_round_wins']} of {len(ROUNDS)} rounds."
        + (
            " A validation photo can be in several rounds; each appearance counts."
            if partition == "val"
            else ""
        ),
        "",
        md_table(summary.assign(**{c: (summary[c] * 100).round(1) for c in percent})),
        "",
        f"![deletion curves](../../results/figures/explain_deletion_{partition}.png)",
        "",
        "A big gap shows that the head's decision rests on the tiles it attends to. It does not by itself "
        "show that those tiles are symptoms (that is the human check), and healthy photos have none. "
        "Gap by true class (percentage points, per k):",
        "",
        md_table(by_class.round(1)),
        "",
    ]


def background_section(partition: str) -> list[str]:
    table = OUT / f"background_{partition}.csv"
    title = f"## Attention on background tiles, {photos_label(partition)} (seed-0 models)"
    if not table.exists():
        return [title, "", "Not run yet.", ""]
    data = seed0(pd.read_csv(table))
    by_class = data.groupby("true_class")[["attention_share", "tile_share"]].mean()
    by_class.loc["all"] = data[["attention_share", "tile_share"]].mean()
    by_class["attention_per_tile_share"] = by_class["attention_share"] / by_class["tile_share"]
    TABLES.mkdir(parents=True, exist_ok=True)
    by_class.reset_index().to_csv(TABLES / f"explain_background_{partition}.csv", index=False)
    shown = by_class.reset_index().assign(
        attention_share=lambda t: (t["attention_share"] * 100).round(1),
        tile_share=lambda t: (t["tile_share"] * 100).round(1),
        attention_per_tile_share=lambda t: t["attention_per_tile_share"].round(2),
    )
    suspect = data.drop_duplicates("photo_id")["mask_suspect"].sum()
    return [
        title,
        "",
        f"A tile is background when more than {BACKGROUND_SHARE:.0%} of its pixels are outside the Otsu leaf "
        "mask. `attention_per_tile_share` below 1 means background tiles get less than their share. "
        f"Mask check: {suspect} photos have a leaf mask covering less than {MASK_SANE[0]:.0%} or more than "
        f"{MASK_SANE[1]:.0%} of the photo (the side guess may be inverted there).",
        "",
        md_table(shown),
        "",
    ]


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa; ``counts[i, j]`` = raters who gave item i answer j (same number of raters per item)."""
    raters = counts.sum(1)
    if len(set(raters)) != 1 or raters[0] < 2:
        raise ValueError("Fleiss' kappa needs the same number (at least 2) of raters for every item")
    n = raters[0]
    agreement = ((counts**2).sum(1) - n) / (n * (n - 1))
    chance = ((counts.sum(0) / counts.sum()) ** 2).sum()
    return float("nan") if chance == 1 else float((agreement.mean() - chance) / (1 - chance))


def rating_summary(ratings: pd.DataFrame, examples: pd.DataFrame) -> dict:
    """Checks the ratings file against the blind sheet and gives shares and agreement.

    Every rater must have rated every item once, with a valid answer and one kind per rater."""
    if not set(RATING_COLUMNS) <= set(ratings):
        raise ValueError(f"the ratings file needs the columns {RATING_COLUMNS}")
    if not ratings["rating"].isin(ANSWERS).all() or not ratings["rater_kind"].isin(RATER_KINDS).all():
        raise ValueError(f"ratings must be one of {ANSWERS} and rater_kind one of {RATER_KINDS}")
    if (
        ratings.duplicated(["item_id", "rater"]).any()
        or (ratings.groupby("rater")["rater_kind"].nunique() > 1).any()
    ):
        raise ValueError("a rater rated an item twice, or has two kinds")
    items = set(examples["item_id"])
    if (
        not set(ratings["item_id"]) <= items
        or (ratings.groupby("rater")["item_id"].nunique() != len(items)).any()
    ):
        raise ValueError("every rater must rate every blind item, and only those")
    wide = ratings.pivot(index="item_id", columns="rater", values="rating")
    counts = np.stack([(wide == a).sum(1).to_numpy() for a in ANSWERS], axis=1)
    shares = pd.crosstab(ratings["rater"], ratings["rating"], normalize="index").reindex(
        columns=list(ANSWERS), fill_value=0
    )
    shares.loc["all ratings"] = (
        ratings["rating"].value_counts(normalize=True).reindex(list(ANSWERS), fill_value=0)
    )
    by_category = (
        ratings.merge(examples[["item_id", "category"]], on="item_id")
        .groupby("category")["rating"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=list(ANSWERS), fill_value=0)
    )
    return {
        "shares": shares,
        "by_category": by_category,
        "raters": ratings.drop_duplicates("rater").set_index("rater")["rater_kind"],
        "items": len(items),
        "all_agree": float((wide.nunique(axis=1) == 1).mean()),
        "kappa": fleiss_kappa(counts),
    }


def ratings_section() -> list[str]:
    title = "## Blind rating of the heat maps (Objective 6)"
    examples_path = OUT / "examples_val.csv"
    if not RATINGS.exists() or not examples_path.exists() or RATINGS.read_text().count("\n") < 2:
        return [title, "", "No ratings yet.", ""]
    summary = rating_summary(pd.read_csv(RATINGS), pd.read_csv(examples_path))
    TABLES.mkdir(parents=True, exist_ok=True)
    summary["shares"].reset_index().to_csv(TABLES / "explain_ratings.csv", index=False)
    show = lambda table: md_table((table * 100).round(1).reset_index())  # noqa: E731
    raters = ", ".join(f"`{name}` ({kind})" for name, kind in summary["raters"].items())
    return [
        title,
        "",
        f"{summary['items']} blind heat maps (the rule's validation examples and random extras, shown "
        f"without the true class or the prediction). Raters, all disclosed: {raters}. Question: does the hot "
        "area cover visible symptoms? Percentages of ratings.",
        "",
        show(summary["shares"]),
        "",
        f"All raters gave the same answer on {summary['all_agree']:.1%} of the maps; Fleiss' kappa "
        f"{summary['kappa']:.2f} (three answers; 0 = chance level, 1 = perfect agreement). By example "
        "category (`extra` = random photos; `wrong` = the rule's confident mistakes), seen only after "
        "rating:",
        "",
        show(summary["by_category"]),
        "",
    ]


def report() -> None:
    evaluation = load_config("configs/base.yaml")["evaluation"]
    text = [
        "# Stage 12: does the attention head look at real symptoms?",
        "",
        "Generated by `python -m src.explain report` from the files in `results/explain/`; nothing is typed "
        "by hand. The declaration is `runs/explain_plan.json` (handbook §24, 2026-11-20; the Paddy Doctor "
        "extension, 2026-11-21).",
        "",
    ]
    for partition in ("test", "val", "paddy"):
        text += deletion_section(partition, evaluation["bootstrap"], evaluation["comparison"])
    for partition in ("test", "val", "paddy"):
        text += background_section(partition)
    text += ratings_section()
    for partition in ("val", "test", "paddy"):
        path = OUT / f"examples_{partition}.csv"
        if path.exists():
            text += [
                f"## Examples chosen by the rule ({partition} predictions)",
                "",
                md_table(pd.read_csv(path).round(3)),
                "",
            ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(text))
    print(f"report -> {REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["select", "heatmaps", "plan", "deletion", "background", "report"])
    parser.add_argument("--partition", choices=["val", "test", "paddy"], default="val")
    args = parser.parse_args()
    {
        "select": lambda: select(args.partition),
        "heatmaps": lambda: heatmaps(args.partition),
        "plan": plan,
        "deletion": lambda: deletion(args.partition),
        "background": lambda: background(args.partition),
        "report": report,
    }[args.action]()


if __name__ == "__main__":
    main()
