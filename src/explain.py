"""Stage 12: does the attention head look at real symptoms? (handbook Stage 12; declared in §24, 2026-11-20)

    uv run python -m src.explain select   [--partition val|test]   # rule-based example list (CSV first)
    uv run python -m src.explain heatmaps [--partition val|test]   # labelled and blind heat maps
    uv run python -m src.explain plan                              # freeze the declaration (once)
    uv run python -m src.explain deletion   --partition test|val   # remove top-k / random-k tiles
    uv run python -m src.explain background --partition test|val   # attention on background tiles
    uv run python -m src.explain report                            # tables, figure and report

Models: the five session-split attention runs (``stage7_session_attention_r1..r5``, tested once) and, on
validation photos only, their two stability seeds. Every action is resumable and writes only under
``results/``, ``docs/reports/`` and (the declaration and the run-once marker) ``runs/``. Test photos are read
by ``deletion`` and ``background`` only, after ``plan`` froze what is measured.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import followup_report, metrics, otsu, stats, tiling
from src import train_mil as tm
from src.config import REPO_ROOT, cache_folder, dataset_dir, load_config
from src.embed import load_version, photo_key, scale_factors, version_settings
from src.evaluate import checkpoint, digest
from src.inventory import read_inventory
from src.inventory_report import md_table
from src.mil_models import MILModel
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

SEED_TAGS = {0: "", 1: "_s1", 2: "_s2"}  # model training seeds; only seed 0 was tested
ROUNDS = (1, 2, 3, 4, 5)
KS = (1, 2, 4, 8, 16)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
PER_CELL = 3  # examples per class and category
EXTRA_ITEMS = 14  # 4 classes x 3 categories x 3 = 36 rule examples, plus 14 random ones = 50 rated heat maps
SELECT_SEED = 0
BACKGROUND_SHARE = 0.9  # a tile is background when more than this share of its pixels is
CATEGORIES = ("correct", "wrong", "random")


def run_name(seed: int, round_: int) -> str:
    return f"stage7_session_attention{SEED_TAGS[seed]}_r{round_}"


def runs_for(partition: str) -> list[str]:
    """Test: only the five runs that were tested. Validation: all three training seeds."""
    seeds = (0,) if partition == "test" else tuple(SEED_TAGS)
    return [run_name(s, r) for s in seeds for r in ROUNDS]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


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
    classes: list[str],
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
                if not np.allclose(weights[i, :n].cpu().numpy(), stored[photo_id], atol=1e-3):
                    raise ValueError(f"{photo_id}: recomputed attention differs from the saved attention")
        record("none", 0, -1, ids, y, mask, tiles)
        for k in KS:
            record("top", k, -1, ids, y, top_k_mask(weights, mask, k), tiles)
            for seed in RANDOM_SEEDS:
                record("random", k, seed, ids, y, random_k_mask(mask, k, generators[seed]), tiles)
    long = pd.concat(parts, ignore_index=True)
    long["correct"] = long["pred"] == long["true"]
    return long


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
    random = (
        long[long["condition"] == "random"]
        .groupby(["photo_id", "k"])
        .agg(p_random=("p_true", "mean"), correct_random=("correct", "mean"))
    )
    out = (
        top[["p_true", "correct"]].rename(columns={"p_true": "p_top", "correct": "correct_top"}).join(random)
    )
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


def background_share(weights: np.ndarray, fraction: np.ndarray, threshold: float = BACKGROUND_SHARE) -> dict:
    """Share of attention, and share of tiles, on tiles that are more than ``threshold`` background."""
    is_background = fraction > threshold
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
    long_side: int = 900,
) -> None:
    """Photo and overlay side by side. Works for any photo, boxes and weights (primary or Paddy Doctor)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    scale = long_side / max(rgb.shape[:2])
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
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------- files of the runs


def read_run_predictions(partition: str) -> pd.DataFrame:
    """Seed-0 predictions of the five rounds on ``partition``."""
    name = "test_predictions.csv" if partition == "test" else "predictions.csv"
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


class Photos:
    """Photo pixels at the common leaf scale (as the cache saw them) and the cache's tile boxes."""

    def __init__(self) -> None:
        self.config = load_config(EMBED_CONFIG)
        self.root = dataset_dir(self.config, "primary")
        self.tile = self.config["embed"]["tile"]
        self.cache = cache_folder(self.config["embed"]["name"])
        inventory = read_inventory(self.config["inventory"]["files"]["primary"])
        self.factors = scale_factors(self.config, sorted(inventory.loc[inventory["opened_ok"], "photo_id"]))

    def factor(self, photo_id: str) -> float:
        return self.factors[photo_id]

    def rgb(self, photo_id: str) -> np.ndarray:
        change = version_settings({}, self.config["seed"], 0, photo_id)
        return load_version(str(self.root / photo_id), self.factor(photo_id), change)

    def boxes(self, photo_id: str) -> np.ndarray:
        with np.load(self.cache / "v0" / f"{photo_key(photo_id)}.npz") as saved:
            return saved["boxes"]


# ---------------------------------------------------------------- actions


def select(partition: str) -> None:
    if partition == "test":
        require_plan()
    config = load_config(EMBED_CONFIG)
    classes = config["classes"]
    predictions = read_run_predictions(partition)
    examples = add_extras(
        rank_examples(predictions, classes, PER_CELL, SELECT_SEED), predictions, EXTRA_ITEMS, SELECT_SEED
    )
    path = OUT / f"examples_{partition}.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), examples, check_dtype=False, atol=1e-6)  # same rule
    else:
        write_csv(examples, path)
    if not RATINGS.exists():
        write_csv(pd.DataFrame(columns=RATING_COLUMNS), RATINGS)
    print(f"{len(examples)} examples ({len(examples) - EXTRA_ITEMS} by rule, {EXTRA_ITEMS} extra) -> {path}")


def heatmaps(partition: str) -> None:
    examples = pd.read_csv(OUT / f"examples_{partition}.csv")
    photos = Photos()
    attention = {r: read_attention(run_name(0, r), partition) for r in ROUNDS}
    for row in examples.itertuples():
        labelled = (
            OUT
            / "heatmaps"
            / partition
            / f"{row.true_class}_{row.category}_{row.rank}_{photo_key(row.photo_id)}.png"
        )
        blind = OUT / "blind" / partition / f"{row.item_id}.png"
        if blind.exists() and (labelled.exists() or row.category == "extra"):
            continue
        weights, boxes, rgb = (
            attention[row.round][row.photo_id],
            photos.boxes(row.photo_id),
            photos.rgb(row.photo_id),
        )
        check_grid(rgb, boxes, photos.tile)
        draw_heatmap(rgb, boxes, weights, photos.tile, blind, row.item_id)
        if row.category != "extra":
            title = (
                f"{row.true_class}, {row.category} #{row.rank}: predicted {row.pred_class} "
                f"({row.confidence:.2f}); {row.photo_id}, round {row.round}; peak tile "
                f"{weights.max() * len(weights):.1f}x uniform"
            )
            draw_heatmap(rgb, boxes, weights, photos.tile, labelled, title)
    print(f"heat maps -> {OUT / 'heatmaps' / partition} and {OUT / 'blind' / partition}")


def plan() -> None:
    version = require_clean_code()
    root = get_default_runs_dir()
    path = root / "explain_plan.json"
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
            "partitions": ["test", "val"],
            "ranking": "the model's own attention on the full bag, taken once",
            "removal": "switch the tiles off in the real-tile mask (equal to removing them)",
            "scores": ["accuracy", "macro_f1", "true-class probability"],
        },
        "background": {
            "threshold": BACKGROUND_SHARE,
            "mask": "Otsu leaf mask of the photo at the common leaf scale, a measuring tool only",
            "partitions": ["test", "val"],
        },
        "examples": {"per_cell": PER_CELL, "extra": EXTRA_ITEMS, "seed": SELECT_SEED},
        "test_runs": [n for n, v in runs.items() if v["tested"]],
        "runs": runs,
    }
    if path.exists():
        print(f"{path} already declared; not changed")
        return
    with path.open("x") as out:
        json.dump(declaration, out, indent=2)
    print(f"declared {path}")


def start_marker(partition: str, action: str) -> dict:
    """The declaration, and (for test) the run-once marker: a later call may resume, never restart."""
    plan_path = require_plan()
    declaration = json.loads(plan_path.read_text())
    version = require_clean_code()
    if partition == "test":
        marker = get_default_runs_dir() / f"explain_{action}_started.json"
        content = {"evaluation_code": version, "plan_sha256": digest(plan_path)}
        if marker.exists():
            if json.loads(marker.read_text())["plan_sha256"] != content["plan_sha256"]:
                raise ValueError(f"{marker}: started under another plan")
        else:
            with marker.open("x") as out:
                json.dump(content, out)
    return declaration


def load_run(name: str, declaration: dict):
    """Model, folds and config of one run, after checking the files are the ones that were declared."""
    root = get_default_runs_dir()
    expected = declaration["runs"][name]
    record = json.loads((root / name / "record.json").read_text())
    if (
        digest(root / name / "record.json") != expected["record_sha256"]
        or digest(checkpoint(root / name)) != expected["checkpoint_sha256"]
    ):
        raise ValueError(f"{name}: changed after the declaration")
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
    names = declaration["test_runs"] if partition == "test" else runs_for("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in names:
        scores_path, photos_path = (
            OUT / f"deletion_{partition}_{name}.csv",
            OUT / f"deletion_photos_{partition}_{name}.csv",
        )
        if scores_path.exists() and photos_path.exists():
            continue
        model, store, folds, config = load_run(name, declaration)
        r, classes = config["round"], config["classes"]
        bags = tm.Bags(store, folds, r, partition, classes, 1, 0.0, config["mil"]["batch"], config["seed"])
        stored = read_attention(name, partition)
        long = delete_and_score(model.to(device), bags, classes, device, stored)
        write_csv(photo_table(long, classes).assign(run=name, round=r), photos_path)
        write_csv(score_conditions(long, classes).assign(run=name, round=r), scores_path)
        print(f"{name}: deletion test on {partition} photos done", flush=True)


def background(partition: str) -> None:
    declaration = start_marker(partition, "background")
    names = declaration["test_runs"] if partition == "test" else runs_for("val")
    photos, fractions = Photos(), OUT / "background_fraction"
    folds, _ = read_folds(load_config(SPLIT_CONFIG))
    folds = folds.set_index("photo_id")
    rows = []
    for name in names:
        r = int(name.rsplit("_r", 1)[1])
        for photo_id, weights in read_attention(name, partition).items():
            path = fractions / f"{photo_key(photo_id)}.npy"
            if not path.exists():
                rgb, boxes = photos.rgb(photo_id), photos.boxes(photo_id)
                check_grid(rgb, boxes, photos.tile)
                fractions.mkdir(parents=True, exist_ok=True)
                np.save(
                    path, tile_background_fraction(otsu.leaf_mask(rgb), boxes, photos.tile).astype(np.float32)
                )
            rows.append(
                {
                    "run": name,
                    "round": r,
                    "photo_id": photo_id,
                    "true_class": folds.loc[photo_id, "class"],
                    **background_share(weights, np.load(path)),
                    "n_tiles": len(weights),
                }
            )
        print(f"{name}: background score on {partition} photos done", flush=True)
    write_csv(pd.DataFrame(rows), OUT / f"background_{partition}.csv")


# ---------------------------------------------------------------- report


def seed0(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of the seed-0 runs (the headline models); the `_s1`, `_s2` runs are stability checks."""
    return frame[~frame["run"].str.contains("_s[12]_")]


def read_all(pattern: str) -> pd.DataFrame | None:
    files = sorted(OUT.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def deletion_summary(
    scores: pd.DataFrame, photos: pd.DataFrame, blocks: pd.Series, settings: dict, comparison: dict
) -> pd.DataFrame:
    """Per k on the pooled photos of the runs given: top-k against random-k, and the verdict of rule 9.

    ``blocks`` maps photo_id to capture block. The range is given twice (decision D9): resampling photos
    and resampling whole capture blocks; the verdict needs both above 0 and the round wins.
    """
    random = (
        scores[scores["condition"] == "random"]
        .groupby(["run", "round", "k"], as_index=False)
        .mean(numeric_only=True)
    )
    top = scores[scores["condition"] == "top"].set_index(["run", "round", "k"])
    random = random.set_index(["run", "round", "k"])
    full = scores[scores["condition"] == "none"].set_index(["run", "round"])
    rows = []
    for k in sorted(top.index.get_level_values("k").unique()):
        t, rnd = top.xs(k, level="k"), random.xs(k, level="k")
        p = photos[photos["k"] == k]
        gap = (p["p_random"] - p["p_top"]).to_numpy()
        photo_draws = [
            gap[i].mean()
            for i in stats.resample_indices(
                p["true_class"].to_numpy(), settings["resamples"], settings["stratified"], 0
            )
        ]
        block_draws = [
            gap[i].mean()
            for i in followup_report.block_samples(
                p["photo_id"].map(blocks).to_numpy(), settings["resamples"], 0
            )
        ]
        low, high = stats.interval(np.array(photo_draws), settings["confidence"])
        block_low, block_high = stats.interval(np.array(block_draws), settings["confidence"])
        wins = int((rnd["p_true"] > t["p_true"]).sum())
        better = min(low, block_low) > 0 and wins >= comparison["min_round_wins"]
        rows.append(
            {
                "k": k,
                "full_macro_f1": full["macro_f1"].mean(),
                "top_macro_f1": t["macro_f1"].mean(),
                "random_macro_f1": rnd["macro_f1"].mean(),
                "full_p_true": full["p_true"].mean(),
                "top_p_true": t["p_true"].mean(),
                "random_p_true": rnd["p_true"].mean(),
                "gap_p_true": gap.mean(),
                "gap_low": low,
                "gap_high": high,
                "gap_block_low": block_low,
                "gap_block_high": block_high,
                "round_wins": wins,
                "rounds": len(t),
                "verdict": "top-k matters more than random-k" if better else "no clear difference",
            }
        )
    return pd.DataFrame(rows)


def deletion_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, key, label in zip(axes, ("macro_f1", "p_true"), ("test macro-F1", "true-class probability")):
        axis.plot(summary["k"], summary[f"top_{key}"], "o-", label="top-k attended removed")
        axis.plot(summary["k"], summary[f"random_{key}"], "s--", label="random k removed")
        axis.axhline(summary[f"full_{key}"].iloc[0], color="grey", lw=0.8, label="nothing removed")
        axis.set(xscale="log", xlabel="tiles removed (k)", ylabel=label)
        axis.set_xticks(list(summary["k"]), [str(k) for k in summary["k"]])
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def report() -> None:
    config = load_config("configs/base.yaml")
    settings, comparison = config["evaluation"]["bootstrap"], config["evaluation"]["comparison"]
    pct = lambda frame, cols: frame.assign(**{c: (frame[c] * 100).round(1) for c in cols})  # noqa: E731
    text = [
        "# Stage 12: does the attention head look at real symptoms?",
        "",
        "Generated by `python -m src.explain report` from the files in `results/explain/`; nothing is typed "
        "by hand. The declaration is `runs/explain_plan.json` (handbook §24, 2026-11-20).",
        "",
    ]
    for partition in ("test", "val"):
        scores, photos = (
            read_all(f"deletion_{partition}_*.csv"),
            read_all(f"deletion_photos_{partition}_*.csv"),
        )
        if scores is None:
            text += [f"## Deletion test, {partition} photos", "", "Not run yet.", ""]
            continue
        summary = deletion_summary(
            seed0(scores), seed0(photos), followup_report.block_of(), settings, comparison
        )
        summary.to_csv(TABLES / f"explain_deletion_{partition}.csv", index=False)
        scores.to_csv(TABLES / f"explain_deletion_rounds_{partition}.csv", index=False)
        deletion_figure(summary, FIGURES / f"explain_deletion_{partition}.png")
        shown = pct(
            summary,
            [
                "full_macro_f1",
                "top_macro_f1",
                "random_macro_f1",
                "full_p_true",
                "top_p_true",
                "random_p_true",
                "gap_p_true",
                "gap_low",
                "gap_high",
                "gap_block_low",
                "gap_block_high",
            ],
        )
        text += [
            f"## Deletion test, {partition} photos (seed-0 models)",
            "",
            "Percentages. `gap` = true-class probability with random-k removed minus with top-k removed; "
            "ranges = 95% paired bootstrap over photos and over whole capture blocks (decision D9). Rule: "
            "top-k matters more than random-k only if both ranges are above 0 and the gap is positive in at "
            f"least {comparison['min_round_wins']} of 5 rounds. "
            + (
                "Validation photos can be in the validation set of several rounds; each appearance counts."
                if partition == "val"
                else ""
            ),
            "",
            md_table(shown),
            "",
            f"![deletion curves](../../results/figures/explain_deletion_{partition}.png)",
            "",
        ]
    for partition in ("test", "val"):
        table = OUT / f"background_{partition}.csv"
        if not table.exists():
            text += [f"## Attention on background tiles, {partition} photos", "", "Not run yet.", ""]
            continue
        data = pd.read_csv(table)
        data = seed0(data)
        by_class = data.groupby("true_class")[["attention_share", "tile_share"]].mean()
        by_class.loc["all"] = data[["attention_share", "tile_share"]].mean()
        by_class["attention_per_tile_share"] = by_class["attention_share"] / by_class["tile_share"]
        by_class.reset_index().to_csv(TABLES / f"explain_background_{partition}.csv", index=False)
        shown = pct(by_class.reset_index(), ["attention_share", "tile_share"]).round(2)
        text += [
            f"## Attention on background tiles, {partition} photos (seed-0 models)",
            "",
            f"A tile is background when more than {BACKGROUND_SHARE:.0%} of its pixels are outside the Otsu "
            "leaf mask. `attention_per_tile_share` below 1 means background tiles get less than their share.",
            "",
            md_table(shown),
            "",
        ]
    for partition in ("val", "test"):
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
    parser.add_argument("--partition", choices=["val", "test"], default="val")
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
