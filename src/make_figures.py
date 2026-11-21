"""Stage 14: the final figures, built from run records and the per-stage tables (same sources as make_tables).

    uv run python -m src.make_figures    # writes results/figures/final_*.{png,pdf} and final_figures.json

One style for every figure: matplotlib only, sized for a two-column IEEE paper (3.5 in for one column, 7.16 in for
the page), and readable in greyscale (marker shapes, line styles and grey levels carry the meaning, not colour).
A figure whose source does not exist yet is skipped and listed as pending in ``final_figures.json``; the sample
photos need ``data/raw/`` and are skipped where it is missing (a worktree). Nothing is evaluated here.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import make_tables as mt  # noqa: E402
from src.records import get_code_version  # noqa: E402
from src.make_tables import CLASSES, FIGURES, Missing  # noqa: E402

COLUMN, PAGE = 3.5, 7.16          # inches: one IEEE column, the full page width
INK, MID, LIGHT = "#1a1a1a", "#777777", "#cfcfcf"
MARKERS = ("o", "s", "^", "D", "v")
SHORT = {"healthy": "healthy", "leaf_blast": "blast", "brown_spot": "brown", "hispa": "hispa"}
PHOTOS = "data/raw/rice_diseases/LabelledRice/Labelled"
SAMPLES_PER_CLASS = 3
COST_COLUMNS = ("method", "seconds_per_photo", "macro_f1")   # what the Stage 13 table must give for the cost chart

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 0.9, "lines.markersize": 4, "figure.dpi": 100, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.prop_cycle": plt.cycler(color=[INK, MID, "#444444", "#aaaaaa"]),
})


# -------------------------------------------------------------------------------------------- the figures
Result = tuple[plt.Figure, list[Path]]


def samples(root: Path) -> Result:
    from PIL import Image, ImageOps

    folder = root / PHOTOS
    if not folder.is_dir():
        raise Missing(f"{PHOTOS} (photos exist only in the main checkout)")
    folds = mt.read_table(root, "splits/folds.csv")
    figure, axes = plt.subplots(len(CLASSES), SAMPLES_PER_CLASS, figsize=(PAGE / 2, PAGE / 2 * 4 / 3 * 0.9))
    used = []
    for row, cls in enumerate(CLASSES):
        ids = sorted(folds.loc[folds["class"] == cls, "photo_id"])
        for col, i in enumerate(np.linspace(0, len(ids) - 1, SAMPLES_PER_CLASS).round().astype(int)):
            with Image.open(folder / ids[i]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((500, 500))
                axes[row, col].imshow(np.asarray(image))
            axes[row, col].set_xticks([]), axes[row, col].set_yticks([])
            axes[row, col].spines[:].set_visible(False)
            used.append(root / PHOTOS / ids[i])
        axes[row, 0].set_ylabel(SHORT[cls], rotation=90)
    figure.tight_layout(pad=0.4)
    return figure, [root / "splits/folds.csv", *used]


def confusion(root: Path) -> Result:
    scores, files = mt.session_scores(root, mt.HEADLINE_CONFUSION)
    figure, axes = plt.subplots(1, len(scores), figsize=(PAGE, 2.0))
    axes = np.atleast_1d(axes)
    first = axes[0]
    for axis, (method, s) in zip(axes, scores.items()):
        counts = np.array(s["confusion"])
        share = counts / counts.sum(axis=1, keepdims=True)
        axis.imshow(share, cmap="Greys", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                axis.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=6,
                          color="white" if share[i, j] > 0.5 else INK)
        axis.set_xticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES], rotation=45, ha="right")
        axis.set_yticks(range(len(CLASSES)), [SHORT[c] for c in CLASSES] if axis is first else [])
        axis.set_title(mt.name_of(method), fontsize=6.5)
        axis.set_xlabel("predicted")
        axis.spines[:].set_visible(False)
    first.get_yticklabels()[-1].set_fontweight("bold")          # hispa, the pre-declared diagnostic
    first.set_ylabel("true class (row %)")
    figure.tight_layout(pad=0.5)
    return figure, files


def partition_order(root: Path) -> Result:
    path = f"{mt.TABLES}/partition_order_inflation.csv"
    po = mt.read_table(root, path)
    order = list(dict.fromkeys(po.condition))
    figure, axis = plt.subplots(figsize=(COLUMN, 2.3))
    models = list(dict.fromkeys(po.model))
    for k, model in enumerate(models):
        part = po[po.model == model].set_index("condition").loc[order]
        x = np.arange(len(order)) + (k - (len(models) - 1) / 2) * 0.22
        axis.errorbar(x, 100 * part.macro_f1_pooled,
                      yerr=[100 * (part.macro_f1_pooled - part.macro_f1_low), 100 * (part.macro_f1_high - part.macro_f1_pooled)],
                      fmt=MARKERS[k], color=INK, mfc=INK if k == 0 else "white", capsize=2, label=mt.name_of(model))
    axis.set_xticks(range(len(order)), order)
    axis.set_xlabel("set-up (A = correct)")
    axis.set_ylabel("test macro-F1 (%), pooled, 95% range")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="y", color=LIGHT, lw=0.4)
    figure.tight_layout(pad=0.5)
    return figure, [root / path]


def ablations(root: Path) -> Result:
    path = f"{mt.TABLES}/session_ablations.csv"
    val = mt.read_table(root, path).query("variant != 'main'").reset_index(drop=True)
    test = mt.ablation_report_rows(root)
    figure, axis = plt.subplots(figsize=(COLUMN, 3.0))
    y = np.arange(len(val))[::-1]
    axis.plot(100 * val["difference_vs_main"], y, "o", mfc="white", color=INK, label="validation (all variants)")
    for n, r in enumerate(test[test["metric"] == "macro_f1"].to_dict("records")):
        i = list(val["variant"]).index(r["variant"])
        low, high = mt.numbers(r["photo range"])            # the report gives main minus variant; the plot shows variant minus main
        axis.hlines(y[i], -high, -low, color=INK, lw=1.2)
        axis.plot(-mt.numbers(r["difference (points)"])[0], y[i], "D", color=INK,
                  label="test, tested once (95% photo range)" if n == 0 else None)
    axis.axvline(0, color=MID, lw=0.6)
    axis.set_yticks(y, val["variant"])
    axis.set_xlabel("variant minus main, macro-F1 (points)")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="x", color=LIGHT, lw=0.4)
    figure.tight_layout(pad=0.5)
    return figure, [root / path, root / mt.ABLATION_REPORT]


def paddy_zoom(root: Path) -> Result:
    path = f"{mt.TABLES}/paddy_results.csv"
    paddy = mt.read_table(root, path)
    zooms = list(dict.fromkeys(z for z in paddy.zoom.astype(str) if z != "-"))
    order = list(dict.fromkeys(paddy.method))
    figure, axis = plt.subplots(figsize=(COLUMN, 0.22 * len(order) + 0.9))
    for k, zoom in enumerate([*zooms, "-"]):
        part = paddy[paddy.zoom.astype(str) == zoom]
        y = [len(order) - 1 - order.index(m) for m in part.method]
        axis.errorbar(100 * part.macro_f1_mean, y, xerr=100 * part.macro_f1_sd, fmt=MARKERS[k % len(MARKERS)], color=INK,
                      mfc=INK if k == 0 else "white", capsize=1.5, label=zoom if zoom != "-" else "no zoom levels")
    axis.set_yticks(range(len(order)), [mt.name_of(m) for m in order[::-1]])
    axis.set_xlabel("Paddy Doctor zero-shot macro-F1 (%), mean ± SD over rounds")
    axis.legend(frameon=False, title="zoom", title_fontsize=6)
    axis.grid(axis="x", color=LIGHT, lw=0.4)
    figure.tight_layout(pad=0.5)
    return figure, [root / path]


def deletion(root: Path) -> Result:
    path = f"{mt.TABLES}/explain_deletion_test.csv"
    d = mt.read_table(root, path)
    figure, axes = plt.subplots(1, 2, figsize=(PAGE * 0.7, 2.2))
    for axis, key, label in zip(axes, ("macro_f1", "p_true"), ("test macro-F1", "test true-class probability")):
        axis.plot(d.k, d[f"top_{key}"], "o-", color=INK, label="most attended tiles removed")
        axis.plot(d.k, d[f"random_{key}"], "s--", color=MID, mfc="white", label="random tiles removed")
        axis.axhline(d[f"full_{key}"].iloc[0], color=LIGHT, lw=0.8, label="nothing removed")
        axis.set(xscale="log", xlabel="tiles removed (k)", ylabel=label)
        axis.set_xticks(list(d.k), [str(k) for k in d.k])
    axes[0].legend(frameon=False)
    figure.tight_layout(pad=0.5)
    return figure, [root / path]


def heatmaps(root: Path) -> Result:
    folder = root / "results/explain/heatmaps/val"
    files = sorted(folder.glob("*.png")) if folder.is_dir() else []
    if not files:
        raise Missing("results/explain/heatmaps/val/*.png")
    chosen = []
    for cls in CLASSES:
        mine = [f for f in files if f.name.startswith(f"{cls}_")]
        good = [f for f in mine if "_correct_1_" in f.name]
        if not mine:
            raise Missing(f"results/explain/heatmaps/val/{cls}_*.png")
        chosen.append((good or mine)[0])
    figure, axes = plt.subplots(2, 2, figsize=(PAGE, PAGE * 0.55))
    for axis, f in zip(axes.ravel(), chosen):
        axis.imshow(plt.imread(f))
        axis.axis("off")
    figure.tight_layout(pad=0.3)
    return figure, chosen


# the two linear-probe points are close together: the 224 px label goes below its point so the labels do not overlap
COST_LABEL_OFFSETS = {"probe_224": (3, -12)}


def cost(root: Path) -> Result:
    path = f"{mt.TABLES}/cost_results.csv"
    c = mt.read_table(root, path)
    lacking = [col for col in COST_COLUMNS if col not in c.columns]
    if lacking:
        raise Missing(f"{path} (needs columns {', '.join(lacking)})")
    figure, axis = plt.subplots(figsize=(COLUMN, 2.4))
    for k, row in enumerate(c.itertuples()):
        axis.plot(row.seconds_per_photo, 100 * row.macro_f1, MARKERS[k % len(MARKERS)], color=INK,
                  mfc=INK if k % 2 == 0 else "white")
        axis.annotate(mt.name_of(row.method), (row.seconds_per_photo, 100 * row.macro_f1), fontsize=5,
                      xytext=COST_LABEL_OFFSETS.get(row.method, (3, 2)), textcoords="offset points")
    axis.set(xscale="log", xlabel="time per photo, end to end (s)", ylabel="test macro-F1 (%)")
    axis.grid(color=LIGHT, lw=0.4)
    figure.tight_layout(pad=0.5)
    return figure, [root / path]


NOTES = {"heatmaps": "One example per class: the most confident correct photo ('correct, rank 1') of the Stage 12 example rule, "
                     "chosen from the validation predictions and drawn from validation photos. It is the best case of each class, "
                     "not a typical one; say so in the caption.",
         "confusion": "Row-normalised (% of the true class); the bold row is hispa, the pre-declared diagnostic."}


@dataclass(frozen=True)
class Spec:
    name: str
    stage: str
    build: Callable[[Path], Result]


SPECS = [Spec(name, mt.PROMISED_FIGURES[name][0], build) for name, build in (
    ("samples", samples), ("confusion", confusion), ("partition_order", partition_order), ("ablations", ablations),
    ("paddy_zoom", paddy_zoom), ("deletion", deletion), ("heatmaps", heatmaps), ("cost", cost))]


def build_all(root: Path = mt.REPO_ROOT, out: Path | None = None) -> dict:
    out = out or root / FIGURES
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"code_version": get_code_version(), "figures": {}}
    for spec in SPECS:
        stem = mt.PROMISED_FIGURES[spec.name][1]
        try:
            figure, sources = spec.build(root)
        except Missing as missing:
            for ext in ("png", "pdf"):                       # an older picture must not pass for the current one
                (out / f"{stem}.{ext}").unlink(missing_ok=True)
            report["figures"][spec.name] = {"status": "pending", "stage": spec.stage, "reason": f"missing {missing.path}"}
            print(f"{spec.name:16s} pending ({missing.path})")
            continue
        for ext in ("png", "pdf"):
            figure.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
        plt.close(figure)
        rels = [str(p.relative_to(root)) for p in sources]
        report["figures"][spec.name] = {"status": "present", "stage": spec.stage, "files": [f"{stem}.png", f"{stem}.pdf"],
                                        "sources": {rel: mt.sha256(root / rel) for rel in sorted(set(rels))}}
        if spec.name in NOTES:
            report["figures"][spec.name]["note"] = NOTES[spec.name]
        print(f"{spec.name:16s} present -> {stem}.png/.pdf")
    (out / "final_figures.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=mt.REPO_ROOT, help="the checkout to read and write")
    build_all(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
