"""A small fake checkout for the Stage 14 tests: the same files, folders and column names the real stages write,
with made-up values. Used by test_make_tables.py, test_make_figures.py and test_build_submission_notebook.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]
METHODS = ["inceptionv3_224", "efficientnetv2s_224", "convnext_tiny_224", "swin_tiny_224", "probe_224", "probe_512",
           "mean", "max", "attention", "whole224", "whole512"]
FOLDER = {"healthy": "Healthy", "leaf_blast": "LeafBlast", "brown_spot": "BrownSpot", "hispa": "Hispa"}


def score_columns() -> dict:
    cols = {}
    for key, base in (("macro_f1", 0.70), ("balanced_accuracy", 0.71)):
        cols.update({f"{key}_mean": base, f"{key}_sd": 0.03, f"{key}_pooled": base + 0.01, f"{key}_photo_low": base - 0.02,
                     f"{key}_photo_high": base + 0.03, f"{key}_block_low": base - 0.04, f"{key}_block_high": base + 0.05})
    return cols


def write_csv(root: Path, rel: str, frame: pd.DataFrame) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / rel, index=False)


def paired(names: list[str], key: str = "B", wins: str = "round_wins") -> pd.DataFrame:
    rows = []
    for n, name in enumerate(names):
        for metric in ("macro_f1", "balanced_accuracy"):
            rows.append({key: name, "metric": metric, "difference": 0.01 * (n + 1), "photo_low": 0.001 * n, "photo_high": 0.02 * (n + 1),
                         "block_low": -0.01, "block_high": 0.03 * (n + 1), wins: 3 + n % 3, "rounds": 5,
                         "verdict_photo": "A beats B" if n % 2 == 0 else "no clear difference",
                         "verdict_block": "no clear difference"})
    return pd.DataFrame(rows)


def make_runs(root: Path, runs: list[str], seed: int) -> None:
    """Five runs, one per round, each with test predictions on its own 8 photos (every photo once overall)."""
    rng = np.random.default_rng(seed)
    for round_number, run in enumerate(runs, start=1):
        folder = root / "runs" / run
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(8):
            true = CLASSES[i % 4]
            pred = true if rng.random() < 0.7 else CLASSES[rng.integers(4)]
            rows.append({"round": round_number, "partition": "test", "photo_id": f"p{round_number}_{i}", "true_class": true,
                         "pred_class": pred, **{f"prob_{c}": 0.25 for c in CLASSES}})
        pd.DataFrame(rows).to_csv(folder / "test_predictions.csv", index=False)
        (folder / "test_metrics.json").write_text(json.dumps({"macro_f1": 0.6 + 0.01 * round_number,
                                                              "balanced_accuracy": 0.62 + 0.01 * round_number}))


def make_fixture(root: Path, *, paddy: bool = False, explain: bool = False, cost: bool = False) -> Path:
    t = "results/tables"
    final = pd.DataFrame([{"method": m, "ece": 0.05, **score_columns()} for m in METHODS])
    write_csv(root, f"{t}/session_final.csv", final)
    write_csv(root, f"{t}/followup_final.csv", final[final.method != "whole224"].assign(macro_f1_mean=0.8))
    write_csv(root, f"{t}/session_paired.csv", paired(METHODS[:-3] + ["whole224"]))
    write_csv(root, f"{t}/followup_paired.csv", paired(METHODS[:4]))
    write_csv(root, f"{t}/session_conditions_scores.csv", pd.DataFrame([
        {"model": "inceptionv3", "condition": c, "ece": 0.05, **score_columns()} for c in ("224", "128", "224_otsu")]
        + [{"model": "convnext_tiny", "condition": "448", "ece": 0.05, **score_columns()}]))
    write_csv(root, f"{t}/session_ablations.csv", pd.DataFrame([
        {"variant": v, "question": f"what {v} changes", "val_macro_f1_mean": 0.75, "val_macro_f1_sd": 0.05,
         "difference_vs_main": -0.01 * n, "rounds_better_than_main": "–" if n == 0 else n % 3}
        for n, v in enumerate(["main", "otsu_on", "native_scale", "encoder_e3"])]))
    po = [{"model": m, "condition": c, "meaning": f"set-up {c}", "macro_f1_pooled": 0.75 + 0.03 * n, "macro_f1_low": 0.73 + 0.03 * n,
           "macro_f1_high": 0.77 + 0.03 * n, "macro_f1_inflation": 0.03 * n, "macro_f1_inflation_low": 0.01 * n,
           "macro_f1_inflation_high": 0.04 * n, "macro_f1_reading": "reference" if n == 0 else "clearly above A",
           "balanced_accuracy_pooled": 0.76, "balanced_accuracy_inflation": 0.02 * n}
          for m in ("efficientnetv2s", "attention") for n, c in enumerate(("A", "L", "B"))]
    write_csv(root, f"{t}/partition_order_inflation.csv", pd.DataFrame(po))
    write_csv(root, f"{t}/followup_shortcuts.csv", pd.DataFrame([
        {"baseline": b, "locked_macro_f1_mean": 0.8, "locked_macro_f1_sd": 0.01, "locked_balanced_accuracy_mean": 0.8,
         "locked_balanced_accuracy_sd": 0.01, "session_macro_f1_mean": 0.3, "session_macro_f1_sd": 0.1,
         "session_balanced_accuracy_mean": 0.31, "session_balanced_accuracy_sd": 0.1} for b in ("majority", "time_neighbour")]))
    # the runs of every method, and the frozen plans that list them
    selection, plan = {"methods": {}}, {"runs": {}}
    for n, method in enumerate(METHODS):
        runs = [f"stage7_session_{method}_r{r}" for r in range(1, 6)]
        make_runs(root, runs, seed=n)
        selection["methods"][method] = {"runs": runs}
        plan["runs"].update({run: {} for run in runs})
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs/session_selection.json").write_text(json.dumps(selection))
    (root / "runs/session_evaluation_plan.json").write_text(json.dumps(plan))
    conditions = {"convnext_tiny_448": [f"stage5_convnext_tiny_session_448_r{r}" for r in range(1, 6)]}
    make_runs(root, conditions["convnext_tiny_448"], seed=99)
    (root / "runs/session_conditions_selection.json").write_text(json.dumps({"configurations": conditions}))
    (root / "runs/session_conditions_plan.json").write_text(json.dumps({"runs": {r: {} for r in conditions["convnext_tiny_448"]}}))
    ablation_runs = {}
    for variant in ("otsu_on", "native_scale"):
        runs = [f"stage7_session_ablate_{variant}_r{r}" for r in range(1, 6)]
        make_runs(root, runs, seed=7)
        ablation_runs.update({run: {} for run in runs})
    (root / "runs/session_ablations_plan.json").write_text(json.dumps({"runs": ablation_runs}))
    report = ["# Stage 8", "", "## 1. Validation macro-F1", "", "| variant | x |", "|---|---|", "| main | 1 |", "",
              "## 2. Headline ablations on test (main against variant)", "",
              "| variant | metric | difference (points) | photo range | block range | rounds won by main | reading (photo range) | verdict (block range) |",
              "|---|---|---|---|---|---|---|---|"]
    for variant in ("otsu_on", "native_scale"):
        for metric in ("macro_f1", "balanced_accuracy"):
            report.append(f"| {variant} | {metric} | 5.2 | [3.8, 6.7] | [3.3, 7.3] | 4/5 | main beats variant | main beats variant |")
    (root / "docs/reports").mkdir(parents=True, exist_ok=True)
    (root / "docs/reports/session_ablation_report.md").write_text("\n".join(report) + "\n")
    (root / "docs/commitments.md").write_text(
        "# Written commitments\n\n"
        "| ID | Commitment | Delivered by (stage) | Where it is delivered |\n|---|---|---|---|\n"
        "| P1-6 | Five repetitions; mean ± std | All stages (5 rounds) | ⬜ |\n"
        "| P1-10 | Hispa recall reported | Stages 7, 14 | ⬜ |\n"
        "| P1-9 | No generative oversampling | Stages 5, 7 | ⬜ |\n"
        "| O6 | Zero-shot evaluation and attention maps | Stages 10, 12 | ⬜ |\n")
    if paddy:
        rows = []
        for method in ("attention", "swin_tiny_224"):
            for zoom in (("native", "matched", "wide") if method == "attention" else ("-",)):
                rows.append({"method": method, "zoom": zoom, "headline": zoom in ("native", "-"), "macro_f1_mean": 0.6, "macro_f1_sd": 0.05,
                             "balanced_accuracy_mean": 0.61, "balanced_accuracy_sd": 0.05})
        write_csv(root, f"{t}/paddy_results.csv", pd.DataFrame(rows))
    if explain:
        write_csv(root, f"{t}/explain_deletion_test.csv", pd.DataFrame([
            {"k": k, "full_macro_f1": 0.78, "top_macro_f1": 0.6 - 0.05 * n, "random_macro_f1": 0.75 - 0.01 * n, "full_p_true": 0.8,
             "top_p_true": 0.6 - 0.05 * n, "random_p_true": 0.78, "gap_p_true": 0.1 * n, "gap_low": 0.05, "gap_high": 0.15,
             "gap_block_low": 0.02, "gap_block_high": 0.2, "round_wins": 5, "rounds": 5,
             "verdict": "top-k matters more than random-k"} for n, k in enumerate((1, 4, 16))]))
        for cls in CLASSES:
            folder = root / "results/explain/heatmaps/val"
            folder.mkdir(parents=True, exist_ok=True)
            plt.imsave(folder / f"{cls}_correct_1_abc.png", np.random.default_rng(0).random((20, 20)))
    if cost:
        write_csv(root, f"{t}/cost_results.csv", pd.DataFrame([
            {"method": m, "seconds_per_photo": s, "macro_f1": f} for m, s, f in (("attention", 0.9, 0.78), ("swin_tiny_224", 0.05, 0.76))]))
    return root


def add_photos(root: Path) -> None:
    """Tiny photos and the locked split file that lists them, for the sample-photo figure."""
    from PIL import Image

    rows = []
    for cls, folder in FOLDER.items():
        for i in range(5):
            path = root / "data/raw/rice_diseases/LabelledRice/Labelled" / folder / f"{i}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((30, 40, 3), 40 * i, dtype="uint8")).save(path)
            rows.append({"photo_id": f"{folder}/{i}.jpg", "class": cls})
    write_csv(root, "splits/folds.csv", pd.DataFrame(rows))
