"""Supplement completed Stage 4–7 artifacts without fitting or selecting any model.

Run with uv run --locked python scripts/complete_stage_reports.py after the pipeline.
Source, predictions and cache contents are read only. Visual cache review is still required.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import embed, metrics, stats, tiling
from src.config import cache_folder, load_config
from src.experiment_report import configurations, predictions
from src.inventory import read_inventory
from src.sanity_baselines import BASELINES, summarise


def calibration(table, name, config):
    test = table.loc[table.partition == "test"]
    classes = config["classes"]
    true = test.true_class.map({c: i for i, c in enumerate(classes)}).to_numpy()
    bins = stats.reliability(
        metrics.probabilities(test, classes), true, config["evaluation"]["calibration_bins"]
    )
    target = Path("results/tables")
    target.mkdir(parents=True, exist_ok=True)
    bins.to_csv(target / f"{name}_reliability.csv")
    fig, ax = plt.subplots(figsize=(5, 5))
    seen = bins[bins.photos > 0]
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(seen.confidence, seen.accuracy, "o-", color="#236b8e")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        title=f"{name}: pooled test predictions",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    figures = Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}_reliability.svg")
    plt.close(fig)
    result = summarise(table, config)
    result["per_round_test"] = metrics.per_round(table, classes)
    result["per_round_validation"] = metrics.per_round(table, classes, "val")
    (target / f"{name}_full_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(result["pooled_test"]["per_class"]).T.to_csv(target / f"{name}_per_class.csv")
    pd.DataFrame(result["pooled_test"]["confusion"], index=classes, columns=classes).to_csv(
        target / f"{name}_confusion.csv"
    )


def cache_counts(config):
    rows, durations = [], []
    for name in config["pipeline"]["caches"]:
        c = load_config(f"configs/embed/{name}.yaml")
        e = c["embed"]
        folder = cache_folder(e["name"])
        embed.completed_cache(c, e["name"])
        inventory = read_inventory(c["inventory"]["files"]["primary"])
        inventory = inventory.loc[inventory.opened_ok].set_index("photo_id")
        factors = embed.scale_factors(c, inventory.index.tolist())
        for version in range(e["versions"]):
            for photo_id, photo in inventory.iterrows():
                change = embed.version_settings(c["augment"], c["seed"], version, photo_id)
                scale = factors[photo_id] * change["zoom"]
                w, h = round(photo.width * scale), round(photo.height * scale)
                if change["rot90"] % 2:
                    w, h = h, w
                expected = (
                    np.zeros((len(e["whole_sizes"]), 4), dtype=np.int32)
                    if e.get("whole_sizes")
                    else tiling.boxes(w, h, e["tile"])
                )
                with np.load(folder / f"v{version}" / f"{embed.photo_key(photo_id)}.npz") as data:
                    np.testing.assert_array_equal(data["boxes"], expected)
                    if len(data["cls"]) != len(expected):
                        raise ValueError(f"wrong descriptor count: {name}, {photo_id}, {version}")
                rows.append(
                    {
                        "cache": e["name"],
                        "photo_id": photo_id,
                        "class": photo["class"],
                        "version": version,
                        "tiles": len(expected),
                    }
                )
        for path in sorted(folder.glob("record_*.json")):
            record = json.loads(path.read_text())
            seconds = (
                datetime.fromisoformat(record["finished"]) - datetime.fromisoformat(record["started"])
            ).total_seconds()
            durations.append(
                {
                    "cache": e["name"],
                    "record": str(path),
                    "seconds": seconds,
                    "photo_versions": record["to_do"],
                    "code_version": record["code_version"],
                }
            )
    target = Path("results/tables")
    target.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame(rows)
    counts.to_csv(target / "cache_tile_counts.csv", index=False)
    counts.groupby(["cache", "version", "class"]).tiles.agg(
        ["count", "sum", "min", "median", "mean", "max"]
    ).to_csv(target / "cache_tile_distribution.csv")
    pd.DataFrame(durations).to_csv(target / "cache_extraction_sessions.csv", index=False)
    # Interrupted sessions have no finished record; timing totals are explicit lower bounds.
    (Path("docs/reports") / "cache_geometry_checks.json").write_text(
        json.dumps(
            {
                "all_photo_version_boxes_match_expected_geometry": True,
                "checked_photo_versions": len(counts),
                "timing_scope": "finished extraction sessions only; interrupted sessions are omitted",
                "visual_review": "required separately",
                "kaggle_availability": "must be verified separately after private dataset upload",
            },
            indent=2,
        )
    )


def main():
    config = load_config("configs/pipeline.yaml")
    for baseline in BASELINES:
        folder = Path("runs") / f"stage4_{baseline}"
        calibration(pd.read_csv(folder / "predictions.csv"), folder.name, config)
    stage5, stage7 = configurations()
    for item in stage5 + stage7:
        calibration(predictions(item["runs"]), item["name"], config)
    cache_counts(config)
    print("Full metrics, calibration charts, cache geometry and timing supplements saved.")


if __name__ == "__main__":
    main()
