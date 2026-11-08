from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.metrics import probabilities
from src.sanity_baselines import BASELINES
from src.stats import reliability

config = load_config("configs/base.yaml")
classes = config["classes"]
fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
for ax, baseline in zip(axes.ravel(), BASELINES):
    folder = Path("runs") / f"stage4_{baseline}"
    table = pd.read_csv(folder / "predictions.csv")
    test = table[table.partition == "test"]
    target = test.true_class.map({c: i for i, c in enumerate(classes)}).to_numpy()
    bins = reliability(probabilities(test, classes), target, config["evaluation"]["calibration_bins"])
    bins.to_csv(folder / "reliability_test.csv")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    seen = bins[bins.photos > 0]
    ax.plot(seen.confidence, seen.accuracy, "o-", color="#236b8e")
    ax.set_title(folder.name.removeprefix("stage4_"))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy")
    ax.grid(alpha=0.2)
fig.suptitle("Stage 4 calibration: pooled out-of-fold test predictions")
fig.tight_layout()
fig.savefig(out / "stage4_reliability.svg")
fig.savefig(out / "stage4_reliability.png", dpi=160)
