"""Stage 2a: how many photos have plain vs cluttered backgrounds, overall and per class.

A class-stratified random sample (``per_class`` photos per class, fixed seed) is frozen once. Two team
members label every photo alone, in their own shuffled order, without seeing its class: P plain /
C cluttered, plus yes/no flags for several leaves, a hand or fingers, and a strong shadow. Agreement is
Cohen's kappa; disagreements on the background are then settled together, and both original labels are kept.

    uv run python -m src.background_audit sample                    # once: freeze the sample (committed)
    uv run python -m src.background_audit label --rater <name>      # each rater, alone
    uv run python -m src.background_audit settle                    # both raters together
    uv run python -m src.background_audit results                   # writes data/background_audit.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import binomtest

from src import review_tool
from src.config import REPO_ROOT, dataset_dir, load_config
from src.inventory import read_inventory

CHOICES = {"p": "plain", "c": "cluttered"}
FLAGS = {"m": "multiple_leaves", "h": "hand", "s": "strong_shadow"}
COLUMNS = ["photo_id", "rater", "background", *FLAGS.values(), "labelled_at"]
INSTRUCTIONS = ("Background: P plain/uniform   C cluttered (other leaves, soil, water, hands, objects)\n"
                "Flags: M several leaves   H hand or fingers   S strong shadow   Enter = accept   "
                "Backspace = undo   Esc = stop")


def draw_sample(inventory: pd.DataFrame, per_class: int, seed: Any) -> pd.DataFrame:
    """``per_class`` random photos from each class (seeded, independent of row order), sorted by photo_id."""
    rng = np.random.default_rng(seed)
    opened = inventory[inventory["opened_ok"]].sort_values("photo_id", ignore_index=True)
    picks = [group.iloc[np.sort(rng.choice(len(group), per_class, replace=False))]
             for _, group in opened.groupby("class", sort=True)]
    return pd.concat(picks)[["photo_id", "class"]].sort_values("photo_id", ignore_index=True)


def sample(config: dict[str, Any]) -> pd.DataFrame:
    """The frozen sample; drawn and written on the first call only."""
    settings = config["background"]
    path = REPO_ROOT / settings["sample_file"]
    if not path.exists():
        inventory = read_inventory(config["inventory"]["files"]["primary"])
        draw_sample(inventory, settings["per_class"], config["seed"]).to_csv(path, index=False)
    return pd.read_csv(path, dtype=str)


def show_photo(config: dict[str, Any], photo_id: str) -> Image.Image:
    with Image.open(dataset_dir(config, "primary") / photo_id) as image:
        image.draft("RGB", (1400, 1400))
        return ImageOps.exif_transpose(image).convert("RGB")


def ask_photos(photo_ids: list[str], config: dict[str, Any], who: str, ask: Any) -> bool:
    """Collect ``who``'s label for every photo; True when all are labelled."""
    def make_row(item: dict[str, Any], answer: str, flags: set[str]) -> dict[str, Any]:
        return {"photo_id": item["photo_id"], "rater": who, "background": answer,
                **{flag: flag in flags for flag in FLAGS.values()},
                "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    return review_tool.collect(
        [{"photo_id": p} for p in photo_ids], "photo_id",
        REPO_ROOT / config["background"]["labels_dir"] / f"{who}.csv", COLUMNS,
        show=lambda item: [show_photo(config, item["photo_id"])],
        title=lambda item, n, total: f"{who}: photo {n}/{total}\n{INSTRUCTIONS}",
        choices=CHOICES, make_row=make_row, toggles=FLAGS, ask=ask)


def label(config: dict[str, Any], rater: str, ask: Any = review_tool.ask) -> None:
    """One rater labels every sampled photo alone, in their own shuffled order (no class shown)."""
    review_tool.refuse_newcomer(REPO_ROOT / config["background"]["labels_dir"], rater)
    photo_ids = sample(config)["photo_id"].tolist()
    order = review_tool.person_order(config["seed"], rater, len(photo_ids))
    if ask_photos([photo_ids[i] for i in order], config, rater, ask):
        print("All photos labelled.")


def labels(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(sample, every rater's rows, one row per sampled photo with each rater's background)."""
    folder = REPO_ROOT / config["background"]["labels_dir"]
    data = review_tool.people_rows(folder, COLUMNS, "rater", config["reviewers"], "photo_id")
    chosen = sample(config)
    wide = data.pivot(index="photo_id", columns="rater", values="background").reindex(chosen["photo_id"])
    return chosen, data, wide


def settle(config: dict[str, Any], ask: Any = review_tool.ask) -> None:
    """Both raters together decide the background of every photo they disagreed on."""
    _, _, wide = labels(config)
    if wide.shape[1] != 2 or wide.isna().any().any():
        raise SystemExit("both raters must finish every photo before settling")
    split = wide.index[wide.iloc[:, 0] != wide.iloc[:, 1]].tolist()
    if ask_photos(split, config, review_tool.SETTLED, ask):
        print(f"All {len(split)} disagreements settled.")


def share(hits: int, n: int) -> dict[str, float]:
    """Proportion with its exact (Clopper-Pearson) 95% confidence range."""
    ci = binomtest(hits, n).proportion_ci(confidence_level=0.95)
    return {"n": n, "share": hits / n, "low": ci.low, "high": ci.high}


def results(config: dict[str, Any]) -> dict[str, Any]:
    """Kappa, plain shares overall and per class (settled labels), flag rates per rater (reads only)."""
    chosen, data, wide = labels(config)
    answered = data.groupby("rater")["photo_id"].nunique().to_dict()
    if wide.shape[1] != 2 or wide.isna().any().any():
        return {"complete": False, "answered": answered, "sampled": len(chosen)}
    first, second = wide.columns
    settled = review_tool.read_rows(REPO_ROOT / config["background"]["labels_dir"]
                                    / f"{review_tool.SETTLED}.csv", COLUMNS)
    final, agree, kappa = review_tool.agreement(wide, settled.set_index("photo_id")["background"])
    if final.isna().any():
        return {"complete": False, "answered": answered, "sampled": len(chosen),
                "unsettled": int(final.isna().sum())}
    flags = data.pivot(index="photo_id", columns="rater", values=list(FLAGS.values())).reindex(wide.index)
    flags.columns = [f"{flag}_{rater}" for flag, rater in flags.columns]
    out = chosen.set_index("photo_id").assign(**{f"background_{r}": wide[r] for r in (first, second)},
                                              settled=~agree, background=final).join(flags).reset_index()
    plain = out["background"] == "plain"
    by_class = {c: share(int(plain[out["class"] == c].sum()), int((out["class"] == c).sum()))
                for c in sorted(out["class"].unique())}
    flag_rates = {flag: {r: float((out[f"{flag}_{r}"].astype(str) == "True").mean()) for r in (first, second)}
                  for flag in FLAGS.values()}
    return {"complete": True, "raters": [first, second], "kappa": kappa, "agreement": float(agree.mean()),
            "n_settled": int((~agree).sum()), "plain": share(int(plain.sum()), len(out)),
            "plain_by_class": by_class, "flag_rates": flag_rates, "table": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["sample", "label", "settle", "results"])
    parser.add_argument("--rater", help="your first name in lower case (for 'label')")
    parser.add_argument("--config", default="configs/audits.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "sample":
        chosen = sample(config)
        print(f"{len(chosen)} photos ({chosen['class'].value_counts().to_dict()}) "
              f"-> {config['background']['sample_file']}")
    elif args.action == "label":
        name = (args.rater or "").strip().lower()
        if name not in config["reviewers"]:
            parser.error(f"--rater must be one of {config['reviewers']}")
        label(config, name)
    elif args.action == "settle":
        settle(config)
    else:
        result = results(config)
        if not result["complete"]:
            print(f"Not complete: {result}")
            return
        result["table"].to_csv(REPO_ROOT / config["background"]["audit_file"], index=False)
        p = result["plain"]
        kappa = "n/a" if result["kappa"] is None else f"{result['kappa']:.2f}"
        print(f"plain {p['share']:.1%} (95% CI {p['low']:.1%}-{p['high']:.1%}, n = {p['n']}); "
              f"kappa {kappa}; {result['n_settled']} disagreements settled")


if __name__ == "__main__":
    main()
