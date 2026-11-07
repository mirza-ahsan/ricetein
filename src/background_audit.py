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

import pandas as pd
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


def draw_sample(config: dict[str, Any]) -> pd.DataFrame:
    """Draw and freeze the sample (the ``sample`` command); refuses to redraw an existing one."""
    path = REPO_ROOT / config["background"]["sample_file"]
    if path.exists():
        raise SystemExit(f"{path} is frozen; it is never redrawn")
    inventory = read_inventory(config["inventory"]["files"]["primary"])
    chosen = review_tool.stratified_sample(inventory, config["background"]["per_class"], config["seed"])
    chosen.to_csv(path, index=False)
    return chosen


def sample(config: dict[str, Any]) -> pd.DataFrame | None:
    """The frozen sample, or None before ``sample`` has been run."""
    path = REPO_ROOT / config["background"]["sample_file"]
    return pd.read_csv(path, dtype=str) if path.exists() else None


def ask_photos(photo_ids: list[str], config: dict[str, Any], who: str, ask: Any) -> bool:
    """Collect ``who``'s label for every photo; True when all are labelled."""
    root = dataset_dir(config, "primary")

    def make_row(item: dict[str, Any], answer: str, flags: set[str]) -> dict[str, Any]:
        return {"photo_id": item["photo_id"], "rater": who, "background": answer,
                **{flag: flag in flags for flag in FLAGS.values()},
                "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    return review_tool.collect(
        [{"photo_id": p} for p in photo_ids], "photo_id",
        REPO_ROOT / config["background"]["labels_dir"] / f"{who}.csv", COLUMNS,
        show=lambda item: [review_tool.load_photo(root / item["photo_id"], 1400)],
        title=lambda item, n, total: f"{who}: photo {n}/{total}\n{INSTRUCTIONS}",
        choices=CHOICES, make_row=make_row, toggles=FLAGS, ask=ask)


def label(config: dict[str, Any], rater: str, ask: Any = review_tool.ask) -> None:
    """One rater labels every sampled photo alone, in their own shuffled order (no class shown)."""
    review_tool.refuse_newcomer(REPO_ROOT / config["background"]["labels_dir"], rater)
    chosen = sample(config)
    if chosen is None:
        raise SystemExit("run `python -m src.background_audit sample` first")
    photo_ids = chosen["photo_id"].tolist()
    order = review_tool.person_order(config["seed"], rater, len(photo_ids))
    if ask_photos([photo_ids[i] for i in order], config, rater, ask):
        print("All photos labelled.")


def labels(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every rater's rows, one row per sampled photo with each rater's background)."""
    folder = REPO_ROOT / config["background"]["labels_dir"]
    data = review_tool.people_rows(folder, COLUMNS, "rater", config["reviewers"], "photo_id")
    wide = data.pivot(index="photo_id", columns="rater", values="background")
    wide = wide.reindex(sample(config)["photo_id"])
    return data, wide


def settle(config: dict[str, Any], ask: Any = review_tool.ask) -> None:
    """Both raters together decide the background of every photo they disagreed on."""
    _, wide = labels(config)
    if not review_tool.both_done(wide):
        raise SystemExit("both raters must finish every photo before settling")
    split = review_tool.disagreements(wide)
    if ask_photos(split, config, review_tool.SETTLED, ask):
        print(f"All {len(split)} disagreements settled.")


def share(hits: int, n: int) -> dict[str, float]:
    """Proportion with its exact (Clopper-Pearson) 95% confidence range."""
    ci = binomtest(hits, n).proportion_ci(confidence_level=0.95)
    return {"n": n, "share": hits / n, "low": ci.low, "high": ci.high}


def results(config: dict[str, Any]) -> dict[str, Any]:
    """Kappa, plain shares overall and per class (settled labels), flag rates per rater (reads only)."""
    chosen = sample(config)
    if chosen is None:
        return {"complete": False, "answered": {}, "sampled": 0}
    data, wide = labels(config)
    folder = REPO_ROOT / config["background"]["labels_dir"]
    settled = review_tool.settled_answers(folder, COLUMNS, "photo_id", "background")
    status = review_tool.two_person(wide, settled, data.groupby("rater")["photo_id"].nunique().to_dict())
    if not status["complete"]:
        return {**status, "sampled": len(chosen)}
    first, second = status["people"]
    flags = data.pivot(index="photo_id", columns="rater", values=list(FLAGS.values())).reindex(wide.index)
    flags.columns = [f"{flag}_{rater}" for flag, rater in flags.columns]
    out = chosen.set_index("photo_id").assign(**{f"background_{r}": wide[r] for r in (first, second)},
                                              settled=~status["agree"], background=status["final"])
    out = out.join(flags).reset_index()
    plain = out["background"] == "plain"
    by_class = {c: share(int(plain[out["class"] == c].sum()), int((out["class"] == c).sum()))
                for c in sorted(out["class"].unique())}
    flag_rates = {flag: {r: float((out[f"{flag}_{r}"].astype(str) == "True").mean()) for r in (first, second)}
                  for flag in FLAGS.values()}
    return {"complete": True, "raters": [first, second], "kappa": status["kappa"],
            "agreement": float(status["agree"].mean()), "n_settled": int((~status["agree"]).sum()),
            "plain": share(int(plain.sum()), len(out)), "plain_by_class": by_class, "flag_rates": flag_rates,
            "table": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["sample", "label", "settle", "results"])
    parser.add_argument("--rater", help="your first name in lower case (for 'label')")
    parser.add_argument("--config", default="configs/audits.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "sample":
        chosen = draw_sample(config)
        print(f"{len(chosen)} photos ({chosen['class'].value_counts().to_dict()}) "
              f"-> {config['background']['sample_file']}")
    elif args.action == "label":
        label(config, review_tool.person_name(parser, args.rater, config["reviewers"], "--rater"))
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
