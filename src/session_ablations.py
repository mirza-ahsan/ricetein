"""Stage 8 ablations on the session-blocked split (declared 2026-11-20).

Each variant changes one thing in the main configuration (E1 + attention on the session-split selection) and
keeps everything else. **Every variant gets the same tuning budget: none.** All variants are trained with the
settings that the 40-trial study of the main configuration selected (``runs/session_selection.json``), five
rounds, one seed. The two headline ablations (Otsu, native resolution) are tested once; the others are
reported on validation photos and marked so.

    uv run python -m src.session_ablations fit [name ...]   # variants whose cache is complete (default: all); resumable
    uv run python -m src.session_ablations evaluate   # test the headline variants once
    uv run python -m src.session_ablations report     # docs/reports/session_ablation_report.md and its table
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src import followup_report as fr
from src import train_mil
from src.config import REPO_ROOT, cache_folder, load_config
from src.embed import completed_cache
from src.followup import ROUNDS, evaluate_once, read_json
from src.inventory_report import md_table
from src.records import completed_run, get_default_runs_dir
from src.session_study import require_intact
from src.split import read_folds

MAIN = "configs/mil/session_attention.yaml"
PLAN = "session_ablations_plan.json"
REPORT, TABLE = "docs/reports/session_ablation_report.md", "results/tables/session_ablations.csv"
PARTS = {"c": ["cls"], "m": ["mean"], "cm": ["cls", "mean"]}
# name -> what changes. mil: the changed training settings (cache = another description cache, override = other
# selected parameters); tested: evaluated on test (the headline ablations of Stage 8); the rest stay
# on validation.
VARIANTS: dict[str, dict[str, Any]] = {
    "otsu_on": {"mil": {"cache": "e1_otsu"}, "tested": True, "question": "Tiles of Otsu-cleaned photos"},
    "native_scale": {"mil": {"cache": "e1_native"}, "tested": True,
                     "question": "Native resolution instead of the common leaf scale"},
    "tile448": {"mil": {"cache": "e1_tile448"}, "question": "448 px tiles shrunk to 224 px"},
    "encoder_e2": {"mil": {"cache": "e2_convnext_tiny", "override": {"descriptor": "c"}},
                   "question": "Supervised ConvNeXt-T encoder (pooled output only, so the class-token description)"},
    "encoder_e3": {"mil": {"cache": "e3_dinov2_vitb14"}, "question": "DINOv2-B encoder"},
    "encoder_e4": {"mil": {"cache": "e4_dinov3_vits16"}, "question": "DINOv3-S encoder (384 dimensions)"},
    "unweighted_loss": {"mil": {"class_weighting": False}, "question": "Unweighted loss"},
    "single_version": {"mil": {"versions": 1}, "question": "Clean photos only (no cached augmentation)"},
    "tile_dropout_on": {"mil": {"override": {"tile_dropout": 0.15}},
                        "question": "Tile dropout 0.15 (the selected setting is 0)"},
    "description_c": {"mil": {"override": {"descriptor": "c"}}, "question": "Class token only"},
    "description_m": {"mil": {"override": {"descriptor": "m"}}, "question": "Patch average only"},
}
HEADLINE = [name for name, spec in VARIANTS.items() if spec.get("tested")]


def variant_config(name: str) -> dict[str, Any]:
    config = load_config(MAIN)
    config["mil"] = {**config["mil"], "name": f"session_ablate_{name}", **VARIANTS[name]["mil"]}
    return config


def runs_of(name: str) -> list[str]:
    return [f"stage7_session_ablate_{name}_r{r}" for r in ROUNDS]


def main_runs() -> list[str]:
    """The five runs of the main configuration; its selected study must be the declared one, whose settings
    every variant reuses (``tuned_from: session_attention``)."""
    entry = read_json(get_default_runs_dir() / "session_selection.json")["methods"]["attention"]
    if entry["selected"] != "declared":
        raise ValueError("the main attention configuration is the wide study; the variants reuse the declared one")
    return entry["runs"]


def pack_one_key_at_a_time(config: dict[str, Any]) -> None:
    """Build the packed descriptor files one kind at a time, before ``train_mil`` needs them: the frozen
    ``BagStore`` fails for good when one kind is on disk and the other has to be built."""
    mil = config["mil"]
    descriptor = {**read_json(get_default_runs_dir() / "stage7_tune_session_attention" / "metrics.json")["best_params"],
                  **mil.get("override", {})}["descriptor"]
    ids = sorted(read_folds(config)[0].photo_id)
    folder, fingerprint = cache_folder(mil["cache"]), completed_cache(config, mil["cache"])["spec_sha256"]
    target = cache_folder(f"packed/{mil['cache']}", write=True)
    target.mkdir(parents=True, exist_ok=True)                   # `train_mil.pack` does not create it
    for versions in {mil["versions"], 1}:                       # training reads `versions`, the test reads 1
        for key in PARTS[descriptor]:
            train_mil.pack(folder, target, key, ids, versions, fingerprint)


def fit(names: list[str] | None = None) -> None:
    """Train the variants whose cache is complete. Ends with an error if any is still waiting, so that a queue
    never prints DONE for work it skipped."""
    waiting = []
    for name in names or VARIANTS:
        config = variant_config(name)
        if not (cache_folder(config["mil"]["cache"]) / "complete.json").exists():
            waiting.append(name)
            print(f"{name}: waiting for cache {config['mil']['cache']}", flush=True)
            continue
        pack_one_key_at_a_time(config)
        train_mil.final(config, [config["seed"]])
        print(f"{name}: trained", flush=True)
    if waiting:
        raise SystemExit(f"NOT DONE: {len(waiting)} variants are waiting for their cache: {', '.join(waiting)}")


def trained() -> list[str]:
    return [name for name in VARIANTS if all(completed_run(run, ("metrics.json", "predictions.csv", "model.pt"))
                                             for run in runs_of(name))]


def evaluate_headline() -> None:
    """Test the headline variants once, after every one of them is trained (the plan cannot change later)."""
    if not set(HEADLINE) <= set(trained()):
        raise RuntimeError(f"train every headline variant first: {', '.join(HEADLINE)}")
    evaluate_once([run for name in HEADLINE for run in runs_of(name)], PLAN)


def validation_rows() -> list[dict[str, Any]]:
    """Validation macro-F1 of each round: the main configuration, its other training seeds (the noise floor), and
    every trained variant."""
    root, main = get_default_runs_dir(), main_runs()

    def val(runs: list[str]) -> np.ndarray:
        return np.array([read_json(root / run / "metrics.json")["val_macro_f1"] for run in runs])

    reference = val(main)
    seeds = [f"{main[0].rsplit('_r', 1)[0]}_s{seed}_r{r}" for seed in (1, 2) for r in ROUNDS]
    rows = [{"variant": "main", "question": "E1 + attention, selected settings", "scores": reference},
            {"variant": "main, seeds 1 and 2", "question": "The same configuration, other training seeds (noise floor)",
             "scores": np.array([val(seeds[:5]), val(seeds[5:])]).mean(0)}]
    rows += [{"variant": name, "question": VARIANTS[name]["question"], "scores": val(runs_of(name))} for name in trained()]
    return [{"variant": r["variant"], "question": r["question"], "val_macro_f1_mean": r["scores"].mean(),
             "val_macro_f1_sd": r["scores"].std(ddof=1), "difference_vs_main": r["scores"].mean() - reference.mean(),
             "rounds_better_than_main": "–" if r["variant"] == "main" else int((r["scores"] > reference).sum())}
            for r in rows]


def reading(row: Any) -> str:
    """Who is better, from the range and the round wins (A = main, B = variant)."""
    if row.photo_low > 0 and row.round_wins >= 4:
        return "main beats variant"
    if row.photo_high < 0 and row.round_wins <= 1:
        return "variant beats main"
    return "no clear difference"


def report() -> None:
    config = load_config(MAIN)
    table = pd.DataFrame(validation_rows())
    table.to_csv(REPO_ROOT / TABLE, index=False)
    pct = lambda x: f"{100 * x:.1f}"   # noqa: E731
    shown = pd.DataFrame({"variant": table["variant"], "what changes": table["question"],
                          "validation macro-F1": [f"{pct(m)} ± {pct(s)}" for m, s in zip(table.val_macro_f1_mean, table.val_macro_f1_sd)],
                          "difference vs main (points)": [f"{100 * d:+.1f}" for d in table.difference_vs_main],
                          "rounds better than main": table.rounds_better_than_main})
    lines = ["# Stage 8 ablations on the session-blocked split", "",
             "Built by `src/session_ablations.py` from `runs/`; do not edit by hand. Every variant changes one thing and "
             "uses the settings selected for the main configuration (no tuning for any variant, declared 2026-11-20). "
             "Five rounds, one seed.", "",
             "**Limits.** Validation scores favour the main configuration: its settings were chosen on these same validation "
             "photos, the variants' were not. The settings were also tuned for the E1 class-token plus patch-average "
             "description (1,536 numbers), so a variant on another encoder or description may be handicapped (E4 has 384 "
             "dimensions; E2 gives one pooled vector, so its variant changes the encoder and the description together). "
             "The row \"main, seeds 1 and 2\" shows how much the same configuration moves with the training seed.", "",
             "## 1. Validation macro-F1 (main and all trained variants)", "", md_table(shown)]
    lines += ["", "## 2. Headline ablations on test (main against variant)", ""]
    if (get_default_runs_dir() / PLAN).exists():
        fr.require_tested(PLAN)
        paired = fr.paired_rows(fr.read_predictions(main_runs()), {n: fr.read_predictions(runs_of(n)) for n in HEADLINE},
                                fr.block_of(), config)
        paired["difference (points)"] = (100 * paired.difference).round(1)
        paired["photo range"] = [f"[{pct(l)}, {pct(h)}]" for l, h in zip(paired.photo_low, paired.photo_high)]
        paired["block range"] = [f"[{pct(l)}, {pct(h)}]" for l, h in zip(paired.block_low, paired.block_high)]
        paired["reading (photo range)"] = [reading(r) for r in paired.itertuples()]
        paired["rounds won by main"] = paired.round_wins.astype(str) + "/" + paired.rounds.astype(str)
        paired["verdict (block range)"] = paired.verdict_block.replace({"A beats B": "main beats variant"})
        lines.append(md_table(paired.rename(columns={"B": "variant"})[
            ["variant", "metric", "difference (points)", "photo range", "block range", "rounds won by main",
             "reading (photo range)", "verdict (block range)"]]))
    else:
        lines.append("The headline variants are not tested yet.")
    (REPO_ROOT / REPORT).write_text("\n".join(lines) + "\n")
    print(f"report: {len(table)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["fit", "evaluate", "report"])
    parser.add_argument("names", nargs="*", help="for fit: only these variants")
    args = parser.parse_args()
    if args.names and args.action != "fit":
        parser.error(f"{args.action} takes no names")
    require_intact()
    {"fit": lambda: fit(args.names or None), "evaluate": evaluate_headline, "report": report}[args.action]()
    print(f"SESSION ABLATIONS {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
