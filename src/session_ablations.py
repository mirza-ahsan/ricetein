"""Stage 8 ablations on the session-blocked split (declared 2026-11-20, handbook §24).

Each variant changes one thing in the main configuration (E1 + attention on the session-split selection) and
keeps everything else. **Every variant gets the same tuning budget: none.** All variants are trained with the
settings that the 40-trial study of the main configuration selected (``runs/session_selection.json``), five
rounds, one seed. The two headline ablations (Otsu, native resolution) are tested once; the others are
reported on validation photos and marked so.

    uv run python -m src.session_ablations fit        # every variant whose cache is complete; resumable
    uv run python -m src.session_ablations evaluate   # test the headline variants once
    uv run python -m src.session_ablations report     # docs/reports/session_ablation_report.md and its table
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from src import followup_report as fr
from src import train_mil
from src.config import REPO_ROOT, load_config
from src.embed import completed_cache
from src.followup import ROUNDS, evaluate_once, read_json
from src.records import get_default_runs_dir
from src.session_study import require_intact

MAIN = "configs/mil/session_attention.yaml"
PLAN = "session_ablations_plan.json"
REPORT, TABLE = "docs/reports/session_ablation_report.md", "results/tables/session_ablations.csv"
# name -> what changes. cache: another description cache; mil: other training settings; override: other selected
# parameters. tested: evaluated on test (the headline ablations of handbook Stage 8); the rest stay on validation.
VARIANTS: dict[str, dict[str, Any]] = {
    "otsu_on": {"cache": "e1_otsu", "tested": True, "question": "Tiles of Otsu-cleaned photos"},
    "native_scale": {"cache": "e1_native", "tested": True, "question": "Native resolution instead of the common leaf scale"},
    "tile448": {"cache": "e1_tile448", "question": "448 px tiles shrunk to 224 px"},
    "encoder_e2": {"cache": "e2_convnext_tiny", "override": {"descriptor": "c"},
                   "question": "Supervised ConvNeXt-T encoder (pooled output only)"},
    "encoder_e3": {"cache": "e3_dinov2_vitb14", "question": "DINOv2-B encoder"},
    "encoder_e4": {"cache": "e4_dinov3_vits16", "question": "DINOv3-S encoder"},
    "unweighted_loss": {"mil": {"class_weighting": False}, "question": "Unweighted loss"},
    "single_version": {"mil": {"versions": 1}, "question": "Clean photos only (no cached augmentation)"},
    "tile_dropout_on": {"override": {"tile_dropout": 0.15}, "question": "Tile dropout 0.15 (the selected setting is 0)"},
    "description_c": {"override": {"descriptor": "c"}, "question": "Class token only"},
    "description_m": {"override": {"descriptor": "m"}, "question": "Patch average only"},
}


def variant_config(name: str) -> dict[str, Any]:
    spec, config = VARIANTS[name], load_config(MAIN)
    config["mil"] = {**config["mil"], "name": f"session_ablate_{name}", **spec.get("mil", {}),
                     **({"cache": spec["cache"]} if "cache" in spec else {}),
                     **({"override": spec["override"]} if "override" in spec else {})}
    return config


def runs_of(name: str) -> list[str]:
    return [f"stage7_session_ablate_{name}_r{r}" for r in ROUNDS]


def main_runs() -> list[str]:
    return read_json(get_default_runs_dir() / "session_selection.json")["methods"]["attention"]["runs"]


def fit() -> None:
    for name in VARIANTS:
        config = variant_config(name)
        try:
            completed_cache(config, config["mil"]["cache"])
        except FileNotFoundError:
            print(f"{name}: waiting for cache {config['mil']['cache']}", flush=True)
            continue
        train_mil.final(config, [config["seed"]])
        print(f"{name}: trained", flush=True)


def evaluate_headline() -> None:
    runs = [run for name, spec in VARIANTS.items() if spec.get("tested") for run in runs_of(name)]
    evaluate_once(runs, PLAN)


def validation_rows(names: list[str]) -> list[dict[str, Any]]:
    """Validation macro-F1 of each round, for the main configuration and every trained variant."""
    root = get_default_runs_dir()

    def val(runs: list[str]) -> list[float]:
        return [json.loads((root / run / "metrics.json").read_text())["val_macro_f1"] for run in runs]

    main = np.array(val(main_runs()))
    rows = []
    for name in names:
        scores = np.array(val(runs_of(name)))
        rows.append({"variant": name, "question": VARIANTS[name]["question"], "tested": bool(VARIANTS[name].get("tested")),
                     "val_macro_f1_mean": scores.mean(), "val_macro_f1_sd": scores.std(ddof=1),
                     "difference_vs_main": scores.mean() - main.mean(), "rounds_better_than_main": int((scores > main).sum())})
    rows.append({"variant": "main", "question": "E1 + attention, selected settings", "tested": True,
                 "val_macro_f1_mean": main.mean(), "val_macro_f1_sd": main.std(ddof=1), "difference_vs_main": 0.0,
                 "rounds_better_than_main": 0})
    return rows


def trained() -> list[str]:
    root = get_default_runs_dir()
    return [name for name in VARIANTS if all((root / run / "metrics.json").exists() for run in runs_of(name))]


def report() -> None:
    config, names = load_config(MAIN), trained()
    table = pd.DataFrame(validation_rows(names))
    table.to_csv(REPO_ROOT / TABLE, index=False)
    pct = lambda x: f"{100 * x:.1f}"   # noqa: E731
    lines = ["# Stage 8 ablations on the session-blocked split", "",
             "Built by `src/session_ablations.py` from `runs/`; do not edit by hand. Every variant changes one thing and "
             "uses the settings selected for the main configuration (no tuning for any variant, handbook §24 2026-11-20). "
             "Five rounds, one seed.", "",
             "**Validation scores favour the main configuration:** its settings were chosen on these same validation "
             "photos, the variants' were not. Read validation differences as a bound, not as a test.", "",
             "## 1. Validation macro-F1 (all trained variants)", "",
             "| variant | what changes | validation macro-F1 | difference vs main (points) | rounds better than main |", "|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.variant} | {r.question} | {pct(r.val_macro_f1_mean)} ± {pct(r.val_macro_f1_sd)} | "
                     f"{100 * r.difference_vs_main:+.1f} | {r.rounds_better_than_main}/5 |")
    headline = [n for n in names if VARIANTS[n].get("tested")]
    lines += ["", "## 2. Headline ablations on test (main against variant; positive = the main configuration is better)", ""]
    if headline and (get_default_runs_dir() / PLAN).exists():
        fr.require_tested(PLAN)
        blocks = fr.block_of()
        main = fr.read_predictions(main_runs())
        others = {name: fr.read_predictions(runs_of(name)) for name in headline}
        paired = fr.paired_rows(main, others, blocks, config)
        lines += ["| variant | metric | difference (points) | photo range | block range | rounds won by main | verdict (photo / block) |",
                  "|---|---|---|---|---|---|---|"]
        for r in paired.itertuples():
            lines.append(f"| {r.B} | {r.metric} | {100 * r.difference:.1f} | [{100 * r.photo_low:.1f}, {100 * r.photo_high:.1f}] | "
                         f"[{100 * r.block_low:.1f}, {100 * r.block_high:.1f}] | {r.round_wins}/{r.rounds} | "
                         f"{r.verdict_photo} / {r.verdict_block} |")
    else:
        lines.append("The headline variants are not tested yet.")
    (REPO_ROOT / REPORT).write_text("\n".join(lines) + "\n")
    print(f"report: {len(table)} rows")


ACTIONS = {"fit": fit, "evaluate": evaluate_headline, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=ACTIONS)
    args = parser.parse_args()
    require_intact()
    ACTIONS[args.action]()
    print(f"SESSION ABLATIONS {args.action} DONE", flush=True)


if __name__ == "__main__":
    main()
