"""Attention against each of the nine Stage 5 conditions on the session-blocked split (declared 2026-11-21; handbook §24).

    uv run python -m src.paired_conditions report

Reads saved test predictions only: it never fits or evaluates anything, so rerunning it is safe. It refuses to run
until every run of ``runs/session_evaluation_plan.json`` (attention) and ``runs/session_conditions_plan.json`` (the
conditions) has its ``test_started.json`` and its test predictions. Each condition is compared with attention in both
directions, with the declared photo range and the block range. The comparison is post-hoc (declared after both methods'
test means were seen) and decides no selection.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import followup_report as fr
from src.config import REPO_ROOT, load_config
from src.followup import read_json
from src.inventory_report import md_table
from src.records import get_default_runs_dir
from src.session_conditions import PLAN as CONDITIONS_PLAN, SELECTION
from src.session_conditions_report import LABELS, split_key
from src.session_study import PLAN as ATTENTION_PLAN

TABLE = REPO_ROOT / "results/tables/paired_conditions.csv"
REPORT = REPO_ROOT / "docs/reports/paired_conditions_report.md"
MODEL_NAMES = {"inceptionv3": "InceptionV3", "efficientnetv2s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-T",
               "swin_tiny": "Swin-T"}
ATTENTION, NONE = "attention beats it", "no clear difference"
OTHER = "it beats attention"


def condition_name(key: str) -> str:
    model, condition = split_key(key)
    return f"{MODEL_NAMES[model]}, {LABELS[condition]}"


def ordered(keys: list[str]) -> list[str]:
    """The ConvNeXt-T 448 px condition first (the one with the higher mean), then the others in file order."""
    return sorted(keys, key=lambda key: split_key(key)[1] != "448")


def require_started(plan_name: str) -> None:
    """Every run of the plan was tested once: its test_started.json and test predictions exist."""
    root = get_default_runs_dir()
    for run in read_json(root / plan_name)["runs"]:
        if not (root / run / "test_started.json").exists():
            raise RuntimeError(f"{run} has no test_started.json: it was not tested under runs/{plan_name}")


def overall(ahead: str, behind: str) -> str:
    """One verdict from the two directions of the same test (A beats B, and B beats A)."""
    if ahead == "A beats B":
        return ATTENTION
    return OTHER if behind == "A beats B" else NONE


def paired_conditions(attention: pd.DataFrame, conditions: dict[str, pd.DataFrame], blocks: pd.Series,
                      config: dict) -> pd.DataFrame:
    """One row per condition and metric: attention minus the condition, both ranges, round wins on each side, verdicts."""
    rows = []
    for key, table in conditions.items():
        ahead = fr.paired_rows(attention, {key: table}, blocks, config).set_index("metric")
        behind = fr.paired_rows(table, {"attention": attention}, blocks, config).set_index("metric")
        for metric in ahead.index:
            a, b = ahead.loc[metric], behind.loc[metric]
            rows.append({"condition": key, "metric": metric, "difference": a["difference"],
                         "photo_low": a["photo_low"], "photo_high": a["photo_high"],
                         "block_low": a["block_low"], "block_high": a["block_high"],
                         "attention_round_wins": a["round_wins"], "condition_round_wins": b["round_wins"],
                         "rounds": a["rounds"],
                         "verdict_photo": overall(a["verdict_photo"], b["verdict_photo"]),
                         "verdict_block": overall(a["verdict_block"], b["verdict_block"])})
    return pd.DataFrame(rows)


def build_report(paired: pd.DataFrame, scores: pd.DataFrame) -> str:
    shown = pd.DataFrame({
        "condition": paired["condition"].map(condition_name), "metric": paired["metric"],
        "difference, attention minus condition (points)": paired["difference"].map(fr.percent),
        "photo range": [fr.interval_text(a, b) for a, b in zip(paired["photo_low"], paired["photo_high"])],
        "block range": [fr.interval_text(a, b) for a, b in zip(paired["block_low"], paired["block_high"])],
        "rounds won (attention / condition)": [f"{a}/{c} of {n}" for a, c, n in zip(
            paired["attention_round_wins"], paired["condition_round_wins"], paired["rounds"])],
        "verdict (photo range)": paired["verdict_photo"], "verdict (block range)": paired["verdict_block"]})
    return f"""# Attention against the nine Stage 5 conditions (session-blocked split, test)

Built by `src/paired_conditions.py` from `runs/`; do not edit by hand. Declared on 2026-11-21 (handbook §24). All
scores are percentages, and differences are in percentage points.

**Which photos:** test photos of the session-blocked split (`splits/session_blocked/folds.csv`). Nothing was trained
or tested again: the predictions are those of the attention runs (tested 2026-11-17, `runs/session_evaluation_plan.json`)
and of the nine conditions (tested 2026-11-21, `runs/session_conditions_plan.json`).

**Limits to read first**

- The comparison was declared **after** both methods' test means were seen (ConvNeXt-T at 448 px has the higher mean),
  so it is post-hoc. It decides no selection.
- There are 9 conditions, 2 metrics and 2 ranges, so read a single row with that number of comparisons in mind.
- Test scores use one training seed. The two ranges are around the pooled score (every test photo once).
  "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks and is wider when
  neighbouring photos are alike.
- The conditions use the learning rate selected at 224 px; they were not tuned at their own size. Attention was tuned
  with its own search (`docs/reports/session_report.md`).

**Rule** (the decision rule, stated under both ranges). "Attention beats it" needs the 95% range of attention minus the
condition above 0 **and** attention winning at least 4 of 5 rounds. "It beats attention" is the same test with the sides
swapped. Otherwise: no clear difference. A claim that attention (or a condition) is best is made only where this rule
supports it.

## 1. Test scores (mean ± SD over the 5 rounds, then the photo range and the block range of the pooled test predictions)

{md_table(fr.final_text(scores))}

## 2. Attention against each condition

{md_table(shown)}

{verdict_summary(paired)}

Table: `results/tables/paired_conditions.csv`.
"""


def verdict_summary(paired: pd.DataFrame) -> str:
    lines = []
    for metric in fr.METRICS:
        part = paired[paired["metric"] == metric]
        for label, column in (("photo range", "verdict_photo"), ("block range", "verdict_block")):
            counts = ", ".join(f"{(part[column] == v).sum()} {v}" for v in (ATTENTION, NONE, OTHER))
            lines.append(f"- {metric}, {label}: {counts}.")
    return "Verdicts over the nine conditions:\n\n" + "\n".join(lines)


def report() -> None:
    for plan in (ATTENTION_PLAN, CONDITIONS_PLAN):
        fr.require_tested(plan)
        require_started(plan)
    config, root = load_config("configs/pipeline.yaml"), get_default_runs_dir()
    configurations = read_json(root / SELECTION)["configurations"]
    attention_runs = read_json(root / "session_selection.json")["methods"]["attention"]["runs"]
    if {r for names in configurations.values() for r in names} != set(read_json(root / CONDITIONS_PLAN)["runs"]):
        raise RuntimeError(f"runs/{SELECTION} and runs/{CONDITIONS_PLAN} name different runs")
    if not set(attention_runs) <= set(read_json(root / ATTENTION_PLAN)["runs"]):
        raise RuntimeError(f"the attention runs of runs/session_selection.json are not all in runs/{ATTENTION_PLAN}")
    blocks, keys = fr.block_of(), ordered(list(configurations))
    attention = fr.read_predictions(attention_runs)
    conditions = {key: fr.read_predictions(configurations[key]) for key in keys}
    paired = paired_conditions(attention, conditions, blocks, config)
    named = {"attention": attention} | {condition_name(key): table for key, table in conditions.items()}
    scores = pd.DataFrame([{"method": name, **fr.method_scores(table, blocks, config)} for name, table in named.items()])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(TABLE, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(paired, scores), encoding="utf-8")
    print(json.dumps({"rows": len(paired), "conditions": len(keys)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["report"])
    parser.parse_args()
    report()


if __name__ == "__main__":
    main()
