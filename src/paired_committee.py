"""The Stage 11 committee ``row2_plain`` against ConvNeXt-T at 448 px alone (declared 2026-11-21; handbook §24).

    uv run python -m src.paired_committee report

Reads saved test predictions only: it never fits or evaluates anything, so rerunning it is safe. It refuses to run
until the committee and the five ConvNeXt-T runs have their ``test_started.json``, and until both plan files exist.
Round r of the committee is compared with ConvNeXt-T round r on the same test photos, with the declared photo range
and the block range, and the Stage 11 verdict rule (``src.committee.two_way``) in both directions. The comparison is
post-hoc (declared after both test means were seen) and decides no selection. ConvNeXt-T at 448 px is a member of the
committee, so the two are not independent.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import committee as cm
from src import followup_report as fr
from src.config import REPO_ROOT, load_config
from src.followup import read_json
from src.inventory_report import md_table
from src.records import get_default_runs_dir

COMMITTEE, SINGLE = "row2_plain", "convnext_tiny_448"
COMMITTEE_PLAN, CONDITIONS_PLAN = "committee_plan.json", "session_conditions_plan.json"
TABLE = REPO_ROOT / "results/tables/paired_committee.csv"
REPORT = REPO_ROOT / "docs/reports/paired_committee_report.md"
NAMES = {COMMITTEE: "Committee (ConvNeXt-T 448 px + attention)", SINGLE: "ConvNeXt-T 448 px alone"}


def single_runs() -> list[str]:
    """The five ConvNeXt-T 448 px runs: the committee member runs, which must be the tested condition runs."""
    root = get_default_runs_dir()
    member = read_json(root / "committee_selection.json")["member_runs"][SINGLE]
    condition = read_json(root / "session_conditions_selection.json")["configurations"][SINGLE]
    if member != condition:
        raise RuntimeError("the committee's ConvNeXt-T 448 px runs differ from runs/session_conditions_selection.json")
    return member


def require_started(runs: list[str], planned: list[str], plan_name: str) -> None:
    """Every run is named by the plan file and has its test_started.json (it was tested once under that plan)."""
    root = get_default_runs_dir()
    for run in runs:
        if run not in planned:
            raise RuntimeError(f"{run} is not in runs/{plan_name}")
        if not (root / run / "test_started.json").exists():
            raise RuntimeError(f"{run} has no test_started.json: it was not tested under runs/{plan_name}")


def require_same_photos(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Every test round holds the same photos in A and B."""
    test_a, test_b = (t[t["partition"] == "test"] for t in (a, b))
    rounds = sorted(test_a["round"].unique())
    if rounds != sorted(test_b["round"].unique()):
        raise RuntimeError("the committee and ConvNeXt-T have different test rounds")
    for r in rounds:
        photos = [set(t[t["round"] == r]["photo_id"]) for t in (test_a, test_b)]
        if photos[0] != photos[1] or len(test_a[test_a["round"] == r]) != len(test_b[test_b["round"] == r]):
            raise RuntimeError(f"round {r}: the test photos of the committee and ConvNeXt-T differ")


def build_report(paired: pd.DataFrame, scores: pd.DataFrame) -> str:
    shown = pd.DataFrame({
        "metric": paired["metric"],
        "difference, committee minus ConvNeXt-T (points)": paired["difference"].map(fr.percent),
        "photo range": [fr.interval_text(a, b) for a, b in zip(paired["photo_low"], paired["photo_high"])],
        "block range": [fr.interval_text(a, b) for a, b in zip(paired["block_low"], paired["block_high"])],
        "rounds won (committee / ConvNeXt-T)": [f"{w}/{lost} of {n}" for w, lost, n in zip(
            paired["round_wins"], paired["round_losses"], paired["rounds"])],
        "verdict (photo range)": [cm.readable(v, "committee", "ConvNeXt-T") for v in paired["verdict_photo"]],
        "verdict (block range)": [cm.readable(v, "committee", "ConvNeXt-T") for v in paired["verdict_block"]]})
    return f"""# Committee against ConvNeXt-T at 448 px alone (session-blocked split, test)

Built by `src/paired_committee.py` from `runs/`; do not edit by hand. Declared on 2026-11-21 (handbook §24). All
scores are percentages, and differences are in percentage points.

**Which photos:** test photos of the session-blocked split (`splits/session_blocked/folds.csv`). Nothing was trained
or tested again: the predictions are those of the committee `row2_plain` (tested once, `runs/{COMMITTEE_PLAN}`) and of
the five ConvNeXt-T 448 px runs (tested once, `runs/{CONDITIONS_PLAN}`). Every test round holds the same photos in both.

**Limits to read first**

- The comparison was declared **after** both test means were seen (committee 81.6, ConvNeXt-T 448 px 79.2), so it is
  post-hoc. It decides no selection. Stage 11 tested the committee against attention only.
- ConvNeXt-T at 448 px is **one of the two members** of the committee, so the two are not independent. The comparison
  asks whether adding the tile model to the best whole-image model helps.
- There are 2 metrics and 2 ranges, so read a single row with that number of comparisons in mind.
- Test scores use one training seed. The two ranges are around the pooled score (every test photo once).
  "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks and is wider when
  neighbouring photos are alike.

**Rule** (the decision rule with the Stage 11 refinement, stated under both ranges). "Committee beats ConvNeXt-T" needs the
95% range of the difference above 0 **and** the committee winning at least 4 of 5 rounds. "ConvNeXt-T beats committee"
needs the range below 0 and ConvNeXt-T winning at least 4 of 5 rounds (a tied round is won by neither). Otherwise: no
clear difference.

## 1. Test scores (mean ± SD over the 5 rounds, then the photo range and the block range of the pooled test predictions)

{md_table(fr.final_text(scores))}

## 2. Committee against ConvNeXt-T at 448 px

{md_table(shown)}

{verdict_summary(paired)}

Table: `results/tables/paired_committee.csv`.
"""


def verdict_summary(paired: pd.DataFrame) -> str:
    lines = [f"- {r.metric}, {kind} range: {cm.readable(getattr(r, f'verdict_{kind}'), 'committee', 'ConvNeXt-T')}."
             for r in paired.itertuples() for kind in ("photo", "block")]
    return "Verdicts:\n\n" + "\n".join(lines)


def report() -> None:
    root = get_default_runs_dir()
    runs = single_runs()
    plans = {name: read_json(root / name) for name in (COMMITTEE_PLAN, CONDITIONS_PLAN)}   # missing plan: an error
    require_started([f"committee_{COMMITTEE}"], [f"committee_{c}" for c in plans[COMMITTEE_PLAN]["committees"]],
                    COMMITTEE_PLAN)
    require_started(runs, list(plans[CONDITIONS_PLAN]["runs"]), CONDITIONS_PLAN)
    config, blocks = load_config("configs/pipeline.yaml"), fr.block_of()
    committee = cm.committee_table(COMMITTEE)
    single = fr.read_predictions(runs)
    require_same_photos(committee, single)
    paired = cm.paired_with_two_way(committee, single, COMMITTEE, SINGLE, blocks, config)
    scores = pd.DataFrame([{"method": NAMES[name], **fr.method_scores(table, blocks, config)}
                           for name, table in ((COMMITTEE, committee), (SINGLE, single))])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(TABLE, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(paired, scores), encoding="utf-8")
    print(json.dumps({"rows": len(paired), "runs": len(runs)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["report"])
    parser.parse_args()
    report()


if __name__ == "__main__":
    main()
