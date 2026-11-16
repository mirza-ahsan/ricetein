"""Tables and the report of the session-split tuning protocol (declared 2026-11-16; handbook §24).

    uv run python -m src.session_report

Reads saved predictions, run metrics and the follow-up tables only: it never fits or evaluates anything, so
rerunning it is safe. It refuses to run until every run in ``runs/session_evaluation_plan.json`` has its
test predictions.
"""

from __future__ import annotations

import json

import pandas as pd

from src import followup_report as fr
from src.config import REPO_ROOT, load_config
from src.followup import AGGREGATORS, MAX_EXTENSIONS, read_json
from src.inventory_report import md_table
from src.records import get_default_runs_dir
from src.session_study import PLAN

TABLES = REPO_ROOT / "results/tables"
REPORT = REPO_ROOT / "docs/reports/session_report.md"


def side_by_side(final: pd.DataFrame, locked: pd.DataFrame, refit: pd.DataFrame) -> pd.DataFrame:
    """Test macro-F1 (mean and SD over rounds) of each method under three set-ups: the locked split, the
    session-blocked split with the locked-split settings (nine methods), and the session-blocked split
    tuned on its own validation photos (``final``)."""
    locked, refit = locked.set_index("method"), refit.set_index("method")
    rows = []
    for row in final.to_dict("records"):
        method = row["method"]
        rows.append({"method": method,
                     "locked_mean": locked.loc[method, "macro_f1_mean"], "locked_sd": locked.loc[method, "macro_f1_sd"],
                     "refit_mean": refit["session_macro_f1_mean"].get(method),
                     "refit_sd": refit["session_macro_f1_sd"].get(method),
                     "tuned_mean": row["macro_f1_mean"], "tuned_sd": row["macro_f1_sd"]})
    return pd.DataFrame(rows)


def build_report(tables: dict[str, pd.DataFrame]) -> str:
    sides, shortcuts = tables["side_by_side"], tables["shortcuts"]
    shown_search = fr.search_text(tables["search"], "selected runs differ from those tested on 2026-11-16")
    shown_final = fr.final_text(tables["final"])

    def cell(mean: float, sd: float) -> str:
        return "—" if pd.isna(mean) else fr.plus_minus(mean, sd)

    shown_sides = pd.DataFrame({
        "method": sides["method"],
        "locked split": [cell(m, s) for m, s in zip(sides["locked_mean"], sides["locked_sd"])],
        "session split, locked-split settings": [cell(m, s) for m, s in zip(sides["refit_mean"], sides["refit_sd"])],
        "session split, tuned on it": [cell(m, s) for m, s in zip(sides["tuned_mean"], sides["tuned_sd"])]})
    shown_shortcuts = pd.DataFrame({"baseline": shortcuts["baseline"]} | {
        label: [fr.plus_minus(m, s)
                for m, s in zip(shortcuts[f"session_{metric}_mean"], shortcuts[f"session_{metric}_sd"])]
        for metric, label in (("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced acc."))})
    return f"""# Session-blocked split: every method tuned on its own validation photos

Built by `src/session_report.py` from `runs/`, `splits/session_blocked/folds.csv` and the follow-up tables;
do not edit by hand. Protocol: handbook §24, 2026-11-16. All scores are percentages.

**Which photos each number comes from**

- Sections 1 and 6 are **validation** scores of the session-blocked split (used for selection only).
- Sections 2, 3 and 5 are **test** scores of the session-blocked split (`splits/session_blocked/folds.csv`).
- Section 4 shows test scores of both splits side by side.
- "mean ± SD" is over the five rounds. Both ranges are around the pooled score (every test photo once), so a
  mean can sit slightly outside its range when the rounds differ a lot.
- "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks (photos
  shot without a pause of more than 60 s from one camera) and is wider when neighbouring photos are alike.

**Limits to read first**

- The session-blocked split was declared on 2026-11-14, **after** the test results of the locked split were seen.
- Before this protocol was declared (2026-11-16), the session-split test photos had been scored once, with the
  settings chosen on the locked split (`docs/reports/followup_report.md`, section 2). Every selection here
  uses session-split validation scores only, and each selected configuration was tested once.
- Class shares on the session split are reported, not enforced (`docs/reports/session_split_report.md`).
- Tuning budgets differ between method families: each of the five head methods keeps the better of two
  40-trial studies, a CNN has three learning rates plus up to {MAX_EXTENSIONS} added ones, and a probe has six
  L2 strengths plus up to {MAX_EXTENSIONS} added ones. Within a family the budget is the same.
  The head studies are widened only towards higher learning rates; a CNN or probe grid can grow at either end.
- A photo that is a validation photo in one round is a test photo in another round (41.9% of the photos). Settings
  chosen on the five-round validation mean are therefore not independent of the pooled test photos. This holds
  for every method, and a larger search can gain more from it.
- Test scores use one training seed (seed 0). The stability seeds are validation only and cover mean, max and
  attention, so the "4 of 5 rounds" rule does not include seed-to-seed variation.
- Section 1 column "selected runs differ from those tested on 2026-11-16": "no" means the search chose the
  locked-split setting again, so that method's test score was already known before this protocol was declared.
  It is always "yes" for the five head methods, whose runs are new.
- The photo range is stratified by class; the block range resamples whole blocks and is not, so it also
  contains class-count variation. That is part of why it is wider.

## 1. Searches (validation macro-F1, mean over 5 rounds)

{md_table(shown_search)}

## 2. Results on the session-blocked split (test)

{md_table(shown_final)}

## 3. Attention against every other method (session-blocked split, test)

{md_table(fr.paired_text(tables["paired"]))}

{fr.changed_verdicts(tables["paired"])}

## 4. Three set-ups side by side (test macro-F1, mean ± SD over 5 rounds)

The locked split column is the final locked-split configuration of the follow-up study. The middle column
refits the nine core settings chosen on the locked split; the whole-photo controls were not part of it.

{md_table(shown_sides)}

## 5. Shortcut controls on the session-blocked split (test, mean ± SD over 5 rounds)

{md_table(shown_shortcuts)}

## 6. Stability seeds of the selected tile studies (validation macro-F1)

Seed 0 is the declared run; seeds 1 and 2 are the declared stability seeds. Validation only.

{md_table(fr.stability_text(tables["stability"]))}
"""


def main() -> None:
    fr.require_tested(PLAN)
    config, root = load_config("configs/pipeline.yaml"), get_default_runs_dir()
    selection, blocks = read_json(root / "session_selection.json"), fr.block_of()
    methods = selection["methods"]
    if {run for entry in methods.values() for run in entry["runs"]} != set(read_json(root / PLAN)["runs"]):
        raise RuntimeError(f"runs/session_selection.json and runs/{PLAN} name different runs")
    # cheap and most likely to fail: do these first
    stability = fr.stability_table({name: methods[name]["study"] for name in AGGREGATORS})
    shortcuts = pd.read_csv(TABLES / "followup_shortcuts.csv")
    locked, refit = pd.read_csv(TABLES / "followup_final.csv"), pd.read_csv(TABLES / "followup_split_sensitivity.csv")
    predictions = {method: fr.read_predictions(entry["runs"]) for method, entry in methods.items()}
    final = pd.DataFrame([{"method": method, **fr.method_scores(table, blocks, config)}
                          for method, table in predictions.items()])
    paired = fr.paired_rows(predictions["attention"], {m: t for m, t in predictions.items() if m != "attention"},
                            blocks, config)
    tables = {"search": fr.search_table(selection), "final": final, "paired": paired,
              "side_by_side": side_by_side(final, locked, refit), "shortcuts": shortcuts, "stability": stability}
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if name != "shortcuts":   # unchanged copy of followup_shortcuts.csv
            frame.to_csv(TABLES / f"session_{name}.csv", index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(tables), encoding="utf-8")
    print(json.dumps({name: len(frame) for name, frame in tables.items()}))


if __name__ == "__main__":
    main()
