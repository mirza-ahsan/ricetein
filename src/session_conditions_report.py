"""Tables and the report of Stage 5 other conditions (declared 2026-11-20; handbook §24).

    uv run python -m src.session_conditions_report

Reads saved predictions only: it never fits or evaluates anything, so rerunning it is safe. It refuses to run until
every run in ``runs/session_conditions_plan.json`` has its test predictions. Each condition (128 px, 224 px + Otsu,
448 px) is set against the same model at 224 px (the run of ``runs/session_selection.json``), in both directions,
with the declared photo range and the block range.
"""

from __future__ import annotations

import json

import pandas as pd

from src import followup_report as fr
from src.config import REPO_ROOT, load_config
from src.followup import MODELS, read_json
from src.inventory_report import md_table
from src.records import get_default_runs_dir
from src.session_conditions import PLAN, SELECTION

TABLES = REPO_ROOT / "results/tables"
REPORT = REPO_ROOT / "docs/reports/session_conditions_report.md"
LABELS = {"224": "224 px", "128": "128 px", "224_otsu": "224 px + Otsu", "448": "448 px"}
AHEAD, BEHIND = "condition beats 224 px", "224 px beats condition"


def split_key(key: str) -> tuple[str, str]:
    """'convnext_tiny_224_otsu' -> ('convnext_tiny', '224_otsu'); the condition names are the keys of LABELS."""
    for condition in LABELS:
        if key.endswith(f"_{condition}"):
            return key[: -len(condition) - 1], condition
    raise ValueError(f"unknown condition in {key}")


def paired_both_ways(model: str, condition: str, tables: dict[str, pd.DataFrame], blocks: pd.Series,
                     config: dict) -> pd.DataFrame:
    """The condition against the same model at 224 px, and 224 px against the condition."""
    here, base = tables[f"{model}_{condition}"], tables[f"{model}_224"]
    rows = []
    for direction, a, b in ((AHEAD, here, base), (BEHIND, base, here)):
        frame = fr.paired_rows(a, {"other": b}, blocks, config).drop(columns="B")
        rows.append(frame.assign(model=model, condition=condition, direction=direction))
    return pd.concat(rows, ignore_index=True)


def build_report(tables: dict[str, pd.DataFrame], best_448: str) -> str:
    scores, paired = tables["scores"], tables["paired"]
    shown_scores = fr.final_text(scores.assign(method=scores["model"] + ", " + scores["condition"].map(LABELS)))
    shown = paired.assign(B=paired["model"] + ", " + paired["condition"].map(LABELS) + ": " + paired["direction"])

    def section(direction: str, first_column: str) -> str:
        part = shown[shown["direction"] == direction]
        table = fr.paired_text(part).rename(columns={"A (attention) vs": first_column})
        return f"{md_table(table)}\n\n{fr.changed_verdicts(part)}"

    return f"""# Stage 5 other conditions on the session-blocked split

Declared on 2026-11-20 (handbook §24) before any of these runs. Each of the four whole-image models is trained at
128 px and at 224 px with Otsu background removal; the model with the best validation score at 224 px
({best_448}) is also trained at 448 px. Every model keeps the learning rate selected at 224 px on validation
photos, and the recipe and budget are those of the 224 px runs. Nothing was searched again. Every condition was
tested once, on the test photos of the 5 rounds. The 224 px rows are the runs of `docs/reports/session_report.md`.

Scores are test macro-F1 and balanced accuracy: mean ± SD over the 5 rounds, then the declared 95% photo range
and the 95% block range of the pooled test predictions (a block is a run of photos from one camera with no pause over
60 s; blocks are resampled together). In each comparison A is the run named first in the row: "A beats B" needs the
range above 0 **and** at least 4 of 5 round wins, and a row that fails either is "no clear difference". Differences are
in percentage points (A minus B). There are 9 conditions, 2 scores and 2 directions, so read a single row with that
number of comparisons in mind.

## Scores of every model and condition

{md_table(shown_scores)}

## Does the condition beat the same model at 224 px?

{section(AHEAD, "condition against 224 px of the same model")}

## Does 224 px beat the condition?

The same test with the two sides swapped (the ranges are mirrored; only the round wins can differ, through ties).

{section(BEHIND, "224 px against the condition")}

Tables: `results/tables/session_conditions_scores.csv` and `results/tables/session_conditions_paired.csv`.
"""


def main() -> None:
    fr.require_tested(PLAN)
    config, root = load_config("configs/pipeline.yaml"), get_default_runs_dir()
    selection, blocks = read_json(root / SELECTION), fr.block_of()
    configurations = selection["configurations"]
    if {run for names in configurations.values() for run in names} != set(read_json(root / PLAN)["runs"]):
        raise RuntimeError(f"runs/{SELECTION} and runs/{PLAN} name different runs")
    baseline = read_json(root / "session_selection.json")["methods"]
    models = [m for m in MODELS if any(split_key(key)[0] == m for key in configurations)]
    runs = {f"{m}_224": baseline[f"{m}_224"]["runs"] for m in models} | configurations
    predictions = {key: fr.read_predictions(names) for key, names in runs.items()}
    order = [f"{m}_{c}" for m in models for c in LABELS if f"{m}_{c}" in predictions]
    scores = pd.DataFrame([{"model": split_key(key)[0], "condition": split_key(key)[1],
                            **fr.method_scores(predictions[key], blocks, config)} for key in order])
    paired = pd.concat([paired_both_ways(*split_key(key), predictions, blocks, config) for key in configurations],
                       ignore_index=True)
    tables = {"scores": scores, "paired": paired}
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(TABLES / f"session_conditions_{name}.csv", index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(tables, selection["best_448_model"]), encoding="utf-8")
    print(json.dumps({name: len(frame) for name, frame in tables.items()}))


if __name__ == "__main__":
    main()
