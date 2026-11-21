"""Stage 12 step 4: import the three human heat-map ratings and report them next to the model ratings.

    python -m src.heatmap_ratings import    # reads cache/rating_kit/returned/ratings_{ahsan,umer,harris}.csv
    python -m src.heatmap_ratings report    # needs all three imported; reads examples_val.csv for the categories

``import`` only checks the returned files and appends them to ``data/heatmap_ratings.csv`` (model rows are never
rewritten). The names stay in ``cache/rating_kit/rater_map.csv``, which is git-ignored.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.explain import ANSWERS, RATINGS, TABLES, fleiss_kappa, rating_summary
from src.inventory_report import md_table

KIT = REPO_ROOT / "cache/rating_kit"
EXAMPLES = REPO_ROOT / "results/explain/examples_val.csv"
REPORT = REPO_ROOT / "docs/reports/heatmap_ratings_report.md"
BY_KIND = TABLES / "explain_ratings_by_kind.csv"
ITEMS = [f"H{i:02d}" for i in range(1, 51)]
RATERS = {"ahsan": "human_rater_1", "umer": "human_rater_2", "harris": "human_rater_3"}   # file name -> rater id
COLUMNS = ["item_id", "rater", "rater_kind", "rating", "note"]


# ------------------------------------------------------------------------------------------------ import
def read_returned(path) -> pd.DataFrame:
    """One returned file, checked: 50 rows H01..H50 once each, every answer yes / partly / no (case and spaces
    normalised). Anything else stops with the file and the row (as a spreadsheet shows it: header = row 1)."""
    if not path.exists():
        raise ValueError(f"{path.name}: file not found")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", skipinitialspace=True)
    frame.columns = [c.strip().lower() for c in frame.columns]
    if list(frame.columns) != ["item_id", "rating", "note"]:
        raise ValueError(f"{path.name}: the header must be item_id,rating,note, found {','.join(frame.columns)}")
    frame["item_id"] = frame["item_id"].str.strip().str.upper()
    frame["rating"] = frame["rating"].str.strip().str.lower()
    for row, (item, answer) in enumerate(zip(frame["item_id"], frame["rating"]), start=2):
        if answer not in ANSWERS:
            raise ValueError(f"{path.name}, row {row} ({item}): rating {answer!r} is not one of {', '.join(ANSWERS)}")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"{path.name}: item {frame.loc[frame['item_id'].duplicated(), 'item_id'].iloc[0]} is rated twice")
    if sorted(frame["item_id"]) != ITEMS:
        missing = sorted(set(ITEMS) - set(frame["item_id"]))
        extra = sorted(set(frame["item_id"]) - set(ITEMS))
        raise ValueError(f"{path.name}: needs exactly H01 to H50 once each (missing {missing}, unknown {extra})")
    return frame.sort_values("item_id").reset_index(drop=True)


def import_ratings(returned, ratings_path=None, kit=None) -> int:
    """Validate all three files, then append them in one go. Nothing is written when any check fails."""
    ratings_path, kit = ratings_path or RATINGS, kit or KIT
    existing = pd.read_csv(ratings_path, dtype=str) if ratings_path.exists() else pd.DataFrame(columns=COLUMNS)
    done = set(existing["rater"]) & set(RATERS.values())
    if done:
        raise ValueError(f"already imported: {', '.join(sorted(done))}; nothing was changed")
    files = {name: read_returned(returned / f"ratings_{name}.csv") for name in RATERS}
    rows = [
        {"item_id": r.item_id, "rater": RATERS[name], "rater_kind": "human", "rating": r.rating, "note": r.note.strip()}
        for name, frame in files.items()
        for r in frame.itertuples()
    ]
    text = ratings_path.read_text()
    with ratings_path.open("a", newline="") as out:
        if text and not text.endswith("\n"):
            out.write("\n")
        csv.DictWriter(out, COLUMNS, lineterminator="\n").writerows(rows)
    kit.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": [f"ratings_{n}.csv" for n in RATERS], "rater": list(RATERS.values())}).to_csv(
        kit / "rater_map.csv", index=False
    )
    return len(rows)


# ------------------------------------------------------------------------------------------------ report
def majority(wide: pd.DataFrame) -> pd.Series:
    """The answer given most often per item; NaN when two or more answers tie for the top."""
    counts = pd.DataFrame({a: (wide == a).sum(axis=1) for a in ANSWERS})
    top = counts.max(axis=1)
    return counts.idxmax(axis=1).where((counts == top.to_numpy()[:, None]).sum(axis=1) == 1)


def wide_of(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot(index="item_id", columns="rater", values="rating")


def human_against_model(human: pd.DataFrame, model: pd.DataFrame) -> dict:
    h, m = majority(wide_of(human)), majority(wide_of(model)).reindex(wide_of(human).index)
    both = h.notna() & m.notna()
    return {
        "maps": len(h),
        "equal": int((h[both] == m[both]).sum()),
        "with_majority": int(both.sum()),
        "human_ties": int(h.isna().sum()),
        "model_ties": int(m.isna().sum()),
        "all_equal": int(((h == m) & both).sum()),
    }


def by_kind_table(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for kind, s in summaries.items():
        shares = s["shares"].loc["all ratings"]
        rows.append({"kind": kind, "raters": len(s["raters"]), "maps": s["items"],
                     **{a: float(shares[a]) for a in ANSWERS},
                     "all_agree": s["all_agree"], "kappa": s["kappa"]})
    return pd.DataFrame(rows)


def percent(table: pd.DataFrame) -> pd.DataFrame:
    return (table * 100).round(1).reset_index()


def report(ratings_path=None, examples_path=None) -> None:
    ratings = pd.read_csv(ratings_path or RATINGS, dtype=str, keep_default_na=False)
    examples = pd.read_csv(examples_path or EXAMPLES)
    kinds = {k: ratings[ratings["rater_kind"] == k] for k in ("human", "model")}
    if kinds["human"]["rater"].nunique() != 3 or kinds["model"]["rater"].nunique() < 2:
        raise ValueError("the report needs the three human raters and the model raters in the ratings file")
    summaries = {k: rating_summary(frame, examples) for k, frame in kinds.items()}
    table = by_kind_table(summaries)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(BY_KIND, index=False)
    link = human_against_model(kinds["human"], kinds["model"])
    text = [
        "# Stage 12: human ratings of the 50 blind heat maps",
        "",
        "Generated by `python -m src.heatmap_ratings report` from `data/heatmap_ratings.csv`; nothing is typed by "
        "hand. The three human raters (`human_rater_1..3`) are the team members; the model raters "
        "(`model_rater_1..3`) are disclosed model judgements (handbook §24, 2026-11-08). Question: does the hot area "
        "cover visible symptoms? Each rater saw the same 50 blind sheets, without the true class or the prediction. "
        "Percentages of ratings. The table is also `results/tables/explain_ratings_by_kind.csv`.",
        "",
    ]
    for kind in ("human", "model"):
        s = summaries[kind]
        text += [
            f"## Shares of answers: {kind} raters",
            "",
            md_table(percent(s["shares"])),
            "",
            f"All {len(s['raters'])} {kind} raters gave the same answer on {s['all_agree']:.1%} of the maps; "
            f"Fleiss' kappa {s['kappa']:.2f} (0 = chance level, 1 = perfect agreement).",
            "",
        ]
    text += [
        "## Human against model",
        "",
        "The majority answer of a map is the answer at least two of the three raters gave; when all three differ "
        "there is no majority (a tie).",
        "",
        f"- The human majority equals the model majority on {link['equal']} of {link['with_majority']} maps that have "
        f"a majority on both sides ({link['equal'] / link['with_majority']:.1%})." if link["with_majority"] else
        "- No map has a majority on both sides.",
        f"- Ties: {link['human_ties']} maps have no human majority and {link['model_ties']} have no model majority "
        f"(of {link['maps']}).",
        f"- Counting a tie as a difference, they agree on {link['all_equal']} of {link['maps']} maps "
        f"({link['all_equal'] / link['maps']:.1%}).",
        "",
        "## Human raters by example category",
        "",
        "`correct` and `wrong` = the rule's confident right and wrong answers; `random` = random photos of each class; `extra` = further random photos. "
        "Seen only after rating. Percentages of the human ratings in each category.",
        "",
        md_table(percent(summaries["human"]["by_category"])),
        "",
        "Limits: three raters who are also the authors, one model family rated, 50 maps.",
        "",
    ]
    REPORT.write_text("\n".join(text))
    print(f"report -> {REPORT}\ntable  -> {BY_KIND}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["import", "report"])
    parser.add_argument("--returned", default=str(KIT / "returned"), help="folder with the three filled CSVs")
    args = parser.parse_args()
    if args.action == "import":
        print(f"appended {import_ratings(Path(args.returned))} human ratings to {RATINGS}")
    else:
        report()


if __name__ == "__main__":
    main()
