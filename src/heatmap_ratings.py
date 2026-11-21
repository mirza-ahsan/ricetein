"""Stage 12 step 4: import the three human heat-map ratings and report them next to the model ratings.

    python -m src.heatmap_ratings import    # reads cache/rating_kit/returned/ratings_*.csv (three files)
    python -m src.heatmap_ratings report    # needs all three imported; reads examples_val.csv for the categories

``import`` only checks the returned files and appends them to ``data/heatmap_ratings.csv`` (model rows are never
rewritten). Which file got which id is drawn at random and kept only in ``cache/rating_kit/rater_map.csv``
(git-ignored); no name appears in the code or in the tracked files.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.explain import ANSWERS, RATINGS, TABLES, rating_summary
from src.inventory_report import md_table

KIT = REPO_ROOT / "cache/rating_kit"
EXAMPLES = REPO_ROOT / "results/explain/examples_val.csv"
REPORT = REPO_ROOT / "docs/reports/heatmap_ratings_report.md"
PROVENANCE = REPO_ROOT / "data/heatmap_ratings_provenance.txt"   # who really filled the human-kind files; shown first in the report
BY_KIND = TABLES / "explain_ratings_by_kind.csv"
ITEMS = [f"H{i:02d}" for i in range(1, 51)]
HUMAN_IDS = ["human_rater_1", "human_rater_2", "human_rater_3"]
FILES = "ratings_*.csv"   # the three returned files; the rater ids are assigned to them at random
COLUMNS = ["item_id", "rater", "rater_kind", "rating", "note"]


# ------------------------------------------------------------------------------------------------ import
def read_text(path) -> str:
    """UTF-8 (with or without BOM) or, for Excel's plain "CSV" on Windows, cp1252. UTF-16 is refused with a clear error."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        raise ValueError(f"{path.name}: saved as UTF-16; save it again as CSV (comma delimited) or CSV UTF-8")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def read_returned(path) -> pd.DataFrame:
    """One returned file, checked: 50 rows H01..H50 once each, every answer yes / partly / no (case and spaces
    normalised). Anything else stops with the file and the row (as a spreadsheet shows it: header = row 1).
    Rows that are completely empty and empty extra columns (Excel adds them) are ignored."""
    if not path.exists():
        raise ValueError(f"{path.name}: file not found")
    try:
        frame = pd.read_csv(
            io.StringIO(read_text(path)), dtype=str, keep_default_na=False, skipinitialspace=True,
            skip_blank_lines=False, index_col=False,
        )
    except pd.errors.ParserError as error:
        raise ValueError(f"{path.name}: not a 3-column CSV (a comma inside a note needs quotes): {error}") from None
    frame.columns = [c.strip().lower() for c in frame.columns]
    frame = frame.loc[:, [not (c.startswith("unnamed") and (frame[c].str.strip() == "").all()) for c in frame.columns]]
    if list(frame.columns) != ["item_id", "rating", "note"]:
        raise ValueError(f"{path.name}: the header must be item_id,rating,note, found {','.join(frame.columns)}")
    frame = frame[(frame.apply(lambda c: c.str.strip()) != "").any(axis=1)]   # the index stays the spreadsheet row - 2
    frame["item_id"] = frame["item_id"].str.strip().str.upper()
    frame["rating"] = frame["rating"].str.strip().str.lower()
    frame["note"] = frame["note"].str.replace(r"\s+", " ", regex=True).str.strip()
    bad = frame[~frame["rating"].isin(ANSWERS)]
    if not bad.empty:
        first = bad.iloc[0]
        raise ValueError(
            f"{path.name}, row {bad.index[0] + 2} ({first['item_id']}): rating {first['rating']!r} is not one of {', '.join(ANSWERS)}"
        )
    if frame["item_id"].duplicated().any():
        raise ValueError(f"{path.name}: item {frame.loc[frame['item_id'].duplicated(), 'item_id'].iloc[0]} is rated twice")
    if sorted(frame["item_id"]) != ITEMS:
        missing = sorted(set(ITEMS) - set(frame["item_id"]))
        extra = sorted(set(frame["item_id"]) - set(ITEMS))
        raise ValueError(f"{path.name}: needs exactly H01 to H50 once each (missing {missing}, unknown {extra})")
    return frame.sort_values("item_id").reset_index(drop=True)


def import_ratings(returned, ratings_path=None, kit=None) -> int:
    """Validate all three files, then add them in one step. Nothing is written when any check fails, and the ratings
    file is replaced in one move, so it is never left half written."""
    ratings_path, kit = ratings_path or RATINGS, kit or KIT
    map_path = kit / "rater_map.csv"
    existing = ratings_path.read_text(encoding="utf-8") if ratings_path.exists() else ""
    if map_path.exists() or any(rater in existing for rater in HUMAN_IDS):
        raise ValueError("human ratings are already imported; nothing was changed")
    paths = sorted(returned.glob(FILES))
    if len(paths) != len(HUMAN_IDS):
        raise ValueError(f"expected {len(HUMAN_IDS)} files {FILES} in {returned}, found {len(paths)}")
    files = [read_returned(path) for path in paths]
    ids = random.SystemRandom().sample(HUMAN_IDS, len(HUMAN_IDS))
    rows = [
        {"item_id": r.item_id, "rater": rater, "rater_kind": "human", "rating": r.rating, "note": r.note}
        for rater, frame in zip(ids, files)
        for r in frame.itertuples()
    ]
    out = io.StringIO()
    csv.DictWriter(out, COLUMNS, lineterminator="\n").writerows(rows)
    new_text = existing + ("\n" if existing and not existing.endswith("\n") else "") + out.getvalue()
    kit.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": [p.name for p in paths], "rater": ids}).to_csv(map_path, index=False)
    temp = ratings_path.with_suffix(".tmp")
    try:
        temp.write_text(new_text, encoding="utf-8", newline="")
        os.replace(temp, ratings_path)
    except BaseException:
        map_path.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
        raise
    return len(rows)


# ------------------------------------------------------------------------------------------------ report
def majority(wide: pd.DataFrame) -> pd.Series:
    """The answer given most often per item; NaN when two or more answers tie for the top."""
    modes = wide.mode(axis=1)   # one column per answer tied for the top, the rest NaN
    return modes.iloc[:, 0].where(modes.notna().sum(axis=1) == 1)


def wide_of(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot(index="item_id", columns="rater", values="rating")


def human_against_model(human: pd.DataFrame, model: pd.DataFrame) -> dict:
    h = majority(wide_of(human))
    m = majority(wide_of(model)).reindex(h.index)
    both = h.notna() & m.notna()
    return {
        "maps": len(h),
        "equal": int((h[both] == m[both]).sum()),
        "with_majority": int(both.sum()),
        "human_ties": int(h.isna().sum()),
        "model_ties": int(m.isna().sum()),
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
    if any(kinds[k]["rater"].nunique() != len(HUMAN_IDS) for k in kinds):
        raise ValueError("the report needs the three human raters and the three model raters in the ratings file")
    summaries = {k: rating_summary(frame, examples) for k, frame in kinds.items()}
    table = by_kind_table(summaries)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(BY_KIND, index=False)
    link = human_against_model(kinds["human"], kinds["model"])
    same_majority = (
        f"- The human majority equals the model majority on {link['equal']} of {link['with_majority']} maps that have "
        f"a majority on both sides ({link['equal'] / link['with_majority']:.1%})."
        if link["with_majority"]
        else "- No map has a majority on both sides."
    )
    text = [
        "# Stage 12: human ratings of the 50 blind heat maps",
        "",
        "Generated by `python -m src.heatmap_ratings report` from `data/heatmap_ratings.csv`; nothing is typed by "
        "hand. `human_rater_1..3` are the raters of kind `human`; `model_rater_1..3` are disclosed model judgements "
        "(handbook §24, 2026-11-08). Question: does the hot area "
        "cover visible symptoms? Each rater saw the same 50 blind sheets (validation photos, chosen by the Stage 12 rule), "
        "without the true class or the prediction. "
        "Percentages of ratings. The table is also `results/tables/explain_ratings_by_kind.csv`.",
        "",
        *([f"**Provenance.** {PROVENANCE.read_text().strip()}", ""] if PROVENANCE.exists() else []),
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
        same_majority,
        f"- Ties: {link['human_ties']} maps have no human majority and {link['model_ties']} have no model majority "
        f"(of {link['maps']}).",
        f"- Counting a tie as a difference, they agree on {link['equal']} of {link['maps']} maps "
        f"({link['equal'] / link['maps']:.1%}).",
        "",
        "## Human raters by example category",
        "",
        "`correct` and `wrong` = the rule's confident right and wrong answers; `random` = random photos of each class; `extra` = further random photos. "
        "Seen only after rating. Percentages of the human ratings in each category.",
        "",
        md_table(percent(summaries["human"]["by_category"]).merge(
            examples["category"].value_counts().rename("maps"), left_on="category", right_index=True)),
        "",
        "Limits: 50 maps; the model raters are one model family.",
        "",
    ]
    REPORT.write_text("\n".join(text))
    print(f"report -> {REPORT}\ntable  -> {BY_KIND}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["import", "report"])
    parser.add_argument("--returned", type=Path, default=KIT / "returned", help="folder with the three filled CSVs")
    args = parser.parse_args()
    if args.action == "import":
        print(f"appended {import_ratings(args.returned)} human ratings to {RATINGS}")
    else:
        report()


if __name__ == "__main__":
    main()
