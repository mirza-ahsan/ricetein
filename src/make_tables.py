"""Stage 14: the final tables, built only from run records and from per-stage tables that were themselves
generated from ``runs/``.

    uv run python -m src.make_tables            # writes results/tables/final_*.{csv,tex} and final_provenance.json
    uv run python -m src.make_tables --check    # lists every promised table, figure and written commitment

Nothing is typed by hand: every number below is read from a file under ``runs/`` or from a ``results/tables``
file that a stage report generated from ``runs/``. A table whose source does not exist yet is still written, as
one line ``pending: <stage>``, so it is never left out silently. The test files of a run are opened only when
the run is listed in its study's frozen evaluation plan (golden rule 4). Percentages have one decimal.
``results/tables/final_provenance.json`` lists the source files of every table with their SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src import metrics
from src.config import REPO_ROOT
from src.records import get_code_version

TABLES = "results/tables"
FIGURES = "results/figures"
CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]   # fixed forever (class-order rule)
DASH = "–"

NAMES = {
    "inceptionv3_224": "InceptionV3 (224 px)", "efficientnetv2s_224": "EfficientNetV2-S (224 px)",
    "convnext_tiny_224": "ConvNeXt-T (224 px)", "swin_tiny_224": "Swin-T (224 px)",
    "probe_224": "Linear probe (224 px)", "probe_512": "Linear probe (512 px)",
    "mean": "Mean pooling over tiles", "max": "Max pooling over tiles", "attention": "Attention MIL (ours)",
    "whole224": "Same head, whole photo (224 px)", "whole512": "Same head, whole photo (512 px)",
    "time_neighbour": "Time neighbour (no image)", "colour_only": "Colour only", "camera_only": "Camera only",
    "background_only": "Background only", "properties_only": "File properties only",
    "background_strict": "Background only (strict)", "random_prior": "Random by class share",
    "majority": "Majority class",
    "inceptionv3": "InceptionV3", "efficientnetv2s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-T",
    "swin_tiny": "Swin-T", "convnext_tiny_448": "ConvNeXt-T (448 px)",
}
ROUNDS = 5
METRICS = {"macro_f1": "macro-F1", "balanced_accuracy": "balanced accuracy"}
HEADLINE_CONFUSION = ("attention", "swin_tiny_224", "convnext_tiny_224", "convnext_tiny_448")


# ------------------------------------------------------------------------------------------------ helpers
class Missing(Exception):
    """A source file does not exist yet; the table is written as pending."""

    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


def pct(x: float) -> str:
    return DASH if pd.isna(x) else f"{100 * x:.1f}"


def pm(mean: float, sd: float) -> str:
    return DASH if pd.isna(mean) else f"{pct(mean)} ± {pct(sd)}"


def pts(x: float) -> str:
    return DASH if pd.isna(x) else f"{100 * x:+.1f}"


def rng(low: float, high: float) -> str:
    return f"[{pct(low)}, {pct(high)}]"


def name_of(key: str) -> str:
    return NAMES.get(key, key.replace("_", " "))


def read_table(root: Path, rel: str) -> pd.DataFrame:
    if not (root / rel).exists():
        raise Missing(rel)
    return pd.read_csv(root / rel)


def read_json(root: Path, rel: str) -> Any:
    if not (root / rel).exists():
        raise Missing(rel)
    return json.loads((root / rel).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ run records
def tested_runs(root: Path, runs: list[str], plan_rel: str, files: tuple[str, ...]) -> list[Path]:
    """Paths of ``files`` of every run, after checking that the study's frozen plan lists the run: the test
    files of a run whose study has not finished its evaluate step are never opened."""
    planned = read_json(root, plan_rel)["runs"]
    if len(runs) != ROUNDS:
        raise ValueError(f"expected {ROUNDS} runs (one per round) for {plan_rel}, found {len(runs)}: {runs[:2]}")
    paths = []
    for run in runs:
        if run not in planned:
            raise ValueError(f"{run} is not in {plan_rel}: its test files must not be read")
        for name in files:
            if not (root / "runs" / run / name).exists():
                raise Missing(f"runs/{run}/{name}")
            paths.append(root / "runs" / run / name)
    return paths


def pooled_test(root: Path, runs: list[str], plan_rel: str) -> tuple[dict[str, Any], list[Path]]:
    """Scores of the pooled test predictions (every test photo once) of one method's five runs."""
    paths = tested_runs(root, runs, plan_rel, ("test_predictions.csv",))
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return metrics.pooled(frame, CLASSES), paths


def session_runs(root: Path) -> dict[str, tuple[list[str], str]]:
    """Method -> (its five runs, the plan that tested them): the headline methods, then the Stage 5 conditions."""
    found = {m: (e["runs"], "runs/session_evaluation_plan.json")
             for m, e in read_json(root, "runs/session_selection.json")["methods"].items()}
    if (root / "runs/session_conditions_selection.json").exists():
        extra = read_json(root, "runs/session_conditions_selection.json")["configurations"]
        found.update({name: (runs, "runs/session_conditions_plan.json") for name, runs in extra.items()})
    return found


def session_scores(root: Path, methods: tuple[str, ...] | None = None) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Pooled session-split test scores per method, and the files read."""
    found, scores, files = session_runs(root), {}, []
    for method in methods or tuple(found):
        if method not in found:
            raise Missing(f"runs/session_conditions_selection.json ({method})")
        scores[method], paths = pooled_test(root, *found[method])
        files += paths
    return scores, files


def test_round_scores(root: Path, runs: list[str], plan_rel: str) -> tuple[dict[str, tuple[float, float]], list[Path]]:
    """Mean and SD over the rounds of each main score, from the runs' own test_metrics.json."""
    paths = tested_runs(root, runs, plan_rel, ("test_metrics.json",))
    values = [json.loads(p.read_text()) for p in paths]
    out = {key: (float(np.mean([v[key] for v in values])), float(np.std([v[key] for v in values], ddof=1)))
           for key in METRICS}
    return out, paths


# ------------------------------------------------------------------------------------------- the builders
Built = tuple[pd.DataFrame, list[Path]]


def main_table(root: Path) -> Built:
    session, locked = read_table(root, f"{TABLES}/session_final.csv"), read_table(root, f"{TABLES}/followup_final.csv")
    sources = [root / f"{TABLES}/session_final.csv", root / f"{TABLES}/followup_final.csv"]
    locked = locked.set_index("method")
    paddy_path = root / f"{TABLES}/paddy_results.csv"
    paddy = pd.read_csv(paddy_path) if paddy_path.exists() else None
    if paddy is not None:
        sources.append(paddy_path)
    rows = []
    for r in session.itertuples():
        row = {"Method": name_of(r.method),
               "Session macro-F1": pm(r.macro_f1_mean, r.macro_f1_sd),
               "Session balanced acc.": pm(r.balanced_accuracy_mean, r.balanced_accuracy_sd)}
        if r.method in locked.index:
            lock = locked.loc[r.method]
            row["Locked macro-F1"] = pm(lock.macro_f1_mean, lock.macro_f1_sd)
            row["Locked balanced acc."] = pm(lock.balanced_accuracy_mean, lock.balanced_accuracy_sd)
        else:
            row["Locked macro-F1"] = row["Locked balanced acc."] = DASH
        row.update(paddy_cells(paddy, r.method))
        rows.append(row)
    conditions = root / f"{TABLES}/session_conditions_scores.csv"
    if conditions.exists():         # the best whole-image model at 448 px is a declared Stage 5 condition: shown, not hidden
        sources.append(conditions)
        big = pd.read_csv(conditions)
        for r in big[(big.model == "convnext_tiny") & (big.condition.astype(str) == "448")].itertuples():
            key = f"{r.model}_{r.condition}"
            rows.append({"Method": name_of(key), "Session macro-F1": pm(r.macro_f1_mean, r.macro_f1_sd),
                         "Session balanced acc.": pm(r.balanced_accuracy_mean, r.balanced_accuracy_sd),
                         "Locked macro-F1": DASH, "Locked balanced acc.": DASH, **paddy_cells(paddy, key)})
    frame = pd.DataFrame(rows).fillna(DASH)       # a method without that zoom level (no zoom levels) shows a dash
    if paddy is None:
        frame["Paddy native macro-F1"] = frame["Paddy native balanced acc."] = "pending: Stage 10"
        frame["Paddy other zooms macro-F1"] = "pending: Stage 10"
    return frame, sources


def paddy_cells(paddy: pd.DataFrame | None, method: str) -> dict[str, str]:
    """Paddy Doctor zero-shot cells of one method: the headline row, then one column per other zoom level."""
    if paddy is None:
        return {}
    mine = paddy[paddy["method"] == method]
    cells: dict[str, str] = {}
    head = mine[mine["headline"].astype(str).str.lower() == "true"]
    first = head.iloc[0] if len(head) else None
    cells["Paddy native macro-F1"] = pm(first.macro_f1_mean, first.macro_f1_sd) if first is not None else DASH
    cells["Paddy native balanced acc."] = pm(first.balanced_accuracy_mean, first.balanced_accuracy_sd) if first is not None else DASH
    other = mine[(mine["headline"].astype(str).str.lower() != "true") & (mine["zoom"].astype(str) != "-")]
    for row in other.itertuples():
        cells[f"Paddy {row.zoom} zoom macro-F1"] = pm(row.macro_f1_mean, row.macro_f1_sd)
    return cells


MODELS = ("inceptionv3", "efficientnetv2s", "convnext_tiny", "swin_tiny")


def condition_name(key: str) -> str:
    """"efficientnetv2s_224_otsu" -> "EfficientNetV2-S, 224 px + Otsu"; unknown keys are shown as written."""
    for model in MODELS:
        if key.startswith(model + "_"):
            return f"{NAMES[model]}, {condition_label(key[len(model) + 1:])}"
    return name_of(key)


def condition_label(condition: str) -> str:
    """"224_otsu" -> "224 px + Otsu"; "448" -> "448 px"."""
    size, _, extra = str(condition).partition("_")
    return f"{size} px" + (" + Otsu" if extra == "otsu" else "")


def conditions_table(root: Path) -> Built:
    path = f"{TABLES}/session_conditions_scores.csv"
    scores = read_table(root, path)
    frame = pd.DataFrame({
        "Model": [name_of(m) for m in scores.model], "Condition": [condition_label(c) for c in scores.condition],
        "Macro-F1": [pm(m, s) for m, s in zip(scores.macro_f1_mean, scores.macro_f1_sd)],
        "Balanced acc.": [pm(m, s) for m, s in zip(scores.balanced_accuracy_mean, scores.balanced_accuracy_sd)]})
    return frame, [root / path]


def per_class_table(root: Path) -> Built:
    scores, files = session_scores(root, tuple(read_json(root, "runs/session_selection.json")["methods"]))
    rows = [{"Method": name_of(m), **{c: pct(s["per_class"][c]["recall"]) for c in CLASSES}} for m, s in scores.items()]
    first = next(iter(scores.values()))
    rows.append({"Method": "Test photos (n)", **{c: str(first["per_class"][c]["support"]) for c in CLASSES}})
    frame = pd.DataFrame(rows).rename(columns={"hispa": "hispa (diagnostic)"})
    return frame, files + [root / "runs/session_selection.json", root / "runs/session_evaluation_plan.json"]


def confusion_table(root: Path) -> Built:
    scores, files = session_scores(root, HEADLINE_CONFUSION)
    rows = []
    for method, s in scores.items():
        for true, counts in zip(CLASSES, s["confusion"]):
            rows.append({"Method": name_of(method), "True class": true, **{f"pred. {c}": str(n) for c, n in zip(CLASSES, counts)}})
    return pd.DataFrame(rows), files


def paired_view(frame: pd.DataFrame, key: str, wins: str, other_wins: str | None = None, label=name_of) -> pd.DataFrame:
    """Every row of a paired-test table, with the verdict words copied as the source gives them."""
    rows = []
    for r in frame.to_dict("records"):
        row = {"Against": label(str(r[key])), "Metric": METRICS[r["metric"]],
               "Difference, pooled (points)": pts(r["difference"]),
               "Photo range": rng(r["photo_low"], r["photo_high"]), "Block range": rng(r["block_low"], r["block_high"]),
               "Rounds won by attention": f"{r[wins]}/{r['rounds']}"}
        if other_wins:
            row["Rounds won by the other"] = f"{r[other_wins]}/{r['rounds']}"
        row["Verdict (photo range)"], row["Verdict (block range)"] = r["verdict_photo"], r["verdict_block"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(rel: str) -> Callable[[Path], Built]:
    """Attention against every other method, from one paired-test table (columns B, round_wins)."""
    return lambda root: (paired_view(read_table(root, rel), "B", "round_wins"), [root / rel])


def paired_conditions_table(root: Path) -> Built:
    path = f"{TABLES}/paired_conditions.csv"          # written by the Stage 5 paired-conditions branch
    frame = read_table(root, path)
    return paired_view(frame, "condition", "attention_round_wins", "condition_round_wins", condition_name), [root / path]


def ablations_table(root: Path) -> Built:
    path = f"{TABLES}/session_ablations.csv"
    ab = read_table(root, path)
    frame = pd.DataFrame({
        "Variant": ab["variant"], "What changes": ab["question"], "Partition": "validation",
        "Macro-F1": [pm(m, s) for m, s in zip(ab.val_macro_f1_mean, ab.val_macro_f1_sd)],
        "Variant minus main (points)": [pts(d) for d in ab.difference_vs_main],
        "Rounds better than main": ab["rounds_better_than_main"].astype(str)})
    return frame, [root / path]


def numbers(text: str) -> list[float]:
    """The numbers in a cell such as ``[3.8, 6.7]`` or ``+5.2``."""
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def markdown_table(text: str, heading: str) -> pd.DataFrame | None:
    """The first pipe table after the line that starts with ``heading``; None when the section has no table."""
    lines, rows, started = text.splitlines(), [], False
    for line in lines:
        if line.startswith(heading):
            started = True
            continue
        if started and line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif started and rows:
            break
    if len(rows) < 3:
        return None
    return pd.DataFrame(rows[2:], columns=rows[0])


ABLATION_REPORT = "docs/reports/session_ablation_report.md"


def ablation_report_rows(root: Path) -> pd.DataFrame:
    """The test rows of the Stage 8 report (main minus variant), as the report wrote them."""
    if not (root / ABLATION_REPORT).exists():
        raise Missing(ABLATION_REPORT)
    rows = markdown_table((root / ABLATION_REPORT).read_text(), "## 2. Headline ablations on test")
    if rows is None:
        raise Missing(f"{ABLATION_REPORT} (test section)")
    return rows


def ablations_test_table(root: Path) -> Built:
    """The variants tested once on the session split: test scores from the runs, the readings of the report."""
    report, plan_rel = ABLATION_REPORT, "runs/session_ablations_plan.json"
    paired = ablation_report_rows(root)
    main = read_table(root, f"{TABLES}/session_final.csv").set_index("method").loc["attention"]
    plan = read_json(root, plan_rel)["runs"]
    rows, files = [], [root / report, root / plan_rel, root / f"{TABLES}/session_final.csv"]
    for r in paired.to_dict("records"):
        runs = sorted(run for run in plan if run.startswith(f"stage7_session_ablate_{r['variant']}_r"))
        own, paths = test_round_scores(root, runs, plan_rel)
        files += paths
        key = r["metric"]
        rows.append({"Variant": r["variant"], "Metric": METRICS[key],
                     "Variant test score": pm(*own[key]), "Main test score": pm(main[f"{key}_mean"], main[f"{key}_sd"]),
                     "Main minus variant, pooled (points)": r["difference (points)"], "Photo range": r["photo range"],
                     "Block range": r["block range"], "Rounds won by main": r["rounds won by main"],
                     "Reading (photo range)": r["reading (photo range)"], "Verdict (block range)": r["verdict (block range)"]})
    return pd.DataFrame(rows), files


def partition_order_table(root: Path) -> Built:
    path = f"{TABLES}/partition_order_inflation.csv"
    po = read_table(root, path)
    frame = pd.DataFrame({
        "Model": [name_of(m) for m in po.model], "Condition": po.condition, "What it is": po.meaning,
        "Macro-F1, pooled": [f"{pct(v)} {rng(lo, hi)}" for v, lo, hi in zip(po.macro_f1_pooled, po.macro_f1_low, po.macro_f1_high)],
        "Inflation vs A (points)": [f"{pts(v)} {rng(lo, hi)}" for v, lo, hi in
                                    zip(po.macro_f1_inflation, po.macro_f1_inflation_low, po.macro_f1_inflation_high)],
        "Reading": po.macro_f1_reading,
        "Balanced acc. (test, pooled)": [pct(v) for v in po.balanced_accuracy_pooled],
        "Balanced acc. inflation (points)": [pts(v) for v in po.balanced_accuracy_inflation]})
    frame.attrs["note"] = ("Set-ups: " + "; ".join(f"{c}: {m}" for c, m in dict(zip(po.condition, po.meaning)).items())
                           + ". Scores are pooled over all scored photos of the set-up (set-up D scores training photos too).")
    return frame, [root / path]


def shortcuts_table(root: Path) -> Built:
    path = f"{TABLES}/followup_shortcuts.csv"
    sc = read_table(root, path)
    frame = pd.DataFrame({"Baseline": [name_of(b) for b in sc.baseline]})
    for split in ("locked", "session"):
        for key, label in (("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced acc.")):
            frame[f"{split.capitalize()} {label}"] = [pm(m, s) for m, s in zip(sc[f"{split}_{key}_mean"], sc[f"{split}_{key}_sd"])]
    return frame, [root / path]


def passthrough(candidates: tuple[str, ...]) -> Callable[[Path], Built]:
    """A stage that is not built yet (committee, cost): its first existing table is shown as written, floats
    of score columns as percentages. The stage's own session adapts this when it knows its columns."""
    def build(root: Path) -> Built:
        for rel in candidates:
            if (root / rel).exists():
                frame = pd.read_csv(root / rel)
                for col in frame.columns:
                    if pd.api.types.is_float_dtype(frame[col]):
                        score = re.search(r"f1|accuracy|recall|precision", col)
                        frame[col] = [DASH if pd.isna(v) else pct(v) if score else f"{v:.3g}" for v in frame[col]]
                return frame, [root / rel]
        raise Missing(candidates[0])
    return build


def deletion_table(root: Path) -> Built:
    path = f"{TABLES}/explain_deletion_test.csv"
    d = read_table(root, path)
    frame = pd.DataFrame({
        "Tiles removed (k)": d["k"].astype(str),
        "Macro-F1, nothing removed": [pct(v) for v in d.full_macro_f1],
        "Macro-F1, top-k attended removed": [pct(v) for v in d.top_macro_f1],
        "Macro-F1, random k removed": [pct(v) for v in d.random_macro_f1],
        "True-class probability, random minus top-k removed (points)": [pts(v) for v in d.gap_p_true],
        "Photo range": [rng(lo, hi) for lo, hi in zip(d.gap_low, d.gap_high)],
        "Block range": [rng(lo, hi) for lo, hi in zip(d.gap_block_low, d.gap_block_high)],
        "Rounds won": [f"{w}/{n}" for w, n in zip(d.round_wins, d.rounds)],
        "Verdict": d["verdict"]})
    return frame, [root / path]


@dataclass(frozen=True)
class Spec:
    name: str
    stage: str
    caption: str
    build: Callable[[Path], Built]
    wide: bool = False
    bold: tuple[str, ...] = ()
    tex_skip: tuple[str, ...] = ()


SPECS = [
    Spec("main", "Stage 10 (Paddy Doctor columns)",
         "Main results: test macro-F1 and balanced accuracy (\\%), mean $\\pm$ SD over the 5 rounds. Session: "
         "session-blocked split, the headline (each method tuned on its own validation photos, each selected "
         "configuration tested once). ConvNeXt-T at 448 px is a Stage 5 condition (learning rate chosen at 224 px, tested "
         "once). Locked: the locked split of the follow-up study. Paddy Doctor: zero-shot, nothing trained or tuned on it.", main_table, wide=True),
    Spec("conditions", "Stage 5",
         "Whole-image models at other input conditions: test macro-F1 and balanced accuracy (\\%), mean $\\pm$ SD "
         "over 5 rounds, session-blocked split.", conditions_table),
    Spec("per_class", "Stage 7 (session evaluation)",
         "Per-class recall (\\%) on the session-blocked test photos, pooled over the 5 rounds (every test photo "
         "once). Hispa is the pre-declared diagnostic class (P1-10).", per_class_table, bold=("hispa (diagnostic)",)),
    Spec("confusion", "Stage 7 (session evaluation)",
         "Confusion matrices (photo counts) of the headline methods on the session-blocked test photos, pooled "
         "over the 5 rounds. Rows are the true class.", confusion_table, wide=True),
    Spec("paired_session", "Stage 7 (session evaluation)",
         "Attention against every other method, session-blocked split, test. Difference of the pooled scores in points "
         "(attention minus the other; it can differ from the difference of the round means in the main table); 95\\% photo "
         "and block bootstrap ranges; verdict words as in the source.",
         paired_table(f"{TABLES}/session_paired.csv"), wide=True),
    Spec("paired_conditions", "Stage 5 paired test (branch stage5/paired-conditions)",
         "Attention against each Stage 5 condition, session-blocked split, test. Difference of the pooled scores in points (attention minus the condition).", paired_conditions_table, wide=True),
    Spec("paired_locked", "Stage 7 (locked split)",
         "Attention against every other method, locked split, test. Difference of the pooled scores in points (attention minus the other).", paired_table(f"{TABLES}/followup_paired.csv"), wide=True),
    Spec("ablations", "Stage 8",
         "Stage 8 ablations, validation macro-F1 (\\%), mean $\\pm$ SD over the 5 rounds, session-blocked split. "
         "Validation scores favour the main configuration, whose settings were chosen on them.", ablations_table, wide=True),
    Spec("ablations_test", "Stage 8",
         "The two variants tested once on the session-blocked test photos: scores (\\%, mean $\\pm$ SD over 5 rounds) and "
         "the difference of the pooled scores, main minus variant, in points (so it can differ from the difference of the means).", ablations_test_table, wide=True),
    Spec("partition_order", "Stage 9",
         "Partition-order experiment: pooled macro-F1 (\\%) and its inflation over the correct set-up (A), with 95\\% ranges.", partition_order_table, wide=True, tex_skip=("What it is",)),
    Spec("shortcuts", "Stage 4 / follow-up",
         "Sanity and shortcut controls that never see the leaf: test scores (\\%), mean $\\pm$ SD over 5 rounds, "
         "on the locked and on the session-blocked split.", shortcuts_table, wide=True),
    Spec("committee", "Stage 11", "Committee of tile-head members.",
         passthrough(("results/tables/committee_results.csv", "results/tables/committee.csv"))),
    Spec("explain", "Stage 12",
         "Deletion test on the session-blocked test photos: removing the most attended tiles against removing "
         "random tiles.", deletion_table, wide=True),
    Spec("cost", "Stage 13", "Parameters, storage, time per photo and peak memory.",
         passthrough(("results/tables/cost_results.csv", "results/tables/cost.csv"))),
]


# ------------------------------------------------------------------------------------------------ output
LATEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
         "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def escape(text: str) -> str:
    """LaTeX-safe text: special characters escaped; ± and dashes turned into commands."""
    out = "".join(LATEX.get(ch, ch) for ch in str(text))
    out = out.replace("±", r"$\pm$").replace("–", "--").replace("—", "---")
    return re.sub(r"(?<![\w$])-(?=\d)", "$-$", out)       # a minus sign, not a hyphen


def column_spec(frame: pd.DataFrame) -> str:
    numeric = re.compile(r"^([+\-–]?\d|\[|–$|\d+/\d+)")
    return "l" + "".join("r" if all(numeric.match(str(v)) for v in frame[c]) else "l" for c in frame.columns[1:])


def comment_sources(sources: list[str]) -> list[str]:
    """Source files for the .tex header: the files of many runs are grouped (all are in final_provenance.json)."""
    kept, grouped = [], {}
    for s in sources:
        parts = s.split("/")
        if parts[0] == "runs" and len(parts) == 3:
            grouped[parts[2]] = grouped.get(parts[2], 0) + 1
        else:
            kept.append(s)
    return kept + [f"{n} files runs/<run>/{name} (listed with their SHA-256 in final_provenance.json)" for name, n in grouped.items()]


def to_latex(name: str, frame: pd.DataFrame, caption: str, sources: list[str], wide: bool,
             bold: tuple[str, ...], skip: tuple[str, ...]) -> str:
    """A booktabs table ready to ``\\input``. ``caption`` is LaTeX already; headers and cells are escaped here."""
    frame = frame.drop(columns=[c for c in skip if c in frame.columns])
    env = "table*" if wide else "table"
    lines = [f"% final_{name}.tex: built by src/make_tables.py; do not edit by hand."]
    lines += [f"% source: {s}" for s in comment_sources(sources)]
    lines += [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:final_{name.replace('_', '-')}}}",
              "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
              f"\\begin{{tabular}}{{{column_spec(frame)}}}", "\\toprule",
              " & ".join(escape(c) for c in frame.columns) + r" \\", "\\midrule"]
    for row in frame.itertuples(index=False):
        cells = [f"\\textbf{{{escape(v)}}}" if c in bold else escape(v) for c, v in zip(frame.columns, row)]
        lines.append(" & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def build_one(root: Path, spec: Spec) -> tuple[pd.DataFrame, list[Path], str, str | None]:
    """The table of one spec with its status and reason. A missing source gives a one-line pending table; a guard
    error (a run outside its frozen plan, a wrong run count) gives an error table and is raised by ``build_all``."""
    try:
        frame, sources = spec.build(root)
    except Missing as missing:
        return pd.DataFrame({"status": [f"pending: {spec.stage}"]}), [], "pending", f"missing {missing.path}"
    except ValueError as error:
        return pd.DataFrame({"status": [f"error: {error}"]}), [], "error", str(error)
    partly = any(frame[c].astype(str).str.startswith("pending:").any() for c in frame.columns)
    return frame, sources, "partly pending" if partly else "present", None


def build_all(root: Path = REPO_ROOT, out: Path | None = None) -> dict[str, Any]:
    """Every table, even when one fails: errors are raised at the end, after the other tables are written."""
    out = out or root / TABLES
    out.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {"code_version": get_code_version(), "tables": {}}
    errors = []
    for spec in SPECS:
        frame, sources, status, reason = build_one(root, spec)
        rels = [str(p.relative_to(root)) for p in sources]
        caption = spec.caption + {"pending": f" Pending: {spec.stage}.", "partly pending": f" Cells marked pending wait for {spec.stage}."}.get(status, "")
        if note := frame.attrs.get("note"):
            caption += " " + escape(note)
        frame.to_csv(out / f"final_{spec.name}.csv", index=False)
        (out / f"final_{spec.name}.tex").write_text(to_latex(spec.name, frame, caption, rels, spec.wide, spec.bold, spec.tex_skip))
        provenance["tables"][spec.name] = {"status": status, "stage": spec.stage, "reason": reason, "rows": len(frame),
                                           "sources": {rel: sha256(root / rel) for rel in sorted(set(rels))}}
        if status == "error":
            errors.append(f"{spec.name}: {reason}")
        print(f"{spec.name:18s} {status:15s} {len(frame)} rows" + (f"  ({reason})" if reason else ""))
    (out / "final_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if errors:
        raise RuntimeError("tables with errors (the others were written): " + "; ".join(errors))
    return provenance


# ------------------------------------------------------------------------------------------- readiness
PROMISED_FIGURES = {   # name -> (stage that produces it, file stem under results/figures)
    "samples": ("Stage 1 data (main checkout only)", "final_samples"),
    "confusion": ("Stage 7", "final_confusion"),
    "partition_order": ("Stage 9", "final_partition_order"),
    "ablations": ("Stage 8", "final_ablations"),
    "paddy_zoom": ("Stage 10", "final_paddy_zoom"),
    "deletion": ("Stage 12", "final_deletion"),
    "heatmaps": ("Stage 12", "final_heatmaps"),
    "cost": ("Stage 13", "final_cost"),
}
DATASET_REPORTS = {   # handbook Stage 14: recount, background audit, duplicates
    "recount": "docs/reports/inventory_report.md", "background audit": "docs/reports/audit_report.md",
    "duplicates": "docs/reports/split_report.md"}
EVIDENCE = {   # commitment -> what must exist; "table:x" needs a built table, "figure:x" a built figure
    "P1-1": ["data/inventory.csv", "docs/reports/inventory_report.md"],
    "P1-2": ["docs/reports/audit_report.md"],
    "P1-3": ["splits/folds.csv", "splits/session_blocked/folds.csv", "docs/reports/split_report.md"],
    "P1-4": ["tests/test_leakage.py", "tests/test_mil_leakage.py", "tests/test_split.py"],
    "P1-5": ["docs/reports/paddy_report.md", "splits/paddy4.csv"],
    "P1-6": ["table:main"],
    "P1-7": ["results/tables/session_conditions_scores.csv", "table:ablations_test"],
    "P1-8": ["table:per_class", "table:confusion", "table:main"],
    "P1-10": ["table:per_class"],
    "P1-14": ["splits/folds.csv", "data/DATASETS.md", "docs/reports/inventory_report.md", "docs/reports/audit_report.md"],
    "P1-15": ["docs/reports/explain_report.md"],
    "P1-16": ["docs/reports/paddy_report.md", "docs/reports/audit_report.md"],
    "O1": ["docs/reports/paddy_report.md", "data/paddy_inventory.csv"],
    "O2": ["results/tables/session_final.csv", "tests/test_leakage.py"],
    "O3": ["results/tables/session_conditions_scores.csv", "table:main"],
    "O4": ["table:main"],
    "O5": ["table:ablations", "table:ablations_test", "table:conditions"],
    "O6": ["docs/reports/paddy_report.md", "docs/reports/explain_report.md", "table:explain"],
    "O7": ["table:cost", "figure:cost"],
}   # P1-9, P1-11, P1-12, P1-13 and P1-17 are checked by a person ("manual")


def commitments(root: Path) -> list[dict[str, str]]:
    """The written commitments of docs/DELIVERABLES.md section 6: id, text, delivered by, status in that file."""
    text = (root / "docs/DELIVERABLES.md").read_text()
    section = text.split("## 6. Written commitments", 1)[1].split("\n## 7", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and re.match(r"^(P1-\d+|O\d+)$", cells[0]) and len(cells) >= 4:
            rows.append({"id": cells[0], "commitment": cells[1], "delivered_by": cells[2], "document_status": cells[3]})
    return rows


def present(root: Path, token: str, built: dict[str, Any], figures: dict[str, Any]) -> bool:
    if token.startswith("table:"):
        return built.get(token[6:], {}).get("status") in ("present", "partly pending")   # P1-6, O3, O4 need the main table, not its Paddy columns
    if token.startswith("figure:"):
        return figures.get(token[7:], {}).get("status") == "present"
    return (root / token).exists()


def stale(root: Path, entry: dict[str, Any]) -> bool:
    """True when a source file of a built table or figure changed or vanished since it was built."""
    return any(not (root / rel).exists() or sha256(root / rel) != digest for rel, digest in entry.get("sources", {}).items())


def check(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Every promised table, figure and written commitment as present or pending, with the stage."""
    prov_path, fig_path = root / TABLES / "final_provenance.json", root / FIGURES / "final_figures.json"
    built = json.loads(prov_path.read_text())["tables"] if prov_path.exists() else {}
    figures = json.loads(fig_path.read_text())["figures"] if fig_path.exists() else {}
    ready: dict[str, Any] = {"tables": {}, "figures": {}, "dataset_reports": {}, "commitments": []}
    for spec in SPECS:
        state = built.get(spec.name, {"status": "not built"})
        ready["tables"][spec.name] = {"status": "stale: rebuild" if stale(root, state) else state["status"], "stage": spec.stage}
    for name, (stage, _) in PROMISED_FIGURES.items():
        state = figures.get(name, {"status": "not built"})
        ready["figures"][name] = {"status": "stale: rebuild" if stale(root, state) else state["status"], "stage": stage}
    pipeline = [p for pattern in ("paper/fig/pipeline*", f"{FIGURES}/final_pipeline*") for p in root.glob(pattern)]
    ready["figures"]["pipeline_diagram"] = {"status": "present" if pipeline else "pending", "stage": "drawn by hand with the paper"}
    for name, rel in DATASET_REPORTS.items():
        ready["dataset_reports"][name] = {"status": "present" if (root / rel).exists() else "pending", "file": rel}
    for c in commitments(root):
        needs = EVIDENCE.get(c["id"])
        if needs is None:
            status, missing = "manual", []
        else:
            missing = [t for t in needs if not present(root, t, ready['tables'], ready['figures'])]
            status = "present" if not missing else "pending"
        ready["commitments"].append({**c, "status": status, "missing": missing})
    counts = {k: sum(v["status"] in ("present", "partly pending") for v in ready[k].values()) for k in ("tables", "figures", "dataset_reports")}
    counts["commitments"] = sum(c["status"] == "present" for c in ready["commitments"])
    ready["present_counts"] = counts
    return ready


def write_readiness(root: Path = REPO_ROOT) -> dict[str, Any]:
    ready = check(root)
    (root / TABLES / "final_readiness.json").write_text(json.dumps(ready, indent=2) + "\n")
    return ready


def print_readiness(ready: dict[str, Any]) -> None:
    for kind in ("tables", "figures", "dataset_reports"):
        print(f"\n{kind}:")
        for name, v in ready[kind].items():
            stage = v.get("stage", v.get("file", ""))
            print(f"  {v['status']:15s} {name}" + ("" if v["status"] == "present" else f"   <- {stage}"))
    print("\ncommitments (docs/DELIVERABLES.md section 6):")
    for c in ready["commitments"]:
        tail = "" if c["status"] == "present" else f"   <- {c['delivered_by']}" + (f"; missing {', '.join(c['missing'])}" if c["missing"] else "")
        print(f"  {c['status']:8s} {c['id']:6s}{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="list what is present and what is pending; write final_readiness.json")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="the checkout to read and write (tests use a fixture)")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.check:
        print_readiness(write_readiness(root))
    else:
        build_all(root)


if __name__ == "__main__":
    main()
