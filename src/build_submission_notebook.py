"""Stage 14 (handbook file 95): assemble the final submission notebook from the real code of this checkout.

    uv run python -m src.build_submission_notebook            # writes submission/final/ricetein_final.ipynb

The notebook goes from data preparation to the final results, in stage order. It is built from

* the launcher notebooks ``notebooks/NN_*.ipynb`` (their cells are kept, but guarded: a heavy step is shown and
  is **not** run unless ``RUN_STAGES = True``),
* the saved reports and the final tables and figures (``results/``), shown as they were saved, and
* every module under ``src/`` (a source appendix with checksums), so the notebook always matches the real code.

The notebook has no outputs (nbstripout rules) and nothing in its default cells trains, splits or evaluates.
``submission/final/`` is git-ignored, like ``submission/phase4/``. The older ``scripts/build_submission_notebook.py``
belongs to the frozen Phase 4 package and is not used here.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import nbformat
from nbformat import v4

from src.config import REPO_ROOT
from src.make_figures import SPECS as FIGURE_SPECS   # the generators' own lists, so the notebook cannot drift
from src.make_tables import PROMISED_FIGURES
from src.make_tables import SPECS as TABLE_SPECS

ROOT = REPO_ROOT
OUTPUT = "submission/final/ricetein_final.ipynb"

# stage number -> (title, what it does, src modules, saved reports shown)
STAGES = {
    0: ("Groundwork", "Seeds, settings files, run records, environment check.",
        ["skeleton", "config", "seeding", "records", "kaggle_bootstrap", "smoke_check"], []),
    1: ("Inventory and the common leaf scale", "Counts, per-photo properties, leaf width and the rescale rule.",
        ["inventory", "inventory_report", "leaf_width"], ["docs/reports/inventory_report.md"]),
    2: ("Audits", "Near-duplicates, background audit, label check, Otsu review.",
        ["duplicates", "background_audit", "label_check", "audit_report", "review_tool", "otsu", "otsu_gallery"],
        ["docs/reports/audit_report.md", "docs/reports/otsu_report.md"]),
    3: ("Splits", "The locked grouped split and the session-blocked split (made once, never regenerated).",
        ["split", "session_split"], ["docs/reports/split_report.md", "docs/reports/session_split_report.md"]),
    4: ("Scoring and sanity baselines", "Metrics, bootstrap, and baselines that never see the leaf.",
        ["metrics", "stats", "sanity_baselines", "shortcut_baselines"], ["docs/reports/sanity_report.md"]),
    5: ("Whole-image models", "Fine-tuned InceptionV3, EfficientNetV2-S, ConvNeXt-T and Swin-T at 128, 224 and 448 px.",
        ["augment", "photo_dataset", "train_wholeimage", "stage5", "session_conditions", "session_conditions_report",
         "paired_conditions"],
        ["docs/reports/session_conditions_report.md", "docs/reports/paired_conditions_report.md"]),
    6: ("Tile descriptions", "Tiling at the common leaf scale and the cached frozen-encoder descriptions.",
        ["tiling", "encoders", "embed", "cache_report", "wholephoto_bags", "pipeline", "preflight"], []),
    7: ("Probes and tile heads", "Linear probes, mean and max pooling, gated attention, tuning and the follow-up study.",
        ["linear_probe", "mil_models", "train_mil", "evaluate", "followup", "followup_report", "session_study",
         "session_report", "experiment_report"],
        ["docs/reports/followup_report.md", "docs/reports/session_report.md"]),
    8: ("Ablations", "One change at a time on the session split.", ["session_ablations"],
        ["docs/reports/session_ablation_report.md"]),
    9: ("Partition order", "What happens to scores when augmentation or grouping comes in the wrong order.",
        ["partition_order", "partition_splits", "partition_report"], ["docs/reports/partition_order_report.md"]),
    10: ("Paddy Doctor zero-shot", "Every trained model scored once on the second dataset.",
         ["paddy_embed", "paddy_zero_shot"], ["docs/reports/paddy_report.md"]),
    11: ("Committee", "Combining members chosen on validation.", ["committee", "paired_committee"],
         ["docs/reports/committee_report.md", "docs/reports/paired_committee_report.md"]),
    12: ("Heat maps and the deletion test", "Do the attention weights point at the symptoms?", ["explain", "heatmap_ratings"],
         ["docs/reports/explain_report.md", "docs/reports/heatmap_ratings_report.md"]),
    13: ("Cost and compression", "Size, speed and memory.", ["cost", "compress", "distill"],
         ["docs/reports/cost_report.md"]),
    14: ("Final tables, figures and this notebook", "Everything below is generated from `runs/`.",
         ["make_tables", "make_figures", "build_submission_notebook"], []),
}
# stage number -> a note shown under the stage heading; the wording agrees with the paper
NOTES = {
    1: "The leaf-width measurements were made by disclosed AI model raters, not by people; the report names them.",
    2: "Every judgement in this stage (duplicate review, background audit, label check, Otsu review) was made by "
       "disclosed AI model raters, not by people. Where a launcher cell below speaks of team members or human parts, "
       "it describes how the tool can be used by people; no person-made judgement is part of the results.",
    5: "The paired tests of Attention MIL against ConvNeXt-T at 448 px and against the 128 px and Otsu conditions were "
       "declared after the means were seen (post-hoc), and are marked so in the report.",
    11: "The paired test of the committee against ConvNeXt-T (448 px) alone was declared after both means were seen "
        "(post-hoc). ConvNeXt-T is a member of the committee, so the two are not independent. The last line of the "
        "committee report, \"Cost column: pending: Stage 13\", was written before Stage 13 ran; the committee's cost "
        "is in the cost table (`final_cost.csv`) shown below.",
    12: "Both sets of heat-map ratings (three raters each) are disclosed AI model raters; no person rated the maps. "
        "The importer's file and column names call the second set `human`, which only records how it was imported: "
        "the provenance note in the report below says that model agents filled it in. In the paper the two sets are "
        "called model raters, set 1 and set 2.",
}
LAUNCHERS = {0: "00", 1: "01", 2: "02", 3: "03", 4: "04", 5: "05", 6: "06", 7: "07", 10: "10", 11: "11", 12: "12",
             13: "13", 14: "14"}   # notebook file prefix of each stage (several files may match)

SETUP = '''import hashlib, json, os, sys
from pathlib import Path
from IPython.display import Image, Markdown, display
import pandas as pd

# The checkout that holds runs/, results/ and docs/ (set RICETEIN_PROJECT_ROOT to point elsewhere).
PROJECT_ROOT = Path(os.environ.get("RICETEIN_PROJECT_ROOT", "..")).resolve()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT))
RUN_STAGES = False      # True re-runs the stage launchers: hours of GPU time, and it needs the data and a clean commit

def show_report(relative):
    path = PROJECT_ROOT / relative
    display(Markdown(path.read_text()) if path.exists() else Markdown(f"*pending: `{relative}` does not exist yet*"))

def show_table(relative):
    path = PROJECT_ROOT / relative
    display(pd.read_csv(path) if path.exists() else Markdown(f"*pending: `{relative}` does not exist yet*"))

def show_figure(relative):
    path = PROJECT_ROOT / relative
    display(Image(filename=str(path)) if path.exists() else Markdown(f"*pending: `{relative}` does not exist yet*"))

def guard(label):
    print(f"Not re-run ({label}). Set RUN_STAGES = True to run it; the saved results are shown below.")

print("Project:", PROJECT_ROOT, "| re-running stages:", RUN_STAGES)
'''


def git(*args: str, root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def guarded(source: str, label: str) -> str:
    """The launcher cell's code, run only when RUN_STAGES is True."""
    if '"""' in source or "'''" in source:
        raise ValueError(f"{label}: a launcher cell with a multi-line string cannot be indented safely; split the string")
    body = "\n".join(("    " + line) if line.strip() else line for line in source.splitlines())
    return f"if RUN_STAGES:\n{body}\nelse:\n    guard({label!r})\n"


def launcher_cells(root: Path, stage: int) -> list:
    """Markdown cells as they are, code cells guarded, from every notebooks/<prefix>*.ipynb of the stage."""
    prefix = LAUNCHERS.get(stage)
    cells = []
    for path in sorted((root / "notebooks").glob(f"{prefix}*.ipynb")) if prefix else []:
        launcher = nbformat.read(path, as_version=4)
        for cell in launcher.cells:
            if cell.cell_type == "markdown":
                cells.append(v4.new_markdown_cell(cell.source))
            elif cell.cell_type == "code" and cell.source.strip():
                cells.append(v4.new_code_cell(guarded(cell.source, path.name), metadata={"launcher": path.name}))
    return cells


def source_cell(root: Path, module: str) -> object:
    relative = f"src/{module}.py"
    text = (root / relative).read_text()
    fence = "`" * max(3, max((len(run) for run in re.findall("`+", text)), default=0) + 1)   # longer than any run in the code
    return v4.new_markdown_cell(
        f"#### `{relative}`\n\n{fence}python\n{text}{'' if text.endswith(chr(10)) else chr(10)}{fence}\n",
        metadata={"bundle_path": relative, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "trailing_newline": text.endswith("\n")})


def modules_present(root: Path) -> set[str]:
    return {p.stem for p in (root / "src").glob("*.py") if p.stem != "__init__"}


def tables_listing(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "results/tables").glob("final_*.csv")) if (root / "results/tables").exists() else []


def build(root: Path = ROOT) -> nbformat.NotebookNode:
    present = modules_present(root)
    mapped = {m for s in STAGES.values() for m in s[2]}
    other = sorted(present - mapped)
    revision, dirty = git("rev-parse", "--short=7", "HEAD", root=root), bool(git("status", "--porcelain", "--", "src", "configs", "splits", root=root))
    cells = [
        v4.new_markdown_cell(
            "# Resolution-preserving weakly supervised rice leaf disease recognition: final notebook\n\n"
            "This notebook goes from data preparation to the final results, in stage order. It is assembled by "
            "`src/build_submission_notebook.py` from the code in `src/`, the stage launcher notebooks and the final "
            "tables and figures in `results/`, so it matches the code of the commit it was built from.\n\n"
            "**By default nothing heavy runs.** Every launcher cell is guarded by `RUN_STAGES = False`, so the cells "
            "show the code of each step and then the saved results (reports, tables, figures), which were generated "
            "from the run records in `runs/`. No cell splits, trains or evaluates on test unless `RUN_STAGES` is set. "
            "The permanent splits are never regenerated.\n\n"
            "Every number in a table or figure below was read from a run record; none is typed by hand. A result that "
            "does not exist yet is shown as *pending* and never filled with an estimate.\n\n"
            "**Names and disclosures.** The method is called Attention MIL: a gated-attention head trained on the cached "
            "descriptions of a frozen DINOv3 ViT-B/16 for the tiles of a photo. It is compared with InceptionV3, "
            "EfficientNetV2-S, ConvNeXt-T and Swin-T fine-tuned on whole photos. The headline split is the "
            "session-blocked split. All audits and heat-map ratings are disclosed AI model ratings, never human ratings. "
            "Comparisons declared after the means were seen are marked post-hoc. The photos themselves are not part of "
            "this package; the only photo images are the two figures `final_samples` and `final_heatmaps` that the "
            "paper also shows. The pipeline diagram (Fig. 1) is drawn in TikZ inside the paper."),
        v4.new_markdown_cell("## Setup and helpers"),
        v4.new_code_cell(SETUP),
    ]
    for stage, (title, what, modules, reports) in STAGES.items():
        cells.append(v4.new_markdown_cell(f"## Stage {stage}: {title}\n\n{what}\n\nCode: "
                                          + (", ".join(f"`src/{m}.py`" for m in modules if m in present) or "none yet")
                                          + (f" (not written yet: {', '.join(f'`{m}`' for m in modules if m not in present)})"
                                             if any(m not in present for m in modules) and any(m in present for m in modules) else "")
                                          + ". The code itself is in the source appendix at the end."))
        if stage in NOTES:
            cells.append(v4.new_markdown_cell(f"**Note.** {NOTES[stage]}"))
        cells += launcher_cells(root, stage)
        for report in reports:
            cells.append(v4.new_code_cell(f"show_report({report!r})"))
    cells.append(v4.new_markdown_cell("## Final tables and figures\n\nGenerated by `python -m src.make_tables` and "
                                      "`python -m src.make_figures`; the check `python -m src.make_tables --check` "
                                      "lists what is present and what is pending."))
    cells.append(v4.new_code_cell("path = PROJECT_ROOT / 'results/tables/final_readiness.json'\n"
                                  "print(path.read_text() if path.exists() else 'readiness check not run yet')"))
    for spec in TABLE_SPECS:
        cells.append(v4.new_markdown_cell(f"### Table: {spec.name} ({spec.stage})"))
        cells.append(v4.new_code_cell(f"show_table('results/tables/final_{spec.name}.csv')"))
    for spec in FIGURE_SPECS:
        cells.append(v4.new_markdown_cell(f"### Figure: {spec.name} ({spec.stage})"))
        cells.append(v4.new_code_cell(f"show_figure('results/figures/{PROMISED_FIGURES[spec.name][1]}.png')"))
    cells.append(v4.new_markdown_cell("## Source appendix\n\nEvery module of `src/`, exactly as in the commit this notebook was "
                                      "built from (SHA-256 in each cell's metadata). Nothing here is executed."))
    for _, _, modules, _ in STAGES.values():
        for module in modules:
            if module in present:
                cells.append(source_cell(root, module))
    if other:
        cells.append(v4.new_markdown_cell("### Other modules"))
        cells += [source_cell(root, m) for m in other]
    pyproject = (root / "pyproject.toml").read_text()
    cells.append(v4.new_markdown_cell(f"#### `pyproject.toml`\n\n```toml\n{pyproject}```\n",
                                      metadata={"bundle_path": "pyproject.toml", "sha256": hashlib.sha256(pyproject.encode()).hexdigest()}))
    notebook = v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "ricetein (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"},
        "submission": {"source_revision": revision + ("-dirty" if dirty else ""), "built_by": "src/build_submission_notebook.py",
                       "stages_with_code": [s for s, v in STAGES.items() if any(m in present for m in v[2])],
                       "final_tables_present": tables_listing(root)},
    }
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"final-{index:04d}"
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None, help=f"default: <root>/{OUTPUT}")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build(root)
    nbformat.write(notebook, output)
    size = output.stat().st_size / 1e6
    print(f"Wrote {output}: {len(notebook.cells)} cells, {size:.1f} MB")


if __name__ == "__main__":
    main()
