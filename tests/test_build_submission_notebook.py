"""Stage 14 notebook builder: valid, output-free, in stage order, safe by default, and always matching src/."""

import ast
import hashlib
import json

import nbformat

from src import build_submission_notebook as bn

REPO = bn.ROOT


def build(tmp_path):
    notebook = bn.build(REPO)
    path = tmp_path / "ricetein_final.ipynb"
    nbformat.write(notebook, path)
    return notebook, path


def test_the_notebook_is_valid_json_that_nbformat_reads(tmp_path):
    _, path = build(tmp_path)
    json.loads(path.read_text())
    read = nbformat.read(path, as_version=4)
    nbformat.validate(read)
    assert len(read.cells) > 100 and len({c.id for c in read.cells}) == len(read.cells)


def test_it_has_no_outputs_and_no_execution_counts(tmp_path):
    notebook, path = build(tmp_path)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
    assert path.stat().st_size < 3_000_000                         # nbstripout rule: no large embedded outputs


def test_every_code_cell_compiles(tmp_path):
    notebook, _ = build(tmp_path)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_stages_come_in_order_with_data_preparation_first(tmp_path):
    notebook, _ = build(tmp_path)
    headings = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown" and c.source.startswith("## Stage")]
    numbers = [int(h.split(":")[0].split()[-1]) for h in headings]
    assert numbers == sorted(numbers) == list(bn.STAGES)
    titles = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown" and c.source.startswith("## ")]
    assert titles.index("## Setup and helpers") < titles.index(headings[0]) and titles.index(headings[1]) < titles.index(headings[3])
    assert "## Final tables and figures" in titles and titles.index("## Final tables and figures") < titles.index("## Source appendix")


def test_nothing_heavy_runs_by_default(tmp_path):
    notebook, _ = build(tmp_path)
    assert "RUN_STAGES = False" in notebook.cells[2].source
    for cell in notebook.cells:
        if cell.cell_type != "code" or "import" not in cell.source or "src." not in cell.source:
            continue
        tree = ast.parse(cell.source)
        assert isinstance(tree.body[0], ast.If) and ast.unparse(tree.body[0].test) == "RUN_STAGES", cell.source[:80]


def test_every_launcher_cell_is_guarded(tmp_path):
    notebook, _ = build(tmp_path)
    launcher = [c for c in notebook.cells if "launcher" in c.metadata]
    assert len(launcher) > 15
    for cell in launcher:
        tree = ast.parse(cell.source)
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If) and ast.unparse(tree.body[0].test) == "RUN_STAGES"
        assert tree.body[0].orelse                              # the else branch says it was not re-run


def test_a_launcher_cell_with_a_multi_line_string_is_refused():
    import pytest

    with pytest.raises(ValueError, match="multi-line string"):
        bn.guarded('x = """a\nb"""', "n.ipynb")


def test_a_source_file_that_contains_a_code_fence_gets_a_longer_fence():
    cell = bn.source_cell(REPO, "build_submission_notebook")
    assert "````python" in cell.source and cell.source.rstrip().endswith("````")


def test_the_source_appendix_is_every_module_of_src(tmp_path):
    notebook, _ = build(tmp_path)
    bundled = {c.metadata["bundle_path"]: c for c in notebook.cells if "bundle_path" in c.metadata}
    on_disk = {f"src/{p.stem}.py" for p in (REPO / "src").glob("*.py") if p.stem != "__init__"}
    assert on_disk <= set(bundled)
    for relative, cell in bundled.items():
        text = (REPO / relative).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == cell.metadata["sha256"]
        assert text in cell.source                                 # the code is really in the cell, not only its hash


def test_a_module_that_is_not_listed_still_reaches_the_appendix(tmp_path):
    extra = tmp_path / "root"
    (extra / "src").mkdir(parents=True)
    (extra / "notebooks").mkdir()
    (extra / "src/brand_new.py").write_text("x = 1\n")
    (extra / "pyproject.toml").write_text("[project]\nname='x'\n")
    notebook = bn.build(extra)
    assert any(c.metadata.get("bundle_path") == "src/brand_new.py" for c in notebook.cells)


def test_final_tables_and_figures_are_shown_for_every_generator_spec(tmp_path):
    notebook, _ = build(tmp_path)
    shown = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    for spec in bn.TABLE_SPECS:
        assert f"final_{spec.name}.csv" in shown
    for spec in bn.FIGURE_SPECS:
        assert f"{bn.PROMISED_FIGURES[spec.name][1]}.png" in shown
