# Building the paper
- Here (`halo`): from the repository root run `overleaf-pdf paper/paper.tex`. It uses the Docker TeX Live image (needs `sudo systemctl start docker` first) and the root `LatexMk` file; output: `paper/build/output.pdf` and `output.log`.
- Overleaf: upload the `paper/` folder (it holds `ref.bib`, `sections/`, `tables/`, `fig/`) and set `paper.tex` as the main file; the compiler is pdfLaTeX with biber.
- Before building, refresh the copied tables and figures with `./scripts/sync_paper_assets.sh` (copies `results/tables/final_*.tex` and `results/figures/final_*.pdf` unchanged).
- Checks: `python3 scripts/check_paper_numbers.py` lists every decimal number of the text and where it comes from.
