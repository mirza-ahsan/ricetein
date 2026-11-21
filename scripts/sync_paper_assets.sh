#!/usr/bin/env bash
# Copy the chosen final tables and figures into paper/ unchanged, so paper/ is self-contained (Overleaf upload).
# Numbers are never typed: results/tables/final_*.tex and results/figures/final_*.pdf are built from runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
tables=(main paired_session conditions paired_conditions paired_locked ablations ablations_test paired_committee
        per_class confusion shortcuts explain explain_ratings cost partition_order)
figures=(confusion heatmaps partition_order paddy_zoom samples cost)
mkdir -p paper/tables paper/fig
for t in "${tables[@]}"; do cp -f "results/tables/final_${t}.tex" "paper/tables/final_${t}.tex"; done
for f in "${figures[@]}"; do cp -f "results/figures/final_${f}.pdf" "paper/fig/final_${f}.pdf"; done
echo "copied ${#tables[@]} tables and ${#figures[@]} figures into paper/"
