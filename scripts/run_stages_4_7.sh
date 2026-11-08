#!/usr/bin/env bash
# Local, full-budget Stages 4–7. Run from a clean commit. No Stage 8 experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # every pinned model is cached on halo
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=10
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec flock -n cache/stages4-7.lock bash -c '
  uv run --locked python -m src.pipeline --config configs/pipeline.yaml --stage all &&
  uv run --locked python scripts/complete_stage_reports.py
'
