#!/usr/bin/env bash
# One queue of the 2026-11-14 follow-up study. Run from a clean commit.
# Known flaw: the DONE line prints even when a step before the last one of an && chain fails.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TQDM_MININTERVAL=10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
run() { uv run --locked python -m src.followup "$@"; }
case "${1:-}" in
  cnn)   run session-cnn && run extend-cnn ;;
  heads) run baselines && run session-heads && run extend-probes && run seeds && run whole 224 && run whole 512 ;;
  wide)  run wide "$2" ;;
  *)     echo "usage: $0 cnn | heads | wide <mean|max|attention>" >&2; exit 2 ;;
esac
echo "FOLLOWUP QUEUE $* DONE"
