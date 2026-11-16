#!/usr/bin/env bash
# One queue of the session-split tuning protocol (declared 2026-11-16; manuals/HANDOFF.md).
# Run from a clean commit. Every action is resumable: relaunching a queue skips finished work.
# Commands are separate statements on purpose: with `set -e` a failed one stops the queue before the DONE line.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TQDM_MININTERVAL=10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
run() { uv run --locked python -m src.session_study "$@"; }
queue="${1:-}"
case "$queue" in
  cnn)
    run cnn
    ;;
  heads)
    run probes
    for name in whole224 whole224_wide whole512 whole512_wide; do
      run study "$name"
    done
    ;;
  mean|max|attention)
    run study "$queue"
    run study "${queue}_wide"
    ;;
  *)
    echo "usage: $0 cnn | heads | mean | max | attention" >&2
    exit 2
    ;;
esac
echo "SESSION QUEUE $queue DONE"
