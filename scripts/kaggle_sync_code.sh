#!/usr/bin/env bash
# Upload src/, configs/, splits/, the pinned Python, the locked package list and the commit ID to the
# private Kaggle dataset described in kaggle/code_dataset/dataset-metadata.json.
#
# Usage: ./scripts/kaggle_sync_code.sh
#   ALLOW_DIRTY=1 ./scripts/kaggle_sync_code.sh   # upload uncommitted code (recorded as <commit>-dirty)
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=kaggle/code_dataset
META="$STAGE/dataset-metadata.json"
DATASET_ID=$(python3 -c "import json; print(json.load(open('$META'))['id'])")
TORCH_INDEX="https://download.pytorch.org/whl/cu128"   # keep in sync with [[tool.uv.index]] in pyproject.toml

# 1. The uploaded code must match a commit, or runs can't be traced.
COMMIT=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- src configs splits pyproject.toml uv.lock .python-version)" ]; then
    if [ "${ALLOW_DIRTY:-0}" != "1" ]; then
        echo "Error: src/, configs/, splits/, pyproject.toml or uv.lock have uncommitted changes."
        echo "Commit them first (or run with ALLOW_DIRTY=1 for a throwaway test)."
        exit 1
    fi
    COMMIT="$COMMIT-dirty"
    echo "Warning: uploading uncommitted code as $COMMIT"
fi

# 2. Rebuild the staging folder (keep only the metadata file).
echo "==> Preparing $STAGE ..."
find "$STAGE" -mindepth 1 -not -name "dataset-metadata.json" -delete
for d in src configs splits; do
    [ -d "$d" ] && cp -r "$d" "$STAGE/"
done
# Small, committed inputs used by image caches and the common-scale geometry calibration.
mkdir -p "$STAGE/data"
cp data/inventory.csv data/pixel_scale_decision.json data/groups.csv data/label_conflicts.csv \
    data/cross_dataset_overlap.csv data/duplicate_miss_rule.json "$STAGE/data/"
cp .python-version "$STAGE/"   # setup_env() builds the Kaggle environment with this exact Python
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$STAGE" -name "*.pyc" -delete

# 3. Locked versions of the main dependencies only (no dev/notebook tools), plus the
#    PyTorch CUDA index so pip can find the +cu128 torch wheels.
echo "==> Exporting locked dependencies ..."
{
    echo "--extra-index-url $TORCH_INDEX"
    uv export --no-default-groups --no-hashes --no-emit-project --quiet
} > "$STAGE/requirements-kaggle.txt"

echo "$COMMIT" > "$STAGE/CODE_VERSION"

# 4. Create the dataset the first time, otherwise upload a new version.
if uv run kaggle datasets status "$DATASET_ID" >/dev/null 2>&1; then
    echo "==> Uploading new version of $DATASET_ID ($COMMIT) ..."
    uv run kaggle datasets version -p "$STAGE" -m "code $COMMIT" --dir-mode zip
else
    echo "==> Creating $DATASET_ID ..."
    uv run kaggle datasets create -p "$STAGE" --dir-mode zip
fi

# 5. Wait until Kaggle has finished processing; notebooks pushed earlier would see the old version.
echo "==> Waiting for Kaggle to process the upload ..."
for _ in $(seq 1 60); do
    # Last line only: the CLI may print warnings (e.g. "outdated kaggle version") before the status.
    STATUS=$( (uv run kaggle datasets status "$DATASET_ID" 2>/dev/null || echo "unknown") | tail -n 1)
    if [ "$STATUS" = "ready" ]; then
        echo "==> $DATASET_ID is ready (code $COMMIT)."
        exit 0
    fi
    sleep 10
done
echo "Warning: $DATASET_ID is still '$STATUS' after 10 minutes; check before pushing notebooks."
exit 1
