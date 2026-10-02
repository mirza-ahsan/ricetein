#!/usr/bin/env bash
# Download a finished Kaggle run's outputs and file its run folders into runs/.
#
# Usage: ./scripts/kaggle_pull.sh <kernel_name>        e.g. 00-smoke, 06-embed
# Outputs land in kaggle/outputs/<kernel_name>/ (git-ignored), replacing any earlier download.
# Run folders are copied into runs/ only if a folder with the same name doesn't exist yet.
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL_NAME="${1:-}"
if [ -z "$KERNEL_NAME" ]; then
    echo "Usage: $0 <kernel_name>   (folders: $(ls kaggle/kernels | tr '\n' ' '))"
    exit 1
fi
META="kaggle/kernels/$KERNEL_NAME/kernel-metadata.json"
if [ ! -f "$META" ]; then
    echo "Error: $META not found."
    exit 1
fi
KERNEL_ID=$(python3 -c "import json; print(json.load(open('$META'))['id'])")
OUTPUT_DIR="kaggle/outputs/$KERNEL_NAME"

# 1. Only complete runs are filed into runs/; failed runs are downloaded for their log.
STATUS_LINE=$(uv run kaggle kernels status "$KERNEL_ID")
echo "$STATUS_LINE"
case "$STATUS_LINE" in
    *COMPLETE*) FILE_RUNS=1 ;;
    *ERROR*)    FILE_RUNS=0; echo "==> Run failed: downloading outputs and log only (nothing filed into runs/)." ;;
    *)          echo "Run is not finished yet; try again later."; exit 1 ;;
esac

# 2. Fresh download (old files from an earlier version would otherwise be mixed in).
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
uv run kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
echo "==> Downloaded to $OUTPUT_DIR (log: $(ls "$OUTPUT_DIR"/*.log 2>/dev/null | head -1))"

# 3. File new run folders into runs/ without overwriting existing ones.
if [ "$FILE_RUNS" = "1" ] && [ -d "$OUTPUT_DIR/runs" ]; then
    mkdir -p runs
    for run in "$OUTPUT_DIR"/runs/*/; do
        [ -d "$run" ] || continue
        name=$(basename "$run")
        if [ -e "runs/$name" ]; then
            echo "    skipped runs/$name (already exists; rename or remove it first)"
        else
            cp -r "$run" "runs/$name"
            echo "    added   runs/$name"
        fi
    done
fi
echo "==> Done."
