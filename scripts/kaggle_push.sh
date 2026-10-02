#!/usr/bin/env bash
# Copy a notebook into its Kaggle folder and start a background run on Kaggle.
#
# Usage: ./scripts/kaggle_push.sh <kernel_name>        e.g. 00-smoke, 06-embed
#   ACCELERATOR=NvidiaTeslaP100 ./scripts/kaggle_push.sh ...   # override the GPU (default: T4)
#   SKIP_SYNC_CHECK=1 ./scripts/kaggle_push.sh ...             # push even if the code dataset is out of date
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL_NAME="${1:-}"
if [ -z "$KERNEL_NAME" ]; then
    echo "Usage: $0 <kernel_name>   (folders: $(ls kaggle/kernels | tr '\n' ' '))"
    exit 1
fi
KERNEL_DIR="kaggle/kernels/$KERNEL_NAME"
META="$KERNEL_DIR/kernel-metadata.json"
if [ ! -f "$META" ]; then
    echo "Error: $META not found."
    exit 1
fi

read -r KERNEL_ID CODE_FILE USES_CODE < <(python3 - "$META" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
uses_code = any(s.endswith("/ricetein-code") for s in m.get("dataset_sources", []))
print(m["id"], m["code_file"], int(uses_code))
EOF
)

# 1. Copy the current notebook in from notebooks/ (the smoke test keeps its own notebook).
if [ -f "notebooks/$CODE_FILE" ]; then
    cp "notebooks/$CODE_FILE" "$KERNEL_DIR/$CODE_FILE"
    echo "==> Copied notebooks/$CODE_FILE"
elif [ ! -f "$KERNEL_DIR/$CODE_FILE" ]; then
    echo "Error: $CODE_FILE not found in notebooks/ or $KERNEL_DIR/."
    exit 1
fi

# 2. Refuse to run against stale code: the code on Kaggle must match what's in src/, configs/,
#    splits/ and the lock file now (commits that only touch notebooks or docs are fine).
if [ "$USES_CODE" = "1" ] && [ "${SKIP_SYNC_CHECK:-0}" != "1" ]; then
    SYNCED=$(cat kaggle/code_dataset/CODE_VERSION 2>/dev/null || echo "none")
    CODE_PATHS=(src configs splits pyproject.toml uv.lock)
    if [ "$SYNCED" = "none" ] || [[ "$SYNCED" == *-dirty ]] \
        || ! git cat-file -e "$SYNCED^{commit}" 2>/dev/null \
        || ! git diff --quiet "$SYNCED" -- "${CODE_PATHS[@]}"; then
        echo "Error: the Kaggle code dataset ($SYNCED) doesn't match your current code."
        echo "Commit, then run ./scripts/kaggle_sync_code.sh (or SKIP_SYNC_CHECK=1 to push anyway)."
        exit 1
    fi
fi

# 3. Push and start the run on a T4 (recent PyTorch builds no longer support the P100).
ACC="${ACCELERATOR:-NvidiaTeslaT4}"
echo "==> Pushing $KERNEL_ID on $ACC ..."
uv run kaggle kernels push -p "$KERNEL_DIR" --accelerator "$ACC"
echo "==> Started. Check with: uv run kaggle kernels status $KERNEL_ID"
echo "    When COMPLETE:       ./scripts/kaggle_pull.sh $KERNEL_NAME"
