#!/usr/bin/env bash
# Upload pinned DINOv3 weights (+ licence) to the private Kaggle dataset described in
# kaggle/weights_dataset/dataset-metadata.json, so Kaggle notebooks load them locally:
# no Hugging Face token or Kaggle secret needed, and every run uses identical weights.
#
# Usage: ./scripts/kaggle_upload_weights.sh                 # DINOv3 ViT-S/16 and ViT-B/16 (~410 MB)
#        INCLUDE_LARGE=1 ./scripts/kaggle_upload_weights.sh  # also ViT-L/16 (+1.2 GB, encoder E5)
# Needs: Hugging Face access to the DINOv3 models on this machine (uv run hf auth whoami).
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=kaggle/weights_dataset
META="$STAGE/dataset-metadata.json"
DATASET_ID=$(python3 -c "import json; print(json.load(open('$META'))['id'])")
MODELS=(facebook/dinov3-vits16-pretrain-lvd1689m facebook/dinov3-vitb16-pretrain-lvd1689m)
if [ "${INCLUDE_LARGE:-0}" = "1" ]; then
    MODELS+=(facebook/dinov3-vitl16-pretrain-lvd1689m)
fi

echo "==> Preparing $STAGE ..."
find "$STAGE" -mindepth 1 -not -name "dataset-metadata.json" -delete

# Copy each model at its current Hugging Face commit; record the commits in REVISIONS.txt.
uv run python - "$STAGE" "${MODELS[@]}" <<'EOF'
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

stage, repos = Path(sys.argv[1]), sys.argv[2:]
lines = []
for repo in repos:
    sha = HfApi().model_info(repo).sha
    src = Path(snapshot_download(repo, revision=sha))
    dst = stage / repo.replace("/", "--")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".gitattributes"))  # follows symlinks
    lines.append(f"{repo} {sha}")
    print(f"    {repo} @ {sha[:10]}")
shutil.copy(stage / repos[0].replace("/", "--") / "LICENSE.md", stage / "LICENSE.md")
(stage / "REVISIONS.txt").write_text("\n".join(lines) + "\n")
EOF
du -sh "$STAGE"

if uv run kaggle datasets status "$DATASET_ID" >/dev/null 2>&1; then
    echo "==> Uploading new version of $DATASET_ID ..."
    uv run kaggle datasets version -p "$STAGE" -m "weights $(date +%F)" --dir-mode zip
else
    echo "==> Creating $DATASET_ID (private) ..."
    uv run kaggle datasets create -p "$STAGE" --dir-mode zip
fi

echo "==> Waiting for Kaggle to process the upload ..."
for _ in $(seq 1 90); do
    STATUS=$(uv run kaggle datasets status "$DATASET_ID" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "ready" ]; then
        echo "==> $DATASET_ID is ready."
        exit 0
    fi
    sleep 10
done
echo "Warning: $DATASET_ID is still '$STATUS' after 15 minutes; check before pushing notebooks."
exit 1
