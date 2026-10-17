#!/usr/bin/env bash
# Market-1501 (Zheng et al., 2015), research use. The original link is
# unreliable; this is a mirror of the same archive on HuggingFace.
set -euo pipefail
DEST="${1:-data/market1501}"
URL="https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/Market-1501-v15.09.15.zip"

mkdir -p "$DEST"
if [[ -d "$DEST/Market-1501-v15.09.15/query" ]]; then
  echo "already extracted at $DEST"
else
  echo "downloading (~150 MB zip, ~1.8 GB extracted)..."
  curl -fL --progress-bar -o "$DEST/market1501.zip" "$URL"
  python3 -m zipfile -e "$DEST/market1501.zip" "$DEST"
  rm -f "$DEST/market1501.zip"
fi

for split in query bounding_box_test bounding_box_train; do
  n=$(find "$DEST/Market-1501-v15.09.15/$split" -name '*.jpg' | wc -l)
  echo "  $split: $n images"
  test "$n" -gt 0
done
