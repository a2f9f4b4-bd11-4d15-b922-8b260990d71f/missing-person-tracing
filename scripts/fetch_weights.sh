#!/usr/bin/env bash
# Download OSNet re-ID weights. torchreid's pretrained=True fetches ImageNet
# weights, which do not work for re-identification.
set -euo pipefail
DEST="${1:-data/weights}"
BASE="https://huggingface.co/kaiyangzhou/osnet/resolve/main"
mkdir -p "$DEST"

declare -A FILES=(
  [osnet_x1_0_msmt17.pth]="osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
  [osnet_ain_x1_0_msmt17.pth]="osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)

for local in "${!FILES[@]}"; do
  if [[ -s "$DEST/$local" ]]; then
    echo "have $local"
  else
    echo "fetching $local"
    curl -fsSL -o "$DEST/$local" "$BASE/${FILES[$local]}"
  fi
  python3 - "$DEST/$local" <<'PY'
import sys, pathlib
size = pathlib.Path(sys.argv[1]).stat().st_size
assert size > 5_000_000, f"{sys.argv[1]} is only {size} bytes; download failed"
print(f"  ok {sys.argv[1]} ({size/1e6:.1f} MB)")
PY
done
