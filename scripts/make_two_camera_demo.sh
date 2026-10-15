#!/usr/bin/env bash
# Ingest one clip as two cameras with a plausible walk gap between them, so
# cross-camera search can be exercised before real two-camera footage exists.
set -euo pipefail
CLIP="${1:-data/samples/people.mp4}"
BASE="${2:-2026-10-13T14:00:00+00:00}"
OFFSET_S="${3:-600}"

CAM2_START="$(python3 - "$BASE" "$OFFSET_S" <<'PY'
import sys, datetime
base = datetime.datetime.fromisoformat(sys.argv[1])
print((base + datetime.timedelta(seconds=int(sys.argv[2]))).isoformat())
PY
)"

echo "cam1 at $BASE"
python -m scripts.ingest_file "$CLIP" --camera cam1 --start "$BASE"
echo "cam2 at $CAM2_START"
python -m scripts.ingest_file "$CLIP" --camera cam2 --start "$CAM2_START"
