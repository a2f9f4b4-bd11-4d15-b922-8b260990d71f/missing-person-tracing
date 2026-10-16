"""Report the R8 latency distribution: median and worst case, not one run."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/metrics/latency.jsonl")
all_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
# R8 is a live-ingest requirement. Rows from recorded-file ingestion carry
# fabricated timestamps, so their "latency" is the age of the footage.
rows = [r for r in all_rows if r.get("live")]
skipped = len(all_rows) - len(rows)
if not rows:
    raise SystemExit(f"no live measurements in {path} ({skipped} non-live rows ignored)")
if skipped:
    print(f"ignoring {skipped} non-live row(s) from recorded-file ingestion\n")

by_camera: dict[str, list[float]] = {}
for r in rows:
    by_camera.setdefault(r["camera_id"], []).append(r["latency_s"])

print(f"{'camera':<10}{'n':<5}{'median':<10}{'p90':<10}{'worst':<10}")
for cam, values in sorted(by_camera.items()):
    values.sort()
    p90 = values[min(int(len(values) * 0.9), len(values) - 1)]
    print(f"{cam:<10}{len(values):<5}{statistics.median(values):<10.2f}"
          f"{p90:<10.2f}{max(values):<10.2f}")

allv = sorted(v for vs in by_camera.values() for v in vs)
print(f"\nall cameras: n={len(allv)} median={statistics.median(allv):.2f}s "
      f"worst={max(allv):.2f}s  (R8 requires <= 10s)")
print("R8: PASS" if max(allv) <= 10.0 else "R8: FAIL")
