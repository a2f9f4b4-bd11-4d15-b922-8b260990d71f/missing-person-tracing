"""Search for a person from a reference photo.

  python -m scripts.search_cli ref.jpg --cameras cam1 cam2 --limit 10
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tracer.deps import get_config, get_store
from tracer.reid import PhotoEmbedder
from tracer.search import run_search

REASONS = {
    "no_tracklets_in_scope": "no sightings at all in that window and camera set",
    "none_above_tau": "no sighting passed the similarity threshold",
    "all_rejected_by_timing": "candidates found, but all impossible in time",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photo")
    ap.add_argument("--cameras", nargs="+", default=None)
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    cfg, store = get_config(), get_store()
    cameras = args.cameras or list(cfg.cameras)
    window = (datetime.fromisoformat(args.frm) if args.frm else None,
              datetime.fromisoformat(args.to) if args.to else None)

    embedding, detections = PhotoEmbedder(cfg).embed_image(Path(args.photo).read_bytes())
    print(f"{len(detections)} person(s) detected; using the largest "
          f"({detections[0]['area']} px2)")

    result = run_search(store, cfg, embedding, "osnet", window, cameras, args.limit)
    if not result.matches:
        best = f" (best score {result.best_score:.3f})" if result.best_score else ""
        print(f"no matches: {REASONS.get(result.reason, result.reason)}{best}")
        return

    print(f"{'rank':<5}{'camera':<10}{'started':<27}{'score':<8}tracklet")
    for i, m in enumerate(result.matches, 1):
        print(f"{i:<5}{m.camera_id:<10}"
              f"{m.started_at.isoformat(timespec='seconds'):<27}"
              f"{m.score:<8.3f}{m.tracklet_id}")


if __name__ == "__main__":
    main()
