"""Ingest a recorded video file as one camera.

  python -m scripts.ingest_file clip.mp4 --camera cam1 \
      --start 2026-10-13T14:00:00+07:00
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from tracer.deps import get_config, get_store
from tracer.ingest import FileSource
from tracer.reid import OsnetEmbedder
from tracer.worker import Pipeline, YoloDetector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--start", required=True,
                    help="ISO 8601 with an offset, e.g. 2026-10-13T14:00:00+07:00")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--protected", action="store_true",
                    help="exempt these tracklets from retention (evaluation footage)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = datetime.fromisoformat(args.start)
    if start.tzinfo is None:
        raise SystemExit("--start must include a UTC offset, e.g. +07:00")

    cfg, store = get_config(), get_store()
    pipeline = Pipeline(cfg, args.camera, store,
                        OsnetEmbedder(args.weights or cfg.weights, device=cfg.device),
                        YoloDetector(cfg))
    written = pipeline.run(FileSource(args.video, start, cfg.sample_fps))
    print(f"wrote {written} tracklet(s) for {args.camera}")

    if args.protected:
        store.conn.execute(
            "UPDATE tracklet SET protected = true WHERE camera_id = %s", (args.camera,))
        print(f"marked {args.camera} tracklets as protected")


if __name__ == "__main__":
    main()
