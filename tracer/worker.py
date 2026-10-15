"""One worker per camera: frames in, tracklets out."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Sequence

import numpy as np

from tracer.clips import ClipRecorder, FrameBuffer, write_thumbnail
from tracer.config import Config
from tracer.detect_track import (Detection, RawTracklet, TrackletBuilder,
                                 TrackletMerger, yolo_detections)
from tracer.models import TrackletRecord
from tracer.store import Store

log = logging.getLogger(__name__)

PREROLL_S = 1.5

Detector = Callable[[np.ndarray, datetime], Sequence[Detection]]


class YoloDetector:
    """YOLO26 + ByteTrack. One instance per camera: Ultralytics keeps tracker
    state inside the model, so sharing it across cameras mixes track IDs."""

    def __init__(self, cfg: Config, weights: str = "yolo26n.pt", conf: float = 0.5) -> None:
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.cfg = cfg
        self.conf = conf
        self.device = os.environ.get("DEVICE", cfg.device)

    def __call__(self, frame: np.ndarray, t: datetime) -> list[Detection]:
        result = self.model.track(
            frame, classes=[0], conf=self.conf, imgsz=self.cfg.imgsz,
            tracker="bytetrack.yaml", persist=True, verbose=False,
            device=0 if self.device in ("cuda", "gpu") else "cpu",
        )[0]
        return yolo_detections(result, frame, t)


class Pipeline:
    def __init__(self, cfg: Config, camera_id: str, store: Store,
                 embedder, detector: Detector, live: bool = False) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self.store = store
        self.embedder = embedder
        self.detector = detector
        self.live = live
        self.merger = TrackletMerger(cfg, store)
        self.buffer = FrameBuffer(seconds=PREROLL_S)
        self.written = 0
        self.latencies: list[float] = []
        self._metrics_path = cfg.data_dir / "metrics" / "latency.jsonl"
        self._recorders: dict[int, ClipRecorder] = {}
        self._builder = TrackletBuilder(cfg, camera_id, self._on_close,
                                        on_drop=self._on_drop)

    def run(self, source: Iterable[tuple[datetime, np.ndarray]]) -> int:
        last_t: datetime | None = None
        for t, frame in source:
            last_t = t
            self.buffer.add(t, frame)
            detections = list(self.detector(frame, t))
            self._open_recorders(detections, t)
            self._builder.add(t, detections)
            for rec in self._recorders.values():
                rec.add(frame)
        if last_t is not None:
            self._builder.flush(last_t)
        return self.written

    def _open_recorders(self, detections: Sequence[Detection], t: datetime) -> None:
        for d in detections:
            if d.track_id not in self._recorders:
                name = f"{self.camera_id}-{int(t.timestamp() * 1000)}-{d.track_id}.mp4"
                self._recorders[d.track_id] = ClipRecorder(
                    self.cfg.data_dir / "clips" / name,
                    fps=self.cfg.sample_fps,
                    preroll=self.buffer.window(t - timedelta(seconds=PREROLL_S), t),
                )

    def _on_drop(self, raw: RawTracklet) -> None:
        """A tracklet dropped as noise still has an open recorder; without
        closing and removing it, the writer keeps receiving every later frame
        and grows a file no row references."""
        recorder = self._recorders.pop(raw.track_id, None)
        if recorder is None:
            return
        path = recorder.close()
        try:
            path.unlink()
        except OSError as exc:
            log.warning("could not remove dropped clip %s: %s", path, exc)

    def _on_close(self, raw: RawTracklet) -> None:
        recorder = self._recorders.pop(raw.track_id, None)
        clip_path = recorder.close() if recorder else None
        try:
            embedding, n_crops = self.embedder.embed_tracklet(raw.crops)
        except ValueError as exc:
            log.warning("skipping tracklet on %s: %s", raw.camera_id, exc)
            return

        rel_clip = str(clip_path.relative_to(self.cfg.data_dir)) if clip_path else ""
        thumb_rel = (f"thumbs/{raw.camera_id}-"
                     f"{int(raw.started_at.timestamp() * 1000)}-{raw.track_id}.jpg")
        write_thumbnail(raw.crops, self.cfg.data_dir / thumb_rel)

        record = TrackletRecord(
            camera_id=raw.camera_id, started_at=raw.started_at, ended_at=raw.ended_at,
            embedding=embedding, thumb_path=thumb_rel, clip_path=rel_clip,
            n_crops=n_crops,
        )
        action, tracklet_id = self.merger.offer(record)
        if action == "new":
            self.written += 1
        log.info("%s tracklet %s on %s (%.1fs, %d crops)", action, tracklet_id,
                 raw.camera_id, raw.duration_s, n_crops)

        if self.live:
            searchable_at = datetime.now(timezone.utc)
            latency = (searchable_at - raw.ended_at).total_seconds()
            self.latencies.append(latency)
            self._write_metric(raw, tracklet_id, searchable_at, latency)

    def _write_metric(self, raw: RawTracklet, tracklet_id: int,
                      searchable_at: datetime, latency: float) -> None:
        """R8 evidence, live sources only: a recorded file's timestamps are
        fabricated, so its latency is the age of the footage."""
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"camera_id": raw.camera_id, "tracklet_id": tracklet_id,
               "last_seen": raw.ended_at.isoformat(),
               "searchable_at": searchable_at.isoformat(),
               "latency_s": round(latency, 3), "live": True}
        with self._metrics_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run one camera's ingest worker")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    from tracer.deps import get_config, get_store
    from tracer.ingest import RtspSource
    from tracer.reid import OsnetEmbedder

    cfg, store = get_config(), get_store()
    camera = cfg.cameras[args.camera]
    if camera.kind != "rtsp" or not camera.source_url:
        raise SystemExit(f"{args.camera} is not an rtsp camera with a source_url")

    pipeline = Pipeline(cfg, args.camera, store,
                        OsnetEmbedder(args.weights or cfg.weights,
                                      device=os.environ.get("DEVICE", cfg.device)),
                        YoloDetector(cfg), live=True)
    log.info("worker for %s reading %s", args.camera, camera.source_url)
    pipeline.run(RtspSource(camera.source_url, cfg.sample_fps))


if __name__ == "__main__":
    main()
