"""Turn per-frame detections into tracklets, merging fragments."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

import numpy as np

from tracer.config import Config
from tracer.models import TrackletRecord, unit_vector
from tracer.reid import Crop

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    track_id: int
    xyxy: tuple[int, int, int, int]
    edge_cut: bool
    crop: np.ndarray


@dataclass
class RawTracklet:
    camera_id: str
    track_id: int
    started_at: datetime
    ended_at: datetime
    crops: list[Crop] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


class TrackletBuilder:
    """Accumulates detections per track ID, closing a tracklet when its track
    is lost or when it exceeds the duration cap."""

    def __init__(self, cfg: Config, camera_id: str,
                 on_close: Callable[[RawTracklet], None],
                 on_drop: Callable[[RawTracklet], None] | None = None) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self.on_close = on_close
        self.on_drop = on_drop
        self._open: dict[int, RawTracklet] = {}

    def add(self, t: datetime, detections: Sequence[Detection]) -> None:
        for d in detections:
            tr = self._open.get(d.track_id)
            if tr is None:
                tr = RawTracklet(self.camera_id, d.track_id, t, t)
                self._open[d.track_id] = tr
            tr.ended_at = t
            tr.crops.append(Crop(image=d.crop, t=t, edge_cut=d.edge_cut))

        seen = {d.track_id for d in detections}
        for track_id, tr in list(self._open.items()):
            lost = (t - tr.ended_at).total_seconds()
            if track_id not in seen and lost > self.cfg.lost_track_s:
                self._close(track_id)
            elif tr.duration_s >= self.cfg.max_tracklet_s:
                log.info("force-closing tracklet %s at the %.0fs cap",
                         track_id, self.cfg.max_tracklet_s)
                self._close(track_id)

    def flush(self, t: datetime) -> None:
        for track_id in list(self._open):
            self._close(track_id)

    def _close(self, track_id: int) -> None:
        tr = self._open.pop(track_id)
        if tr.duration_s < self.cfg.min_tracklet_s:
            log.debug("dropping %.2fs tracklet as noise", tr.duration_s)
            if self.on_drop is not None:
                self.on_drop(tr)
            return
        self.on_close(tr)


class TrackletMerger:
    """Rejoins tracklets the tracker split: same camera, small gap, high
    appearance similarity."""

    def __init__(self, cfg: Config, store) -> None:
        self.cfg = cfg
        self.store = store
        self._last: dict[str, tuple[int, TrackletRecord]] = {}

    def offer(self, rec: TrackletRecord) -> tuple[str, int]:
        prev = self._last.get(rec.camera_id)
        if prev is not None:
            prev_id, prev_rec = prev
            gap = (rec.started_at - prev_rec.ended_at).total_seconds()
            sim = float(rec.embedding @ prev_rec.embedding)
            if 0 <= gap < self.cfg.merge_gap_s and sim >= self.cfg.merge_sim:
                merged = self._merge(prev_rec, rec)
                self.store.update_tracklet(prev_id, merged, merged.n_crops)
                self._last[rec.camera_id] = (prev_id, merged)
                log.info("merged tracklet into %s (gap %.2fs, sim %.3f)",
                         prev_id, gap, sim)
                return "merged", prev_id

        new_id = self.store.insert_tracklet(rec)
        self._last[rec.camera_id] = (new_id, rec)
        return "new", new_id

    @staticmethod
    def _merge(a: TrackletRecord, b: TrackletRecord) -> TrackletRecord:
        total = a.n_crops + b.n_crops
        blended = unit_vector((a.embedding * a.n_crops + b.embedding * b.n_crops) / total)
        return TrackletRecord(
            camera_id=a.camera_id,
            started_at=min(a.started_at, b.started_at),
            ended_at=max(a.ended_at, b.ended_at),
            embedding=blended,
            thumb_path=a.thumb_path,
            clip_path=a.clip_path,
            n_crops=min(total, 10),
        )


def yolo_detections(result, frame: np.ndarray, t: datetime,
                    edge_margin: int = 2) -> list[Detection]:
    """Convert one Ultralytics result into Detections, cropping RGB patches."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return []
    h, w = frame.shape[:2]
    out: list[Detection] = []
    for xyxy, track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        edge_cut = (x1 <= edge_margin or y1 <= edge_margin
                    or x2 >= w - edge_margin or y2 >= h - edge_margin)
        out.append(Detection(track_id=int(track_id), xyxy=(x1, y1, x2, y2),
                             edge_cut=edge_cut,
                             crop=frame[y1:y2, x1:x2, ::-1].copy()))
    return out
