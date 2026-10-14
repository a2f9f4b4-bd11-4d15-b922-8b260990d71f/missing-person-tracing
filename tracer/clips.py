"""Rolling frame buffer, clip writing and thumbnails."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from tracer.reid import Crop


class FrameBuffer:
    """Keeps the most recent `seconds` of frames so a clip can start before
    the tracklet did. Bounded by time, so memory cannot grow without limit."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._frames: deque[tuple[datetime, np.ndarray]] = deque()

    def add(self, t: datetime, frame: np.ndarray) -> None:
        self._frames.append((t, frame))
        cutoff = t - timedelta(seconds=self.seconds)
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def window(self, t0: datetime, t1: datetime) -> list[tuple[datetime, np.ndarray]]:
        return [(t, f) for t, f in self._frames if t0 <= t <= t1]

    def __len__(self) -> int:
        return len(self._frames)


def write_clip(frames: Sequence[tuple[datetime, np.ndarray]],
               path: str | Path, fps: float) -> Path:
    if not frames:
        raise ValueError("cannot write a clip with no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0][1].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(fps, 1.0), (w, h))
    try:
        for _, frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def write_thumbnail(crops: Sequence[Crop], path: str | Path) -> Path:
    """Save the largest crop, which is usually the clearest view."""
    if not crops:
        raise ValueError("cannot write a thumbnail with no crops")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = max(crops, key=lambda c: c.image.shape[0] * c.image.shape[1])
    cv2.imwrite(str(path), best.image[:, :, ::-1])
    return path


class ClipRecorder:
    """Streams frames to an mp4 as they arrive, so memory stays flat however
    long the tracklet runs. Pre-roll frames come from a FrameBuffer."""

    def __init__(self, path: str | Path, fps: float,
                 preroll: Sequence[tuple[datetime, np.ndarray]] = ()) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(fps, 1.0)
        self._writer: cv2.VideoWriter | None = None
        self.frame_count = 0
        for _, frame in preroll:
            self.add(frame)

    def add(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self._writer.write(frame)
        self.frame_count += 1

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self.path
