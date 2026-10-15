"""Frame sources: recorded files and live RTSP streams."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

RTSP_OPTIONS = "rtsp_transport;tcp|max_delay;500000|reorder_queue_size;0"


def _require_aware(t: datetime) -> None:
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise ValueError("start_at must be timezone-aware")


class FileSource:
    """Recorded video. Timestamps come from the file's start time plus the
    frame's position, placing a clip on the real-world clock."""

    def __init__(self, path: str | Path, start_at: datetime, sample_fps: float) -> None:
        _require_aware(start_at)
        self.path = Path(path)
        self.start_at = start_at
        self.sample_fps = float(sample_fps)

    def __iter__(self) -> Iterator[tuple[datetime, np.ndarray]]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {self.path}")
        try:
            native = cap.get(cv2.CAP_PROP_FPS) or self.sample_fps
            stride = max(int(round(native / self.sample_fps)), 1)
            index = 0
            while True:
                if not cap.grab():
                    break
                if index % stride == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    yield self.start_at + timedelta(seconds=index / native), frame
                index += 1
        finally:
            cap.release()


class RtspSource:
    """Live stream, newest frame only.

    OpenCV will otherwise hand over frames seconds stale while the decoder
    drains its backlog, inflating every timestamp and corrupting the timing
    check. Timestamps are arrival times; drops trigger reconnection.
    """

    def __init__(self, url: str, sample_fps: float,
                 capture_factory: Callable[[str], object] | None = None,
                 max_backoff_s: float = 8.0, max_reconnects: int | None = None) -> None:
        self.url = url
        self.sample_fps = float(sample_fps)
        self.max_backoff_s = max_backoff_s
        self.max_reconnects = max_reconnects
        self.disconnects = 0
        self._factory = capture_factory or self._default_factory

    @staticmethod
    def _default_factory(url: str):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_OPTIONS)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def __iter__(self) -> Iterator[tuple[datetime, np.ndarray]]:
        interval = 1.0 / self.sample_fps
        attempts = 0
        backoff = 0.5

        while self.max_reconnects is None or attempts < self.max_reconnects:
            attempts += 1
            cap = self._factory(self.url)
            try:
                if not cap.isOpened():
                    log.warning("cannot open %s (attempt %d)", self.url, attempts)
                else:
                    next_at = 0.0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            self.disconnects += 1
                            log.warning("stream %s dropped", self.url)
                            break
                        now = time.monotonic()
                        if now < next_at:
                            continue
                        next_at = now + interval
                        yield datetime.now(timezone.utc), frame
                    backoff = 0.5
            finally:
                cap.release()

            if self.max_backoff_s > 0:
                time.sleep(min(backoff, self.max_backoff_s))
                backoff *= 2
