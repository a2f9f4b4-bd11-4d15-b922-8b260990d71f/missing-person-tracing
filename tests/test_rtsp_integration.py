"""Exercises the real RTSP path. Skipped unless MediaMTX is reachable."""
from __future__ import annotations

import os
import socket
import time

import pytest

from tracer.ingest import RtspSource

RTSP_PORT = 8554


def _find_host() -> str | None:
    """MediaMTX runs on the host network, so it is 127.0.0.1 from the host and
    host.docker.internal from inside a compose container."""
    candidates = [os.environ["TRACER_RTSP_HOST"]] if "TRACER_RTSP_HOST" in os.environ \
        else ["host.docker.internal", "127.0.0.1"]
    for host in candidates:
        try:
            with socket.create_connection((host, RTSP_PORT), timeout=1):
                return host
        except OSError:
            continue
    return None


RTSP_HOST = _find_host()
REPLAY_URL = f"rtsp://{RTSP_HOST}:{RTSP_PORT}/replay"


def _reachable() -> bool:
    return RTSP_HOST is not None


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason="MediaMTX not running; start it with 'docker compose up -d mediamtx'")


def test_frames_arrive_from_the_replay_stream():
    src = RtspSource(REPLAY_URL, sample_fps=10, max_reconnects=1, max_backoff_s=0.0)
    frames = []
    for t, frame in src:
        frames.append((t, frame))
        if len(frames) >= 20:
            break
    assert len(frames) == 20
    assert frames[0][1].shape[0] > 0


def test_arrival_timestamps_track_real_time():
    """If OpenCV were handing over a stale decoder backlog, 20 frames at
    10 fps would arrive far faster than the two seconds they represent."""
    src = RtspSource(REPLAY_URL, sample_fps=10, max_reconnects=1, max_backoff_s=0.0)
    times, first_wall = [], None
    for t, _ in src:
        if first_wall is None:
            first_wall = time.monotonic()
        times.append(t)
        if len(times) >= 20:
            break
    elapsed = time.monotonic() - first_wall

    # Measured from the first frame, not from before connecting: opening an
    # RTSP stream takes seconds, which has nothing to do with frame pacing.
    spread = (times[-1] - times[0]).total_seconds()
    assert spread == pytest.approx(elapsed, abs=0.5)
    assert spread > 1.0, f"20 frames spanned {spread:.2f}s — decoder backlog, not live"
