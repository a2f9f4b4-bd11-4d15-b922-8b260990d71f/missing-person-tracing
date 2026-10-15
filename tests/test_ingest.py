from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.ingest import FileSource, RtspSource

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def clip(tmp_path):
    """A 5-second 20 fps synthetic clip: 100 frames."""
    import cv2
    path = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    for i in range(100):
        w.write(np.full((48, 64, 3), i % 256, dtype=np.uint8))
    w.release()
    return path


def test_a_naive_start_time_is_rejected(clip):
    with pytest.raises(ValueError, match="timezone-aware"):
        FileSource(clip, start_at=datetime(2026, 10, 13, 14, 0, 0), sample_fps=10)


def test_file_timestamps_derive_from_the_start_time_and_frame_position(clip):
    got = list(FileSource(clip, start_at=T0, sample_fps=20))
    assert got[0][0] == T0
    assert got[1][0] == pytest.approx(T0 + timedelta(seconds=0.05),
                                      abs=timedelta(milliseconds=2))
    assert got[-1][0] < T0 + timedelta(seconds=5.1)


def test_sampling_reduces_the_frame_rate(clip):
    at_20 = list(FileSource(clip, start_at=T0, sample_fps=20))
    at_10 = list(FileSource(clip, start_at=T0, sample_fps=10))
    assert len(at_20) == pytest.approx(100, abs=2)
    assert len(at_10) == pytest.approx(50, abs=2)


def test_sampled_timestamps_are_still_wall_clock_correct(clip):
    got = list(FileSource(clip, start_at=T0, sample_fps=10))
    gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(got, got[1:])]
    assert all(abs(g - 0.1) < 0.02 for g in gaps)


def test_frames_are_returned_as_arrays(clip):
    t, frame = next(iter(FileSource(clip, start_at=T0, sample_fps=10)))
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (48, 64, 3)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(FileSource(tmp_path / "nope.mp4", start_at=T0, sample_fps=10))


class FakeCapture:
    """Yields `n` frames, then fails as if the stream dropped."""

    def __init__(self, n: int, fail_forever: bool = False):
        self.n, self.i, self.fail_forever = n, 0, fail_forever
        self.released = False

    def isOpened(self):
        return not self.fail_forever

    def grab(self):
        return self.read()[0]

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((48, 64, 3), self.i % 256, dtype=np.uint8)

    def release(self):
        self.released = True

    def set(self, *_):
        return True

    def get(self, *_):
        return 0.0


def test_rtsp_reconnects_after_the_stream_drops():
    attempts = []

    def factory(url):
        attempts.append(url)
        return FakeCapture(n=5) if len(attempts) <= 2 else FakeCapture(0, fail_forever=True)

    # sample_fps is set absurdly high so the 10 fps throttle cannot discard
    # frames here; this test is about reconnection, not sampling.
    src = RtspSource("rtsp://x/cam1", sample_fps=1_000_000, capture_factory=factory,
                     max_backoff_s=0.0, max_reconnects=3)
    frames = list(src)
    assert len(frames) == 10
    assert src.disconnects >= 1
    assert len(attempts) >= 3


def test_rtsp_discards_backlog_to_hold_the_sample_rate():
    """A stream handing over frames faster than sample_fps must be thinned,
    or timestamps drift away from real time and the timing check breaks."""
    src = RtspSource("rtsp://x/cam1", sample_fps=10,
                     capture_factory=lambda url: FakeCapture(n=200),
                     max_backoff_s=0.0, max_reconnects=1)
    frames = list(src)
    assert len(frames) < 20, f"expected heavy thinning, got {len(frames)} of 200"
    assert len(frames) >= 1


def test_rtsp_timestamps_are_arrival_times_and_increase():
    src = RtspSource("rtsp://x/cam1", sample_fps=10,
                     capture_factory=lambda url: FakeCapture(n=5),
                     max_backoff_s=0.0, max_reconnects=1)
    times = [t for t, _ in src]
    assert times == sorted(times)
    assert all(t.tzinfo is not None for t in times)


def test_rtsp_releases_the_capture_on_exit():
    caps = []

    def factory(url):
        c = FakeCapture(n=3)
        caps.append(c)
        return c

    list(RtspSource("rtsp://x/cam1", sample_fps=10, capture_factory=factory,
                    max_backoff_s=0.0, max_reconnects=1))
    assert all(c.released for c in caps)
