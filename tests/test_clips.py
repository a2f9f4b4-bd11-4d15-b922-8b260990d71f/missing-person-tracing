import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.clips import FrameBuffer, write_clip, write_thumbnail
from tracer.reid import Crop

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def frame(i: int, h: int = 64, w: int = 128) -> np.ndarray:
    return np.full((h, w, 3), i % 256, dtype=np.uint8)


def test_the_buffer_discards_frames_older_than_its_window():
    buf = FrameBuffer(seconds=2.0)
    for i in range(50):
        buf.add(T0 + timedelta(seconds=0.1 * i), frame(i))
    held = buf.window(T0, T0 + timedelta(seconds=100))
    span = (held[-1][0] - held[0][0]).total_seconds()
    assert span <= 2.0 + 0.1
    assert len(held) <= 22


def test_the_buffer_returns_only_the_requested_window():
    buf = FrameBuffer(seconds=10.0)
    for i in range(50):
        buf.add(T0 + timedelta(seconds=0.1 * i), frame(i))
    got = buf.window(T0 + timedelta(seconds=1.0), T0 + timedelta(seconds=2.0))
    assert all(T0 + timedelta(seconds=1.0) <= t <= T0 + timedelta(seconds=2.0)
               for t, _ in got)
    assert len(got) == pytest.approx(11, abs=1)


def test_an_empty_window_returns_nothing():
    buf = FrameBuffer(seconds=5.0)
    assert buf.window(T0, T0 + timedelta(seconds=1)) == []


def test_write_clip_produces_a_playable_mp4(tmp_path):
    frames = [(T0 + timedelta(seconds=0.1 * i), frame(i)) for i in range(30)]
    out = write_clip(frames, tmp_path / "clip.mp4", fps=10.0)
    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width,height",
         "-count_frames", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout
    assert "width=128" in probe and "height=64" in probe
    assert "nb_read_frames=30" in probe


def test_write_clip_with_no_frames_raises(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        write_clip([], tmp_path / "empty.mp4", fps=10.0)


def test_the_thumbnail_is_the_largest_crop(tmp_path):
    crops = [Crop(frame(1, h=100, w=50), T0),
             Crop(frame(2, h=300, w=150), T0 + timedelta(seconds=1)),
             Crop(frame(3, h=120, w=60), T0 + timedelta(seconds=2))]
    out = write_thumbnail(crops, tmp_path / "thumb.jpg")
    assert out.exists()
    import cv2
    img = cv2.imread(str(out))
    assert img.shape[0] == 300 and img.shape[1] == 150


def test_write_thumbnail_with_no_crops_raises(tmp_path):
    with pytest.raises(ValueError, match="no crops"):
        write_thumbnail([], tmp_path / "thumb.jpg")


def test_parent_directories_are_created(tmp_path):
    out = write_thumbnail([Crop(frame(1, 80, 40), T0)], tmp_path / "a" / "b" / "t.jpg")
    assert out.exists()
