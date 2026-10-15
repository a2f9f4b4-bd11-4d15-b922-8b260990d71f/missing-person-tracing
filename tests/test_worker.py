import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.clips import ClipRecorder
from tracer.config import CameraCfg, Config
from tracer.detect_track import Detection
from tracer.worker import Pipeline

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file")},
        lost_track_s=2.0, min_tracklet_s=1.0, max_tracklet_s=60.0,
        merge_gap_s=3.0, merge_sim=0.8,
    )


class FakeEmbedder:
    def embed_tracklet(self, crops):
        from tracer.models import unit_vector
        rng = np.random.default_rng(len(crops))
        return unit_vector(rng.standard_normal(512).astype(np.float32)), min(len(crops), 10)


def source(n: int, fps: float = 10.0):
    for i in range(n):
        yield T0 + timedelta(seconds=i / fps), np.full((120, 80, 3), i % 256, dtype=np.uint8)


def detector_present_for(frames: int):
    seen = {"n": 0}

    def detect(frame, t):
        seen["n"] += 1
        if seen["n"] > frames:
            return []
        return [Detection(track_id=1, xyxy=(5, 5, 55, 105), edge_cut=False,
                          crop=np.full((100, 50, 3), 128, dtype=np.uint8))]
    return detect


def detector_present_between(first: int, last: int):
    """Absent, then present, then absent — so pre-roll frames exist."""
    seen = {"n": 0}

    def detect(frame, t):
        seen["n"] += 1
        if not (first <= seen["n"] <= last):
            return []
        return [Detection(track_id=1, xyxy=(5, 5, 55, 105), edge_cut=False,
                          crop=np.full((100, 50, 3), 128, dtype=np.uint8))]
    return detect


def frame_count(path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def test_a_clean_pass_writes_one_tracklet_with_clip_and_thumbnail(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    written = p.run(source(60))
    assert written == 1

    rows = store.conn.execute("SELECT id, camera_id, thumb_path, clip_path, n_crops "
                              "FROM tracklet").fetchall()
    assert len(rows) == 1
    tid, cam, thumb, clip, n_crops = rows[0]
    assert cam == "cam1"
    assert (cfg.data_dir / thumb).exists()
    assert (cfg.data_dir / clip).exists()
    assert 1 <= n_crops <= 10


def test_tracklet_times_match_when_the_person_was_visible(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    p.run(source(60))
    started, ended = store.conn.execute(
        "SELECT started_at, ended_at FROM tracklet").fetchone()
    assert started == T0
    assert ended == pytest.approx(T0 + timedelta(seconds=2.9),
                                  abs=timedelta(milliseconds=150))


def test_a_person_visible_too_briefly_writes_nothing(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(4))
    assert p.run(source(60)) == 0
    assert store.conn.execute("SELECT count(*) FROM tracklet").fetchone()[0] == 0


def test_the_source_ending_mid_track_still_writes_the_tracklet(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(100))
    assert p.run(source(40)) == 1


def test_the_clip_includes_pre_roll_from_before_the_tracklet(store, cfg):
    """The clip must start before the person was detected, so a reviewer sees
    them arrive. Frames at 10 fps: 16 pre-roll (1.5s) + 30 detected + ~20
    post-roll (2s lost-track) ~= 66. A single pre-roll frame would give ~51.
    """
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_between(21, 50))
    assert p.run(source(80)) == 1
    clip = store.conn.execute("SELECT clip_path FROM tracklet").fetchone()[0]
    count = frame_count(cfg.data_dir / clip)
    assert count >= 58, f"clip has only {count} frames; pre-roll is missing"


def test_the_recorder_writes_a_playable_clip(tmp_path):
    rec = ClipRecorder(tmp_path / "c.mp4", fps=10.0,
                       preroll=[(T0, np.zeros((48, 64, 3), np.uint8))])
    for i in range(20):
        rec.add(np.full((48, 64, 3), i, dtype=np.uint8))
    out = rec.close()
    assert out.exists() and rec.frame_count == 21


def test_the_recorder_records_every_preroll_frame_it_is_given(tmp_path):
    preroll = [(T0 + timedelta(seconds=0.1 * i), np.full((48, 64, 3), i, np.uint8))
               for i in range(12)]
    rec = ClipRecorder(tmp_path / "c.mp4", fps=10.0, preroll=preroll)
    rec.add(np.zeros((48, 64, 3), np.uint8))
    rec.close()
    assert rec.frame_count == 13


def test_the_recorder_tolerates_being_closed_twice(tmp_path):
    rec = ClipRecorder(tmp_path / "c.mp4", fps=10.0, preroll=[])
    rec.add(np.zeros((48, 64, 3), np.uint8))
    rec.close()
    assert rec.close().exists()


def test_a_dropped_short_tracklet_does_not_leak_its_recorder(store, cfg):
    """Tracklets under min_tracklet_s are dropped as noise without reaching
    _on_close, so their ClipRecorder is never closed or removed — it keeps
    receiving every later frame, growing an unreferenced file for the life of
    the worker."""
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(4))
    assert p.run(source(200)) == 0

    assert p._recorders == {}, f"leaked recorders: {list(p._recorders)}"
    clips = list((cfg.data_dir / "clips").glob("*.mp4"))
    sizes = {c.name: c.stat().st_size for c in clips}
    assert all(s < 200_000 for s in sizes.values()), \
        f"a dropped tracklet's clip kept growing: {sizes}"
