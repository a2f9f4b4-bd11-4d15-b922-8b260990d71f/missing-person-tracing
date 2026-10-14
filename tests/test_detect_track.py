from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.config import CameraCfg, Config
from tracer.detect_track import Detection, TrackletBuilder, TrackletMerger
from tracer.fixtures import make_similar_embedding, make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="rtsp")},
        lost_track_s=2.0, min_tracklet_s=1.0, max_tracklet_s=60.0,
        merge_gap_s=3.0, merge_sim=0.8,
    )


def det(track_id: int = 1, h: int = 200) -> Detection:
    return Detection(track_id=track_id, xyxy=(10, 10, 110, 10 + h), edge_cut=False,
                     crop=np.full((h, 100, 3), 128, dtype=np.uint8))


def feed(builder, seconds, track_id=1, step=0.1):
    t = T0
    end = T0 + timedelta(seconds=seconds)
    while t <= end:
        builder.add(t, [det(track_id)])
        t += timedelta(seconds=step)
    return t


def test_a_track_closes_after_the_lost_timeout(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    last = feed(b, 3.0)
    b.add(last + timedelta(seconds=2.5), [])
    assert len(closed) == 1
    assert closed[0].track_id == 1
    assert closed[0].started_at == T0


def test_a_track_does_not_close_before_the_timeout(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    last = feed(b, 3.0)
    b.add(last + timedelta(seconds=1.5), [])
    assert closed == []


def test_tracklets_shorter_than_the_minimum_are_dropped(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    b.add(T0, [det()])
    b.add(T0 + timedelta(seconds=0.4), [det()])
    b.flush(T0 + timedelta(seconds=5))
    assert closed == []


def test_a_long_tracklet_is_force_closed_at_the_cap(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    feed(b, 65.0, step=0.5)
    assert len(closed) >= 1
    assert closed[0].duration_s == pytest.approx(60.0, abs=1.0)


def test_two_track_ids_produce_two_tracklets(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    for i in range(30):
        t = T0 + timedelta(seconds=0.1 * i)
        b.add(t, [det(1), det(2)])
    b.flush(T0 + timedelta(seconds=10))
    assert sorted(c.track_id for c in closed) == [1, 2]


def test_flush_closes_everything_open(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    feed(b, 3.0)
    b.flush(T0 + timedelta(seconds=3.1))
    assert len(closed) == 1


def test_crops_are_collected_on_the_tracklet(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    feed(b, 3.0)
    b.flush(T0 + timedelta(seconds=4))
    assert len(closed[0].crops) > 10
    assert all(c.t is not None for c in closed[0].crops)


class FakeStore:
    def __init__(self):
        self.inserted, self.updated, self.next_id = [], [], 100

    def insert_tracklet(self, rec, protected=False):
        self.next_id += 1
        self.inserted.append((self.next_id, rec))
        return self.next_id

    def update_tracklet(self, tracklet_id, rec, n_crops):
        self.updated.append((tracklet_id, rec, n_crops))


def test_the_first_tracklet_on_a_camera_is_inserted(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    action, tid = m.offer(make_tracklet(camera_id="cam1"))
    assert action == "new" and tid == 101
    assert len(store.inserted) == 1


def test_a_close_similar_follow_up_is_merged(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    second = make_tracklet(camera_id="cam1",
                           started_at=T0 + timedelta(seconds=7),
                           ended_at=T0 + timedelta(seconds=12),
                           embedding=make_similar_embedding(first.embedding, 0.9))
    action, tid = m.offer(second)
    assert action == "merged" and tid == 101
    merged = store.updated[0][1]
    assert merged.started_at == T0
    assert merged.ended_at == T0 + timedelta(seconds=12)
    assert np.isclose(np.linalg.norm(merged.embedding), 1.0, atol=1e-5)


def test_a_distant_follow_up_is_not_merged(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    action, _ = m.offer(make_tracklet(
        camera_id="cam1", started_at=T0 + timedelta(seconds=20),
        ended_at=T0 + timedelta(seconds=25),
        embedding=make_similar_embedding(first.embedding, 0.95)))
    assert action == "new"


def test_a_dissimilar_follow_up_is_not_merged(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    action, _ = m.offer(make_tracklet(
        camera_id="cam1", started_at=T0 + timedelta(seconds=6),
        ended_at=T0 + timedelta(seconds=11),
        embedding=make_similar_embedding(first.embedding, 0.5)))
    assert action == "new"


def test_tracklets_on_different_cameras_are_never_merged(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    action, _ = m.offer(make_tracklet(
        camera_id="cam2", started_at=T0 + timedelta(seconds=6),
        ended_at=T0 + timedelta(seconds=11),
        embedding=make_similar_embedding(first.embedding, 0.95)))
    assert action == "new"


def test_a_dropped_tracklet_is_announced_to_on_drop(cfg):
    closed, dropped = [], []
    b = TrackletBuilder(cfg, "cam1", closed.append, on_drop=dropped.append)
    b.add(T0, [det()])
    b.add(T0 + timedelta(seconds=0.4), [det()])
    b.flush(T0 + timedelta(seconds=5))
    assert closed == []
    assert [t.track_id for t in dropped] == [1]
