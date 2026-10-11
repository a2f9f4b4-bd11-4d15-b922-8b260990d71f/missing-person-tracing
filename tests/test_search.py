from datetime import datetime, timedelta, timezone

import pytest

from tracer.config import CameraCfg, Config, PairCfg
from tracer.fixtures import make_similar_embedding, make_tracklet
from tracer.search import run_search

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
WIDE = (T0 - timedelta(hours=1), T0 + timedelta(hours=1))
CAMS = ["cam1", "cam2", "cam3"]


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={c: CameraCfg(id=c, name=c, kind="rtsp") for c in CAMS},
        pairs={frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0)},
    )


def test_empty_database_reports_no_tracklets_in_scope(store, cfg):
    q = make_tracklet().embedding
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert res.matches == []
    assert res.reason == "no_tracklets_in_scope"
    assert res.best_score is None


def test_candidates_below_tau_report_the_best_score_they_saw(store, cfg):
    q = make_tracklet().embedding
    store.insert_tracklet(make_tracklet(embedding=make_similar_embedding(q, 0.71),
                                        started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert res.matches == []
    assert res.reason == "none_above_tau"
    assert res.best_score == pytest.approx(0.71, abs=1e-3)


def test_matches_above_tau_are_returned_best_first(store, cfg):
    q = make_tracklet().embedding
    ids = {}
    for sim, start in ((0.95, 0), (0.85, 30), (0.80, 60)):
        ids[sim] = store.insert_tracklet(make_tracklet(
            camera_id="cam1", embedding=make_similar_embedding(q, sim),
            started_at=T0 + timedelta(seconds=start),
            ended_at=T0 + timedelta(seconds=start + 5)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert [m.tracklet_id for m in res.matches] == [ids[0.95], ids[0.85], ids[0.80]]
    assert res.reason is None


def test_the_limit_is_applied_after_filtering(store, cfg):
    q = make_tracklet().embedding
    for i in range(15):
        store.insert_tracklet(make_tracklet(
            camera_id="cam1", embedding=make_similar_embedding(q, 0.95 - i * 0.005),
            started_at=T0 + timedelta(seconds=i * 30),
            ended_at=T0 + timedelta(seconds=i * 30 + 5)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS, limit=10)
    assert len(res.matches) == 10


def test_impossible_timing_removes_a_high_scoring_match(store, cfg):
    q = make_tracklet().embedding
    keep = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.95),
        started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    store.insert_tracklet(make_tracklet(
        camera_id="cam2", embedding=make_similar_embedding(q, 0.93),
        started_at=T0 + timedelta(seconds=6), ended_at=T0 + timedelta(seconds=11)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert [m.tracklet_id for m in res.matches] == [keep]


def test_the_camera_filter_is_honoured(store, cfg):
    q = make_tracklet().embedding
    store.insert_tracklet(make_tracklet(camera_id="cam2",
                                        embedding=make_similar_embedding(q, 0.95)))
    res = run_search(store, cfg, q, "osnet", WIDE, ["cam1"])
    assert res.matches == []
    assert res.reason == "no_tracklets_in_scope"


def test_the_clip_space_uses_its_own_threshold(store, cfg):
    q = make_tracklet().embedding
    tid = store.insert_tracklet(make_tracklet(started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    store.set_clip_embedding(tid, make_similar_embedding(q, 0.30))
    res = run_search(store, cfg, q, "clip", WIDE, CAMS)
    assert [m.tracklet_id for m in res.matches] == [tid]
