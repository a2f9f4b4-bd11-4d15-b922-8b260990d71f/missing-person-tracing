from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.fixtures import make_similar_embedding, make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
WIDE = (T0 - timedelta(hours=1), T0 + timedelta(hours=1))


def test_insert_returns_an_id_and_round_trips(store):
    rec = make_tracklet(camera_id="cam1")
    tid = store.insert_tracklet(rec)
    assert tid > 0
    got = store.get_tracklet(tid)
    assert got["camera_id"] == "cam1"
    assert got["started_at"] == rec.started_at
    assert np.allclose(got["embedding"], rec.embedding, atol=1e-6)
    assert got["protected"] is False


def test_candidates_rank_by_cosine_similarity(store):
    base = make_tracklet().embedding
    ids = {}
    for sim in (0.95, 0.80, 0.60):
        rec = make_tracklet(camera_id="cam1", embedding=make_similar_embedding(base, sim),
                            started_at=T0, ended_at=T0 + timedelta(seconds=5))
        ids[sim] = store.insert_tracklet(rec)

    got = store.candidates(base, space="osnet", window=WIDE, cameras=["cam1"], limit=50)
    assert [c.tracklet_id for c in got] == [ids[0.95], ids[0.80], ids[0.60]]
    assert got[0].score > got[1].score > got[2].score
    assert np.isclose(got[0].score, 0.95, atol=1e-3)


def test_candidates_respect_the_time_window(store):
    inside = store.insert_tracklet(make_tracklet(
        started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    store.insert_tracklet(make_tracklet(
        started_at=T0 + timedelta(days=2), ended_at=T0 + timedelta(days=2, seconds=5)))
    got = store.candidates(make_tracklet().embedding, space="osnet",
                           window=(T0 - timedelta(minutes=1), T0 + timedelta(minutes=1)),
                           cameras=["cam1"], limit=50)
    assert [c.tracklet_id for c in got] == [inside]


def test_candidates_respect_the_camera_filter(store):
    a = store.insert_tracklet(make_tracklet(camera_id="cam1"))
    store.insert_tracklet(make_tracklet(camera_id="cam2"))
    got = store.candidates(make_tracklet().embedding, space="osnet",
                           window=WIDE, cameras=["cam1"], limit=50)
    assert [c.tracklet_id for c in got] == [a]


def test_candidates_on_the_clip_space_skip_rows_without_a_clip_embedding(store):
    store.insert_tracklet(make_tracklet())
    got = store.candidates(make_tracklet().embedding, space="clip",
                           window=WIDE, cameras=["cam1"], limit=50)
    assert got == []


def test_search_matches_and_status_transitions(store):
    tid = store.insert_tracklet(make_tracklet())
    sid = store.create_search(photo_path="photos/a.jpg", query_text=None,
                              embedding=make_tracklet().embedding, space="osnet",
                              window=WIDE)
    store.record_matches(sid, [(tid, 0.91)])
    matches = store.get_matches(sid, limit=10)
    assert len(matches) == 1 and matches[0].status == "pending"
    assert matches[0].thumb_url.startswith("/media/")

    store.set_match_status(sid, tid, "confirmed")
    assert store.get_matches(sid, limit=10)[0].status == "confirmed"

    with pytest.raises(ValueError, match="status"):
        store.set_match_status(sid, tid, "maybe")


def test_the_camera_scope_is_stored_on_the_search(store):
    sid = store.create_search(None, None, make_tracklet().embedding, "osnet", WIDE,
                              cameras=["cam1", "cam2"])
    row = store.conn.execute("SELECT cameras FROM search WHERE id = %s", (sid,)).fetchone()
    assert list(row[0]) == ["cam1", "cam2"]


def test_timeline_groups_by_camera_in_time_order(store):
    t1 = store.insert_tracklet(make_tracklet(camera_id="cam1", started_at=T0,
                                             ended_at=T0 + timedelta(seconds=5)))
    t2 = store.insert_tracklet(make_tracklet(camera_id="cam2",
                                             started_at=T0 + timedelta(seconds=30),
                                             ended_at=T0 + timedelta(seconds=35)))
    t3 = store.insert_tracklet(make_tracklet(camera_id="cam1",
                                             started_at=T0 + timedelta(seconds=60),
                                             ended_at=T0 + timedelta(seconds=65)))
    sid = store.create_search(None, None, make_tracklet().embedding, "osnet", WIDE)
    store.record_matches(sid, [(t1, 0.9), (t2, 0.85), (t3, 0.8)])
    lanes = store.timeline(sid)
    assert list(lanes) == ["cam1", "cam2"]
    assert [m.tracklet_id for m in lanes["cam1"]] == [t1, t3]


def test_rejected_matches_leave_the_timeline(store):
    tid = store.insert_tracklet(make_tracklet())
    sid = store.create_search(None, None, make_tracklet().embedding, "osnet", WIDE)
    store.record_matches(sid, [(tid, 0.9)])
    store.set_match_status(sid, tid, "rejected")
    assert store.timeline(sid) == {}
