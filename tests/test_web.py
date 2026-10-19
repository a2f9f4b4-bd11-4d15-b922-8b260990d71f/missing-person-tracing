from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tracer.api import app
from tracer.deps import get_config, get_embedder, get_store
from tracer.fixtures import make_similar_embedding, make_tracklet
from tests.test_api import QUERY, FakeEmbedder, cfg, form, photo  # noqa: F401

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def client(store, cfg, tmp_path):
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "thumbs" / "1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (tmp_path / "secret.txt").write_text("not media")
    cfg = type(cfg)(**{**cfg.__dict__, "data_dir": tmp_path})
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_config] = lambda: cfg
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_upload_page_lists_the_cameras(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "cam1" in r.text and "cam2" in r.text
    assert "<form" in r.text


def test_the_results_fragment_shows_camera_time_score_and_thumbnail(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15),
        thumb_path="thumbs/1.jpg", clip_path="clips/1.mp4"))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    r = client.get(f"/ui/searches/{sid}/results")
    assert r.status_code == 200
    assert "cam1" in r.text
    assert "/media/thumbs/1.jpg" in r.text
    assert "/media/clips/1.mp4" in r.text
    assert "0.9" in r.text
    assert "Confirm" in r.text and "Reject" in r.text


def test_an_empty_result_tells_the_reviewer_why(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.70),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    r = client.get(f"/ui/searches/{sid}/results")
    assert "0.7" in r.text
    assert "threshold" in r.text.lower()


def test_the_timeline_page_renders_one_lane_per_camera(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    client.get(f"/api/searches/{sid}/matches")
    r = client.get(f"/ui/searches/{sid}/timeline")
    assert r.status_code == 200
    assert 'hx-trigger="every 3s"' in r.text
    assert "cam1" in r.text


def test_media_serves_a_file_inside_the_data_directory(client):
    r = client.get("/media/thumbs/1.jpg")
    assert r.status_code == 200
    assert r.content.startswith(b"\xff\xd8")


def test_media_refuses_path_traversal(client):
    for bad in ("/media/../secret.txt", "/media/..%2fsecret.txt",
                "/media/thumbs/../../secret.txt"):
        assert client.get(bad).status_code in (404, 400), bad


def test_media_refuses_an_absolute_path(client):
    assert client.get("/media//etc/passwd").status_code in (404, 400)


def test_the_form_submit_route_returns_the_results_fragment(client, store):
    """The primary UI path: a multipart form post, not the JSON API."""
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15),
        thumb_path="thumbs/1.jpg", clip_path="clips/1.mp4"))
    r = client.post("/ui/searches", files=photo(),
                    data={"from": "2026-10-13T14:00", "to": "2026-10-13T15:00",
                          "cameras": ["cam1", "cam2"]})
    assert r.status_code == 200, r.text
    assert "/media/thumbs/1.jpg" in r.text
    assert "Confirm" in r.text


def test_the_form_submit_route_rejects_a_reversed_window(client):
    r = client.post("/ui/searches", files=photo(),
                    data={"from": "2026-10-13T15:00", "to": "2026-10-13T14:00",
                          "cameras": ["cam1"]})
    assert r.status_code == 422
    assert "window" in r.text.lower()


def test_the_form_submit_route_stores_real_timestamps(client, store):
    client.post("/ui/searches", files=photo(),
                data={"from": "2026-10-13T14:00", "to": "2026-10-13T15:00",
                      "cameras": ["cam1"]})
    frm, to = store.conn.execute(
        "SELECT window_from, window_to FROM search ORDER BY id DESC LIMIT 1").fetchone()
    assert frm is not None and to is not None
    assert frm.tzinfo is not None, "window stored without a timezone"
    assert frm < to


def test_a_naive_form_time_is_read_in_the_configured_timezone(client, store, cfg):
    """datetime-local sends no offset. Interpreting it as UTC means a reviewer
    in UTC+7 entering 14:00 silently searches 21:00 their time."""
    import dataclasses
    tz_cfg = dataclasses.replace(cfg, tz="+07:00", data_dir=cfg.data_dir)
    app.dependency_overrides[get_config] = lambda: tz_cfg

    client.post("/ui/searches", files=photo(),
                data={"from": "2026-10-13T14:00", "to": "2026-10-13T15:00",
                      "cameras": ["cam1"]})
    frm, _ = store.conn.execute(
        "SELECT window_from, window_to FROM search ORDER BY id DESC LIMIT 1").fetchone()
    assert frm.utcoffset().total_seconds() != 0 or frm.hour == 7, \
        f"14:00 at +07:00 should be 07:00Z, got {frm.isoformat()}"


def test_times_are_displayed_in_the_configured_timezone(client, store, cfg):
    """Ingesting at 09:00+07:00 stores 02:00Z. A reviewer in +07:00 should
    read 09:00 on the card, not the raw UTC instant."""
    import dataclasses
    tz_cfg = dataclasses.replace(cfg, tz="+07:00", data_dir=cfg.data_dir)
    app.dependency_overrides[get_config] = lambda: tz_cfg

    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=datetime(2026, 10, 19, 2, 0, 15, tzinfo=timezone.utc),
        ended_at=datetime(2026, 10, 19, 2, 0, 20, tzinfo=timezone.utc),
        thumb_path="thumbs/1.jpg", clip_path="clips/1.mp4"))

    sid = client.post("/api/searches", files=photo(),
                      data={"cameras": ["cam1"]}).json()["search_id"]
    r = client.get(f"/ui/searches/{sid}/results")
    assert "09:00:15" in r.text, "card should show local time"
    assert "02:00:15" not in r.text, "card still shows the raw UTC instant"
