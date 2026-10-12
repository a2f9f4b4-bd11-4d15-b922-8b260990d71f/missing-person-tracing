import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tracer.api import app
from tracer.config import CameraCfg, Config, PairCfg
from tracer.deps import get_config, get_embedder, get_store
from tracer.fixtures import make_similar_embedding, make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
QUERY = make_tracklet().embedding


class FakeEmbedder:
    """Stands in for YOLO + OSNet."""

    def __init__(self, detections=1):
        self.detections = detections

    def embed_image(self, data: bytes):
        if self.detections == 0:
            raise ValueError("no person detected in the reference photo")
        boxes = [{"xyxy": [0, 0, 100, 300 - 50 * i], "area": 100 * (300 - 50 * i)}
                 for i in range(self.detections)]
        return QUERY, boxes


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={c: CameraCfg(id=c, name=c, kind="rtsp") for c in ("cam1", "cam2")},
        pairs={frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0)},
    )


@pytest.fixture
def client(store, cfg):
    embedder = FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_config] = lambda: cfg
    app.dependency_overrides[get_embedder] = lambda: embedder
    c = TestClient(app)
    c.embedder = embedder
    yield c
    app.dependency_overrides.clear()


def photo() -> dict:
    return {"photo": ("ref.jpg", io.BytesIO(b"not-a-real-jpeg"), "image/jpeg")}


def form(**extra) -> dict:
    data = {"from": T0.isoformat(), "to": (T0 + timedelta(hours=1)).isoformat(),
            "cameras": ["cam1", "cam2"]}
    data.update(extra)
    return data


def test_listing_cameras_returns_the_configured_set(client):
    r = client.get("/api/cameras")
    assert r.status_code == 200
    assert {c["id"] for c in r.json()} == {"cam1", "cam2"}


def test_creating_a_search_returns_an_id_and_the_detected_person(client):
    r = client.post("/api/searches", files=photo(), data=form())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["search_id"] > 0
    assert len(body["detections"]) == 1
    assert body["selected"] == 0


def test_several_detected_people_are_ranked_by_area_with_the_largest_selected(client):
    client.embedder.detections = 3
    r = client.post("/api/searches", files=photo(), data=form())
    assert r.status_code == 201
    areas = [d["area"] for d in r.json()["detections"]]
    assert areas == sorted(areas, reverse=True)
    assert r.json()["selected"] == 0


def test_a_photo_with_no_person_is_a_422_with_a_readable_message(client):
    client.embedder.detections = 0
    r = client.post("/api/searches", files=photo(), data=form())
    assert r.status_code == 422
    assert "no person" in r.json()["detail"].lower()


def test_a_reversed_time_window_is_rejected(client):
    r = client.post("/api/searches", files=photo(),
                    data=form(**{"from": (T0 + timedelta(hours=1)).isoformat(),
                                 "to": T0.isoformat()}))
    assert r.status_code == 422
    assert "window" in r.json()["detail"].lower()


def test_an_empty_camera_list_is_rejected(client):
    r = client.post("/api/searches", files=photo(), data=form(cameras=[]))
    assert r.status_code == 422
    assert "camera" in r.json()["detail"].lower()


def test_an_unknown_camera_is_rejected(client):
    r = client.post("/api/searches", files=photo(), data=form(cameras=["cam9"]))
    assert r.status_code == 422
    assert "cam9" in r.json()["detail"]


def test_neither_photo_nor_text_is_rejected(client):
    r = client.post("/api/searches", data=form())
    assert r.status_code == 422


def test_matches_are_returned_with_urls_and_status(client, store):
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    r = client.get(f"/api/searches/{sid}/matches?limit=10")
    assert r.status_code == 200
    m = r.json()["matches"][0]
    assert m["tracklet_id"] == tid
    assert m["camera_id"] == "cam1"
    assert m["thumb_url"].startswith("/media/")
    assert m["clip_url"].startswith("/media/")
    assert m["status"] == "pending"
    assert 0.0 <= m["score"] <= 1.0


def test_an_empty_result_explains_itself(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.70),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    body = client.get(f"/api/searches/{sid}/matches").json()
    assert body["matches"] == []
    assert body["reason"] == "none_above_tau"
    assert body["best_score"] == pytest.approx(0.70, abs=1e-3)


def test_confirming_and_rejecting_a_match(client, store):
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    client.get(f"/api/searches/{sid}/matches")

    r = client.patch(f"/api/searches/{sid}/matches/{tid}", json={"status": "confirmed"})
    assert r.status_code == 200
    assert r.json()["status"] == "confirmed"

    r = client.patch(f"/api/searches/{sid}/matches/{tid}", json={"status": "nonsense"})
    assert r.status_code == 422


def test_the_timeline_groups_matches_into_one_lane_per_camera(client, store):
    a = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    b = store.insert_tracklet(make_tracklet(
        camera_id="cam2", embedding=make_similar_embedding(QUERY, 0.90),
        started_at=T0 + timedelta(seconds=40), ended_at=T0 + timedelta(seconds=45)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    client.get(f"/api/searches/{sid}/matches")

    lanes = client.get(f"/api/searches/{sid}/timeline").json()["lanes"]
    assert [m["tracklet_id"] for m in lanes["cam1"]] == [a]
    assert [m["tracklet_id"] for m in lanes["cam2"]] == [b]

    client.patch(f"/api/searches/{sid}/matches/{b}", json={"status": "rejected"})
    lanes = client.get(f"/api/searches/{sid}/timeline").json()["lanes"]
    assert "cam2" not in lanes


def test_the_camera_scope_chosen_at_search_time_is_honoured_later(client, store):
    """The matches endpoint re-runs the search, so the scope must be stored."""
    store.insert_tracklet(make_tracklet(
        camera_id="cam2", embedding=make_similar_embedding(QUERY, 0.99),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(),
                      data=form(cameras=["cam1"])).json()["search_id"]
    body = client.get(f"/api/searches/{sid}/matches").json()
    assert body["matches"] == []
    assert body["reason"] == "no_tracklets_in_scope"


def test_a_missing_search_is_a_404(client):
    assert client.get("/api/searches/99999/matches").status_code == 404
