"""MVP acceptance: a photo of person A returns A's sightings from both camera
spots at the top, in correct time order."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracer.config import CameraCfg, Config, PairCfg
from tracer.ingest import FileSource
from tracer.reid import OsnetEmbedder, PhotoEmbedder
from tracer.search import run_search
from tracer.worker import Pipeline, YoloDetector

CLIP = Path("data/samples/people.mp4")
WEIGHTS = Path("data/weights/osnet_x1_0_msmt17.pth")
T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
CAM2_OFFSET = timedelta(seconds=600)

pytestmark = pytest.mark.skipif(
    not (CLIP.exists() and WEIGHTS.exists()),
    reason="needs data/samples/people.mp4 and OSNet weights")


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path, device=os.environ.get("DEVICE", "cpu"),
        sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file"),
                 "cam2": CameraCfg(id="cam2", name="cam2", kind="file")},
        pairs={frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0)},
    )


@pytest.fixture
def ingested(store, cfg):
    """One clip as two cameras, ten minutes apart."""
    embedder = OsnetEmbedder(str(WEIGHTS), device=str(cfg.device))
    for camera, start in (("cam1", T0), ("cam2", T0 + CAM2_OFFSET)):
        Pipeline(cfg, camera, store, embedder, YoloDetector(cfg)).run(
            FileSource(CLIP, start, cfg.sample_fps))
    return store


def reference_photo(store, cfg) -> bytes:
    """The thumbnail of the longest cam1 tracklet, padded so the detector has
    a full frame to find a person in."""
    thumb = store.conn.execute(
        "SELECT thumb_path FROM tracklet WHERE camera_id = 'cam1' "
        "ORDER BY ended_at - started_at DESC LIMIT 1").fetchone()[0]
    img = cv2.imread(str(cfg.data_dir / thumb))
    canvas = np.full((img.shape[0] + 120, img.shape[1] + 120, 3), 60, dtype=np.uint8)
    canvas[60:60 + img.shape[0], 60:60 + img.shape[1]] = img
    ok, buf = cv2.imencode(".jpg", canvas)
    assert ok
    return buf.tobytes()


def test_both_cameras_produced_tracklets(ingested):
    rows = dict(ingested.conn.execute(
        "SELECT camera_id, count(*) FROM tracklet GROUP BY camera_id").fetchall())
    assert rows.get("cam1", 0) >= 1
    assert rows.get("cam2", 0) >= 1


def test_mvp_acceptance(ingested, cfg):
    embedding, detections = PhotoEmbedder(
        cfg, weights=str(WEIGHTS)).embed_image(reference_photo(ingested, cfg))
    assert detections, "the reference photo must contain a detectable person"

    result = run_search(ingested, cfg, embedding, "osnet",
                        (T0 - timedelta(hours=1), T0 + timedelta(hours=2)),
                        ["cam1", "cam2"], limit=10)

    assert result.matches, f"no matches: {result.reason} (best {result.best_score})"
    cameras = {m.camera_id for m in result.matches}
    assert cameras == {"cam1", "cam2"}, f"expected both cameras, got {cameras}"

    scores = [m.score for m in result.matches]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] >= cfg.tau["osnet"]


def test_results_can_be_placed_in_time_order_across_cameras(ingested, cfg):
    embedding, _ = PhotoEmbedder(
        cfg, weights=str(WEIGHTS)).embed_image(reference_photo(ingested, cfg))
    result = run_search(ingested, cfg, embedding, "osnet",
                        (T0 - timedelta(hours=1), T0 + timedelta(hours=2)),
                        ["cam1", "cam2"], limit=10)
    in_time_order = sorted(result.matches, key=lambda m: m.started_at)
    assert in_time_order[0].camera_id == "cam1"
    assert in_time_order[-1].camera_id == "cam2"


def test_the_timing_check_rejects_a_physically_impossible_pair(ingested, cfg):
    """With cam2 shifted to overlap cam1, the two sightings become impossible
    and the filter must drop one of them."""
    ingested.conn.execute(
        "UPDATE tracklet SET started_at = started_at - %s::interval, "
        "ended_at = ended_at - %s::interval WHERE camera_id = 'cam2'",
        ("600 seconds", "600 seconds"))
    embedding, _ = PhotoEmbedder(
        cfg, weights=str(WEIGHTS)).embed_image(reference_photo(ingested, cfg))
    result = run_search(ingested, cfg, embedding, "osnet",
                        (T0 - timedelta(hours=1), T0 + timedelta(hours=2)),
                        ["cam1", "cam2"], limit=10)
    for a in result.matches:
        for b in result.matches:
            if a.tracklet_id != b.tracklet_id and a.camera_id != b.camera_id:
                gap = abs((b.started_at - a.ended_at).total_seconds())
                assert gap >= cfg.min_walk(a.camera_id, b.camera_id)
