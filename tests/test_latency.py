import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.config import CameraCfg, Config
from tracer.detect_track import Detection
from tracer.worker import Pipeline

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.78, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="rtsp")},
                  lost_track_s=2.0, min_tracklet_s=1.0, max_tracklet_s=60.0,
                  merge_gap_s=3.0, merge_sim=0.8)


class FakeEmbedder:
    def embed_tracklet(self, crops):
        from tracer.models import unit_vector
        return unit_vector(np.ones(512, dtype=np.float32)), min(len(crops), 10)


def live_source(n: int, fps: float = 10.0):
    """Frames carrying arrival times, as a live stream produces them: the last
    frame is now and earlier ones are in the past. Timestamping forward from
    now would date frames in the future and make latency negative."""
    end = datetime.now(timezone.utc)
    for i in range(n):
        t = end - timedelta(seconds=(n - 1 - i) / fps)
        yield t, np.full((120, 80, 3), i % 256, np.uint8)


def detector_present_for(frames: int):
    seen = {"n": 0}

    def detect(frame, t):
        seen["n"] += 1
        if seen["n"] > frames:
            return []
        return [Detection(track_id=1, xyxy=(5, 5, 55, 105), edge_cut=False,
                          crop=np.full((100, 50, 3), 128, dtype=np.uint8))]
    return detect


def test_latency_is_recorded_for_each_tracklet(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30),
                 live=True)
    p.run(live_source(60))
    assert len(p.latencies) == 1
    assert p.latencies[0] >= 0.0


def test_latency_meets_the_ten_second_requirement(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30),
                 live=True)
    p.run(live_source(60))
    assert max(p.latencies) < 10.0


def test_a_metrics_line_is_written_per_tracklet(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30),
                 live=True)
    p.run(live_source(60))
    lines = (cfg.data_dir / "metrics" / "latency.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["camera_id"] == "cam1"
    assert row["tracklet_id"] > 0
    assert "last_seen" in row and "searchable_at" in row
    assert row["latency_s"] == pytest.approx(p.latencies[0], abs=0.01)


def test_a_recorded_file_ingest_writes_no_latency_metric(store, cfg):
    """R8 is about live ingest. A recorded file carries fabricated wall-clock
    timestamps, so its "latency" is the age of the footage — meaningless, and
    it would make report_latency.py declare R8 failed on a healthy system."""
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    p.run(live_source(60))
    assert p.latencies == []
    assert not (cfg.data_dir / "metrics" / "latency.jsonl").exists()


def test_live_rows_record_that_they_are_live(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30),
                 live=True)
    p.run(live_source(60))
    row = json.loads((cfg.data_dir / "metrics" / "latency.jsonl").read_text().strip())
    assert row["live"] is True
