from datetime import datetime, timedelta, timezone

import pytest

from tracer.config import CameraCfg, Config
from tracer.fixtures import make_tracklet
from tracer.retention import sweep

NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.78, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file")})


def add(store, cfg, *, age_days: float, protected: bool = False) -> int:
    start = NOW - timedelta(days=age_days)
    rec = make_tracklet(camera_id="cam1", started_at=start,
                        ended_at=start + timedelta(seconds=5))
    tid = store.insert_tracklet(rec, protected=protected)
    for rel in (rec.thumb_path, rec.clip_path):
        p = cfg.data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return tid


def test_tracklets_past_the_retention_window_are_deleted(store, cfg):
    old = add(store, cfg, age_days=5)
    fresh = add(store, cfg, age_days=1)
    counts = sweep(store, cfg, now=NOW)
    assert counts["tracklets"] == 1
    remaining = [r[0] for r in store.conn.execute("SELECT id FROM tracklet").fetchall()]
    assert remaining == [fresh]
    assert old not in remaining


def test_protected_tracklets_are_never_deleted(store, cfg):
    kept = add(store, cfg, age_days=30, protected=True)
    counts = sweep(store, cfg, now=NOW)
    assert counts["tracklets"] == 0
    assert store.conn.execute("SELECT id FROM tracklet").fetchone()[0] == kept


def test_clips_and_thumbnails_are_removed_from_disk(store, cfg):
    add(store, cfg, age_days=5)
    paths = (list((cfg.data_dir / "clips").glob("*"))
             + list((cfg.data_dir / "thumbs").glob("*")))
    assert paths
    counts = sweep(store, cfg, now=NOW)
    assert counts["files"] == len(paths)
    assert not any(p.exists() for p in paths)


def test_reference_photos_of_closed_searches_are_deleted(store, cfg):
    photo = cfg.data_dir / "photos" / "ref.jpg"
    photo.parent.mkdir(parents=True, exist_ok=True)
    photo.write_bytes(b"x")
    sid = store.create_search("photos/ref.jpg", None, make_tracklet().embedding,
                              "osnet", (None, None))
    store.conn.execute("UPDATE search SET closed_at = %s WHERE id = %s", (NOW, sid))
    counts = sweep(store, cfg, now=NOW)
    assert counts["photos"] == 1
    assert not photo.exists()


def test_an_open_search_keeps_its_photo(store, cfg):
    photo = cfg.data_dir / "photos" / "open.jpg"
    photo.parent.mkdir(parents=True, exist_ok=True)
    photo.write_bytes(b"x")
    store.create_search("photos/open.jpg", None, make_tracklet().embedding,
                        "osnet", (None, None))
    counts = sweep(store, cfg, now=NOW)
    assert counts["photos"] == 0
    assert photo.exists()


def test_a_sweep_with_nothing_to_do_reports_zeros(store, cfg):
    assert sweep(store, cfg, now=NOW) == {"tracklets": 0, "files": 0,
                                          "orphans": 0, "photos": 0}


def test_the_loop_does_not_sweep_immediately(store, cfg):
    """A short-lived app start — a test client, a health probe, a crash loop —
    must not delete anything before its first interval elapses."""
    import asyncio

    from tracer.retention import retention_loop

    # Aged against real wall-clock, not the fixture's NOW: the loop sweeps
    # with now=None, so a row dated from the fixture's future is never expired
    # and the test would pass without proving anything.
    real_old = datetime.now(timezone.utc) - timedelta(days=10)
    rec = make_tracklet(camera_id="cam1", started_at=real_old,
                        ended_at=real_old + timedelta(seconds=5))
    store.insert_tracklet(rec)

    async def run():
        task = asyncio.create_task(retention_loop(store, cfg, interval_s=5))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert store.conn.execute("SELECT count(*) FROM tracklet").fetchone()[0] == 1


def test_recent_orphans_are_left_alone(store, cfg):
    """Ingest writes a clip before inserting its row. A sweep in that window
    must not delete the file, and a truncated database must not cost you every
    clip on disk."""
    fresh = cfg.data_dir / "clips" / "just-written.mp4"
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_bytes(b"x")

    counts = sweep(store, cfg, now=datetime.now(timezone.utc))
    assert counts["orphans"] == 0
    assert fresh.exists()


def test_stale_orphans_are_still_swept(store, cfg):
    import os
    stale = cfg.data_dir / "clips" / "abandoned.mp4"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x")
    old = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(stale, (old, old))

    counts = sweep(store, cfg, now=datetime.now(timezone.utc))
    assert counts["orphans"] == 1
    assert not stale.exists()
