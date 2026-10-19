"""Final acceptance: a volunteer walks past two cameras; both sightings appear
on the timeline within 10s and rank first.

The walk itself is physical, so this test asserts the machine-checkable parts
against whatever the live workers most recently wrote.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tracer.deps import get_config, get_store

METRICS = Path("data/metrics/latency.jsonl")
WITHIN_MINUTES = 30


def recent_live_rows(within_minutes: int = WITHIN_MINUTES) -> list[dict]:
    if not METRICS.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=within_minutes)
    rows = [json.loads(x) for x in METRICS.read_text().splitlines() if x.strip()]
    return [r for r in rows
            if r.get("live")
            and datetime.fromisoformat(r["searchable_at"]) >= cutoff]


pytestmark = pytest.mark.skipif(
    not recent_live_rows(),
    reason=f"no live sightings in the last {WITHIN_MINUTES} min; run a walk-through "
           "with 'docker compose up -d worker-cam1 worker-cam2'")


def test_both_cameras_produced_live_sightings():
    cameras = {r["camera_id"] for r in recent_live_rows()}
    assert len(cameras) >= 2, f"only {cameras} produced sightings"


def test_every_live_sighting_was_searchable_within_ten_seconds():
    rows = recent_live_rows()
    worst = max(r["latency_s"] for r in rows)
    assert worst <= 10.0, f"worst latency {worst:.2f}s exceeds the 10s requirement"


def test_two_cameras_reach_the_timeline():
    """The newest search must show both cameras as lanes, each in time order."""
    store = get_store()
    row = store.conn.execute(
        "SELECT id FROM search ORDER BY created_at DESC LIMIT 1").fetchone()
    assert row is not None, "no search has been run; use the UI first"
    lanes = store.timeline(row[0])
    assert len(lanes) >= 2, f"timeline has lanes {list(lanes)}; expected two cameras"
    for cam, sightings in lanes.items():
        times = [m.started_at for m in sightings]
        assert times == sorted(times), f"{cam} lane is not in time order"


def test_the_timing_check_held_on_the_live_route():
    store, cfg = get_store(), get_config()
    row = store.conn.execute(
        "SELECT id FROM search ORDER BY created_at DESC LIMIT 1").fetchone()
    lanes = store.timeline(row[0])
    flat = [m for ms in lanes.values() for m in ms]
    for a in flat:
        for b in flat:
            if a.camera_id != b.camera_id and a.started_at <= b.started_at:
                gap = (b.started_at - a.ended_at).total_seconds()
                assert gap >= cfg.min_walk(a.camera_id, b.camera_id), (
                    f"{a.camera_id}->{b.camera_id} gap {gap:.1f}s is impossible")
