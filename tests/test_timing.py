from datetime import datetime, timedelta, timezone

import pytest

from tracer.config import Config, CameraCfg, PairCfg
from tracer.models import Candidate
from tracer.timing import compatible, filter_impossible

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={c: CameraCfg(id=c, name=c, kind="rtsp") for c in ("cam1", "cam2", "cam3")},
        pairs={
            frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0),
            frozenset(("cam1", "cam3")): PairCfg(overlapping=True),
        },
    )


def cand(cam: str, start_s: float, dur_s: float = 5.0, score: float = 0.9) -> Candidate:
    return Candidate(tracklet_id=int(start_s * 1000) + hash(cam) % 1000, camera_id=cam,
                     started_at=T0 + timedelta(seconds=start_s),
                     ended_at=T0 + timedelta(seconds=start_s + dur_s), score=score)


def test_same_camera_is_always_compatible(cfg):
    assert compatible(cand("cam1", 0), cand("cam1", 1), cfg)


def test_gap_longer_than_the_walk_time_is_compatible(cfg):
    assert compatible(cand("cam1", 0), cand("cam2", 10), cfg)


def test_gap_shorter_than_the_walk_time_is_impossible(cfg):
    assert not compatible(cand("cam1", 0), cand("cam2", 6), cfg)


def test_gap_exactly_equal_to_the_walk_time_is_compatible(cfg):
    assert compatible(cand("cam1", 0), cand("cam2", 9), cfg)


def test_overlapping_intervals_on_different_cameras_are_impossible(cfg):
    assert not compatible(cand("cam1", 0, dur_s=10), cand("cam2", 5), cfg)


def test_argument_order_does_not_matter(cfg):
    a, b = cand("cam1", 0), cand("cam2", 6)
    assert compatible(a, b, cfg) == compatible(b, a, cfg)


def test_cameras_configured_as_overlapping_allow_simultaneous_sightings(cfg):
    assert compatible(cand("cam1", 0, dur_s=10), cand("cam3", 2), cfg)


def test_unconfigured_pairs_use_the_default_walk_time(cfg):
    assert not compatible(cand("cam2", 0), cand("cam3", 8), cfg)
    assert compatible(cand("cam2", 0), cand("cam3", 20), cfg)


def test_filter_keeps_the_highest_scorer_and_drops_what_conflicts_with_it(cfg):
    best = cand("cam1", 0, score=0.95)
    impossible = cand("cam2", 6, score=0.90)
    fine = cand("cam3", 60, score=0.80)
    kept = filter_impossible([best, impossible, fine], cfg)
    assert [c.tracklet_id for c in kept] == [best.tracklet_id, fine.tracklet_id]


def test_filter_output_is_ordered_by_score(cfg):
    kept = filter_impossible([cand("cam1", 0, score=0.7), cand("cam1", 30, score=0.9)], cfg)
    assert [c.score for c in kept] == [0.9, 0.7]


def test_filter_is_independent_of_input_order(cfg):
    cands = [cand("cam1", 0, score=0.95), cand("cam2", 6, score=0.90),
             cand("cam3", 60, score=0.80)]
    forward = [c.tracklet_id for c in filter_impossible(cands, cfg)]
    backward = [c.tracklet_id for c in filter_impossible(list(reversed(cands)), cfg)]
    assert forward == backward


def test_a_candidate_conflicting_only_with_a_dropped_one_survives(cfg):
    anchor = cand("cam1", 0, score=0.95)
    dropped = cand("cam2", 6, score=0.90)
    survivor = cand("cam2", 30, score=0.85)
    kept = filter_impossible([anchor, dropped, survivor], cfg)
    assert [c.tracklet_id for c in kept] == [anchor.tracklet_id, survivor.tracklet_id]


def test_empty_input_gives_empty_output(cfg):
    assert filter_impossible([], cfg) == []
