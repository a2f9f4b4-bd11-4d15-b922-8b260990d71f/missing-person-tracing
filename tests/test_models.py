from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.models import Candidate, Match, TrackletRecord, unit_vector
from tracer.fixtures import make_similar_embedding, make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def test_fixture_builds_a_valid_record():
    rec = make_tracklet()
    assert rec.embedding.shape == (512,)
    assert rec.embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(rec.embedding), 1.0, atol=1e-5)


def test_fixture_accepts_overrides():
    rec = make_tracklet(camera_id="cam2", n_crops=3)
    assert rec.camera_id == "cam2"
    assert rec.n_crops == 3


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        make_tracklet(started_at=datetime(2026, 10, 13, 14, 0, 0))


def test_end_before_start_is_rejected():
    with pytest.raises(ValueError, match="ended_at"):
        make_tracklet(started_at=T0, ended_at=T0 - timedelta(seconds=1))


def test_tracklets_shorter_than_one_second_are_rejected():
    with pytest.raises(ValueError, match="at least"):
        make_tracklet(started_at=T0, ended_at=T0 + timedelta(milliseconds=400))


def test_wrong_embedding_dimension_is_rejected():
    with pytest.raises(ValueError, match="512"):
        make_tracklet(embedding=np.ones(128, dtype=np.float32))


def test_unnormalised_embedding_is_rejected():
    with pytest.raises(ValueError, match="normalis"):
        make_tracklet(embedding=np.full(512, 0.5, dtype=np.float32))


def test_crop_count_must_be_between_one_and_ten():
    with pytest.raises(ValueError, match="n_crops"):
        make_tracklet(n_crops=0)
    with pytest.raises(ValueError, match="n_crops"):
        make_tracklet(n_crops=11)


def test_unit_vector_normalises():
    v = unit_vector(np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32))
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)
    assert v.dtype == np.float32


def test_unit_vector_rejects_a_zero_vector():
    # A degenerate crop, or a model in reduced precision, yields a zero vector;
    # normalising it emits NaN that would corrupt every later ranking.
    with pytest.raises(ValueError, match="zero"):
        unit_vector(np.zeros(512, dtype=np.float32))


def test_unit_vector_rejects_nan():
    v = np.ones(512, dtype=np.float32)
    v[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        unit_vector(v)


def test_candidate_and_match_carry_the_expected_fields():
    c = Candidate(tracklet_id=1, camera_id="cam1", started_at=T0,
                  ended_at=T0 + timedelta(seconds=5), score=0.9)
    assert c.duration_s == 5.0
    m = Match(tracklet_id=1, camera_id="cam1", started_at=T0,
              ended_at=T0 + timedelta(seconds=5), score=0.9,
              thumb_url="/media/thumbs/1.jpg", clip_url="/media/clips/1.mp4",
              status="pending")
    assert m.status == "pending"


def test_make_similar_embedding_hits_the_requested_cosine():
    base = make_tracklet().embedding
    for target in (0.95, 0.8, 0.6):
        v = make_similar_embedding(base, target)
        assert np.isclose(float(v @ base), target, atol=1e-3)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
