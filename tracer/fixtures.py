"""Valid random tracklets, so the search/API/UI lane needs no vision models."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from tracer.models import TrackletRecord, unit_vector

_T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
_counter = 0


def make_tracklet(**overrides) -> TrackletRecord:
    global _counter
    _counter += 1
    rng = np.random.default_rng(_counter)
    defaults = dict(
        camera_id="cam1",
        started_at=_T0 + timedelta(seconds=10 * _counter),
        ended_at=_T0 + timedelta(seconds=10 * _counter + 5),
        embedding=unit_vector(rng.standard_normal(512).astype(np.float32)),
        thumb_path=f"thumbs/{_counter}.jpg",
        clip_path=f"clips/{_counter}.mp4",
        n_crops=10,
    )
    defaults.update(overrides)
    return TrackletRecord(**defaults)


def make_similar_embedding(base: np.ndarray, similarity: float) -> np.ndarray:
    """An embedding with approximately `similarity` cosine to `base`."""
    base = unit_vector(base)
    rng = np.random.default_rng(abs(hash(float(similarity))) % (2**32))
    noise = rng.standard_normal(base.shape[0]).astype(np.float32)
    noise -= base * float(noise @ base)
    noise = unit_vector(noise)
    return unit_vector(similarity * base + float(np.sqrt(1 - similarity**2)) * noise)
