from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

EMBEDDING_DIM = 512
MIN_TRACKLET_S = 1.0
MAX_CROPS = 10


def unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(v)):
        raise ValueError("embedding must be finite; got NaN or inf")
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        raise ValueError("cannot normalise a zero vector")
    return (v / norm).astype(np.float32)


def _require_aware(name: str, t: datetime) -> None:
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class TrackletRecord:
    """One continuous sighting of one person on one camera.

    The only interface between the vision pipeline and the database.
    """
    camera_id: str
    started_at: datetime
    ended_at: datetime
    embedding: np.ndarray
    thumb_path: str
    clip_path: str
    n_crops: int

    def __post_init__(self) -> None:
        _require_aware("started_at", self.started_at)
        _require_aware("ended_at", self.ended_at)
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if self.duration_s < MIN_TRACKLET_S:
            raise ValueError(
                f"tracklet must last at least {MIN_TRACKLET_S}s, got {self.duration_s:.3f}s")

        emb = np.asarray(self.embedding)
        if emb.shape != (EMBEDDING_DIM,):
            raise ValueError(f"embedding must have shape ({EMBEDDING_DIM},), got {emb.shape}")
        if emb.dtype != np.float32:
            raise ValueError(f"embedding must be float32, got {emb.dtype}")
        if not np.all(np.isfinite(emb)):
            raise ValueError("embedding must be finite")
        if not np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5):
            raise ValueError("embedding must be L2-normalised to 1.0 +/- 1e-5")

        if not 1 <= self.n_crops <= MAX_CROPS:
            raise ValueError(f"n_crops must be between 1 and {MAX_CROPS}, got {self.n_crops}")

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class Candidate:
    tracklet_id: int
    camera_id: str
    started_at: datetime
    ended_at: datetime
    score: float

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class Match:
    tracklet_id: int
    camera_id: str
    started_at: datetime
    ended_at: datetime
    score: float
    thumb_url: str
    clip_url: str
    status: str
