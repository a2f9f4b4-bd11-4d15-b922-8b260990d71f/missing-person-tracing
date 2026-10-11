"""Reject sightings that are physically impossible given camera geometry.

Two sightings on different cameras cannot both be the same person if the gap
between them is shorter than the time needed to walk between those cameras.
Gaps are measured end-to-start.
"""
from __future__ import annotations

from typing import Sequence

from tracer.config import Config
from tracer.models import Candidate


def compatible(a: Candidate, b: Candidate, cfg: Config) -> bool:
    if a.camera_id == b.camera_id:
        return True
    if cfg.overlapping(a.camera_id, b.camera_id):
        return True
    first, second = sorted((a, b), key=lambda c: c.started_at)
    gap = (second.started_at - first.ended_at).total_seconds()
    return gap >= cfg.min_walk(a.camera_id, b.camera_id)


def filter_impossible(cands: Sequence[Candidate], cfg: Config) -> list[Candidate]:
    """Greedy maximal compatible set, taken in descending score order.

    Deterministic regardless of input order: the highest-scoring sighting
    anchors the route, and each lower-scoring one is kept only if it is
    compatible with everything already kept.
    """
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (-c.score, c.tracklet_id)):
        if all(compatible(c, k, cfg) for k in kept):
            kept.append(c)
    return kept
