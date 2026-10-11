from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tracer.config import Config
from tracer.models import Candidate
from tracer.store import Store
from tracer.timing import filter_impossible

CANDIDATE_POOL = 50


@dataclass(frozen=True)
class SearchResult:
    matches: list[Candidate]
    reason: str | None = None
    best_score: float | None = None


def run_search(store: Store, cfg: Config, embedding: np.ndarray, space: str,
               window: tuple[datetime | None, datetime | None],
               cameras: list[str], limit: int = 10) -> SearchResult:
    """Candidates -> threshold -> impossible-timing filter -> top `limit`."""
    pool = store.candidates(embedding, space, window, cameras, limit=CANDIDATE_POOL)
    if not pool:
        return SearchResult([], reason="no_tracklets_in_scope")

    best = max(c.score for c in pool)
    tau = cfg.tau[space]
    above = [c for c in pool if c.score >= tau]
    if not above:
        return SearchResult([], reason="none_above_tau", best_score=best)

    kept = filter_impossible(above, cfg)
    if not kept:
        return SearchResult([], reason="all_rejected_by_timing", best_score=best)

    return SearchResult(kept[:limit], reason=None, best_score=best)
