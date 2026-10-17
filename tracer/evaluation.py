"""Standard re-ID metrics and threshold tuning."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


def cmc_and_map(query_ids: Sequence[int], query_cams: Sequence[int],
                gallery_ids: Sequence[int], gallery_cams: Sequence[int],
                sims: np.ndarray) -> dict:
    """Rank-1, rank-5 and mAP under the standard Market-1501 protocol.

    Gallery images sharing both identity and camera with the query are
    excluded: matching a person to another shot from the same camera is not
    re-identification.
    """
    query_ids = np.asarray(query_ids)
    query_cams = np.asarray(query_cams)
    gallery_ids = np.asarray(gallery_ids)
    gallery_cams = np.asarray(gallery_cams)

    ranks, aps = [], []
    for i in range(len(query_ids)):
        order = np.argsort(-sims[i])
        keep = ~((gallery_ids[order] == query_ids[i])
                 & (gallery_cams[order] == query_cams[i]))
        hits = (gallery_ids[order] == query_ids[i])[keep]
        if not hits.any():
            continue

        ranks.append(int(np.argmax(hits)))

        positions = np.flatnonzero(hits)
        precisions = (np.arange(len(positions)) + 1) / (positions + 1)
        aps.append(float(precisions.mean()))

    n = len(ranks)
    if n == 0:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "n_queries": 0}
    ranks_arr = np.asarray(ranks)
    return {"rank1": float((ranks_arr < 1).mean()),
            "rank5": float((ranks_arr < 5).mean()),
            "mAP": float(np.mean(aps)),
            "n_queries": n}


def sweep_tau(store, cfg, queries: list[dict], taus: Sequence[float]) -> list[dict]:
    """For each candidate threshold, measure top-1, top-5 and how many wrong
    matches get through.

    `queries` entries are {"embedding": np.ndarray, "person": int,
    "truth": set[int]} where `truth` holds that person's tracklet ids.
    """
    from tracer.search import run_search

    cameras = sorted({r[0] for r in store.conn.execute(
        "SELECT DISTINCT camera_id FROM tracklet").fetchall()})
    rows = []
    for tau in taus:
        tweaked = type(cfg)(**{**cfg.__dict__, "tau": {**cfg.tau, "osnet": float(tau)}})
        top1 = top5 = false_pos = 0
        for q in queries:
            res = run_search(store, tweaked, q["embedding"], "osnet",
                             (None, None), cameras, limit=5)
            ids = [m.tracklet_id for m in res.matches]
            if ids and ids[0] in q["truth"]:
                top1 += 1
            if any(i in q["truth"] for i in ids):
                top5 += 1
            false_pos += sum(1 for i in ids if i not in q["truth"])
        n = max(len(queries), 1)
        rows.append({"tau": float(tau), "top1": top1 / n, "top5": top5 / n,
                     "false_positives": false_pos})
        log.info("tau=%.2f top1=%.3f top5=%.3f fp=%d",
                 tau, top1 / n, top5 / n, false_pos)
    return rows
