"""Tier 2: tune tau through the real search path, on identities DISJOINT from
the reported benchmark.

Market-1501's train identities (751) and test identities (750) do not overlap,
so tuning here and reporting on query/bounding_box_test keeps the benchmark an
unbiased estimate. Tuning on the test split would leak it.

  python -m scripts.tune_tau --identities 150
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tracer.deps import get_config, get_store
from tracer.evaluation import sweep_tau
from tracer.market1501 import (group_by_identity_camera, inject, list_images,
                               parse_name)
from tracer.reid import OsnetEmbedder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/market1501")
    ap.add_argument("--split", default="bounding_box_train",
                    help="tuning split; must be disjoint from the reported split")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--identities", type=int, default=150,
                    help="query identities; small samples make 100%% top-1 meaningless")
    ap.add_argument("--gallery-identities", type=int, default=400)
    ap.add_argument("--taus", default="0.50,0.60,0.65,0.70,0.74,0.78,0.82,0.86,0.90")
    ap.add_argument("--top1-slack", type=float, default=0.02,
                    help="top-1 given up for fewer false positives")
    ap.add_argument("--out", default="data/eval/tau_sweep.json")
    args = ap.parse_args()

    cfg, store = get_config(), get_store()
    embedder = OsnetEmbedder(args.weights or cfg.weights, device=str(cfg.device))

    paths = list_images(args.root, args.split)
    groups = group_by_identity_camera(paths)

    # Hold one camera per identity out of the gallery and use its crops as
    # queries, so every query is a genuine cross-camera match.
    by_person: dict[int, list[int]] = defaultdict(list)
    for pid, cam in groups:
        by_person[pid].append(cam)
    multi_cam = sorted(p for p, cams in by_person.items() if len(cams) >= 2)

    gallery_people = multi_cam[:args.gallery_identities]
    held_out = {p: sorted(by_person[p])[0] for p in gallery_people}

    gallery_paths = []
    for p in paths:
        pid, cam = parse_name(p.name)
        if pid in held_out and cam != held_out[pid]:
            gallery_paths.append(p)

    mapping = inject(store, embedder, gallery_paths,
                     base_time=datetime.fromisoformat("2026-01-01T00:00:00+00:00"))

    queries = []
    for pid in gallery_people:
        if len(queries) >= args.identities or pid not in mapping:
            continue
        candidates = groups.get((pid, held_out[pid]), [])
        if not candidates:
            continue
        img = cv2.imread(str(candidates[0]))
        if img is None:
            continue
        emb = embedder.embed_images([img[:, :, ::-1].copy()])[0]
        queries.append({"embedding": np.asarray(emb, dtype=np.float32),
                        "person": pid, "truth": set(mapping[pid])})

    gallery_size = sum(len(v) for v in mapping.values())
    print(f"\ntuning on {args.split}: {len(queries)} held-out queries against "
          f"{gallery_size} gallery tracklets from {len(mapping)} identities")

    rows = sweep_tau(store, cfg, queries, [float(t) for t in args.taus.split(",")])

    # Most selective threshold that does not meaningfully hurt recall: top-1
    # barely moves across a wide band while false positives change several-fold,
    # and each false positive costs the reviewer a clip to watch.
    best_top1 = max(r["top1"] for r in rows)
    acceptable = [r for r in rows if r["top1"] >= best_top1 - args.top1_slack]
    best = max(acceptable, key=lambda r: r["tau"])

    print(f"\n{'tau':<8}{'top1':<10}{'top5':<10}{'false positives':<16}")
    for r in rows:
        mark = "  <-- chosen" if r is best else ""
        print(f"{r['tau']:<8.2f}{r['top1']:<10.3f}{r['top5']:<10.3f}"
              f"{r['false_positives']:<16}{mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rows": rows, "best_tau": best["tau"], "top1_slack": args.top1_slack,
        "n_queries": len(queries), "gallery_tracklets": gallery_size,
        "gallery_identities": len(mapping), "tuning_split": args.split,
        "weights": Path(args.weights or cfg.weights).name,
        "note": "tuned on identities disjoint from the reported query/test split",
    }, indent=2))
    print(f"\nchose tau={best['tau']:.2f}: top-1 {best['top1']:.3f} with "
          f"{best['false_positives']} false positives (best top-1 {best_top1:.3f})")
    print(f"wrote {out}. Set tau.osnet in config.yaml to this value.")


if __name__ == "__main__":
    main()
