"""Tier 1: Market-1501 benchmark for the embedder alone.

  python -m scripts.eval_market1501 --weights data/weights/osnet_x1_0_msmt17.pth
  python -m scripts.eval_market1501 --weights data/weights/osnet_ain_x1_0_msmt17.pth \
      --model-name osnet_ain_x1_0

These weights are MSMT17-trained, so this is a CROSS-DOMAIN evaluation.
Compare against the cross-domain table in the OSNet-AIN paper, not against
the same-domain ~94% rank-1 figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tracer.evaluation import cmc_and_map
from tracer.market1501 import list_images, parse_name
from tracer.reid import OsnetEmbedder


def embed_split(embedder: OsnetEmbedder, paths: list[Path], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch):
        images = []
        for p in paths[i:i + batch]:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img[:, :, ::-1].copy())
        if images:
            out.append(embedder.embed_images(images))
        print(f"  embedded {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/market1501")
    ap.add_argument("--weights", default="data/weights/osnet_x1_0_msmt17.pth")
    ap.add_argument("--model-name", default="osnet_x1_0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/eval/market1501_results.json")
    args = ap.parse_args()

    embedder = OsnetEmbedder(args.weights, device=args.device, model_name=args.model_name)
    query = list_images(args.root, "query")
    gallery = list_images(args.root, "bounding_box_test")
    print(f"query={len(query)} gallery={len(gallery)}")

    q_emb = embed_split(embedder, query)
    g_emb = embed_split(embedder, gallery)
    sims = q_emb @ g_emb.T

    q_meta = [parse_name(p.name) for p in query]
    g_meta = [parse_name(p.name) for p in gallery]
    result = cmc_and_map([m[0] for m in q_meta], [m[1] for m in q_meta],
                         [m[0] for m in g_meta], [m[1] for m in g_meta], sims)
    result.update(weights=Path(args.weights).name, model=args.model_name,
                  protocol="cross-domain (MSMT17-trained, evaluated on Market-1501)")

    print(f"rank-1 {result['rank1']:.4f}  rank-5 {result['rank5']:.4f}  "
          f"mAP {result['mAP']:.4f}  (n={result['n_queries']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else []
    existing = [r for r in existing if r.get("weights") != result["weights"]]
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
