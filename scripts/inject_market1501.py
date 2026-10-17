"""Inject Market-1501 identities as tracklets, for a labelled gallery.

  python -m scripts.inject_market1501 --root data/market1501 --identities 200
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from tracer.deps import get_config, get_store
from tracer.market1501 import inject, list_images
from tracer.reid import OsnetEmbedder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/market1501")
    ap.add_argument("--split", default="bounding_box_test")
    ap.add_argument("--identities", type=int, default=200)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--base-time", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--out", default="data/eval/market_injection.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg, store = get_config(), get_store()
    paths = list_images(args.root, args.split)
    print(f"{len(paths)} images in {args.split}")

    mapping = inject(store, OsnetEmbedder(args.weights or cfg.weights, device=str(cfg.device)), paths,
                     base_time=datetime.fromisoformat(args.base_time),
                     max_identities=args.identities, data_dir=cfg.data_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2))
    print(f"injected {len(mapping)} identities, "
          f"{sum(len(v) for v in mapping.values())} tracklets -> {out}")


if __name__ == "__main__":
    main()
