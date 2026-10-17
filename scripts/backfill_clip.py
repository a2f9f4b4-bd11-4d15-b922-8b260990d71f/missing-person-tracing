"""Fill clip_emb for existing tracklets, using each tracklet's thumbnail.

  python -m scripts.backfill_clip
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tracer.deps import get_config, get_store
from tracer.textsearch import ClipEncoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--all", action="store_true",
                    help="re-embed rows that already have one")
    args = ap.parse_args()

    cfg, store = get_config(), get_store()
    where = "" if args.all else "WHERE clip_emb IS NULL"
    rows = store.conn.execute(
        f"SELECT id, thumb_path FROM tracklet {where} ORDER BY id").fetchall()
    print(f"{len(rows)} tracklet(s) to embed")
    if not rows:
        return

    enc = ClipEncoder(device=str(cfg.device))
    done = skipped = 0
    for i in range(0, len(rows), args.batch):
        ids, images = [], []
        for tid, thumb in rows[i:i + args.batch]:
            path = Path(thumb)
            if not path.is_absolute():
                path = Path(cfg.data_dir) / thumb
            img = cv2.imread(str(path))
            if img is None:
                skipped += 1
                continue
            ids.append(tid)
            images.append(img[:, :, ::-1].copy())
        if not images:
            continue
        for tid, vec in zip(ids, enc.encode_images(images)):
            store.set_clip_embedding(tid, vec)
            done += 1
        print(f"  {done}/{len(rows)}", end="\r")
    print(f"\nembedded {done} tracklet(s)" + (f", skipped {skipped} unreadable" if skipped else ""))


if __name__ == "__main__":
    main()
