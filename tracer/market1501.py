"""Market-1501 as a source of labelled, genuinely cross-camera gallery data.

The dataset is crops rather than video, so it cannot flow through detection
and tracking — but crops are what the embedder consumes. Injecting them as
tracklets gives real identity labels through the real search path, which
single-subject own footage cannot provide.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from tracer.models import TrackletRecord
from tracer.reid import Crop

log = logging.getLogger(__name__)

# 0002_c1s1_000451_03.jpg -> person 2, camera 1
NAME_RE = re.compile(r"^(-?\d+)_c(\d+)s\d+_\d+_\d+\.jpg$")
DATASET_DIR = "Market-1501-v15.09.15"


def parse_name(filename: str) -> tuple[int, int]:
    m = NAME_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"unexpected Market-1501 filename: {filename}")
    return int(m.group(1)), int(m.group(2))


def list_images(root: str | Path, split: str) -> list[Path]:
    """All non-junk images in a split. Person id -1 is junk, 0 is a distractor
    the standard protocol excludes from queries."""
    base = Path(root)
    folder = base / DATASET_DIR / split
    if not folder.is_dir():
        folder = base / split
    out = []
    for p in sorted(folder.glob("*.jpg")):
        try:
            pid, _ = parse_name(p.name)
        except ValueError:
            continue
        if pid > 0:
            out.append(p)
    return out


def group_by_identity_camera(paths: list[Path]) -> dict[tuple[int, int], list[Path]]:
    groups: dict[tuple[int, int], list[Path]] = {}
    for p in paths:
        groups.setdefault(parse_name(p.name), []).append(p)
    return groups


def _load_crops(paths: list[Path], t: datetime) -> list[Crop]:
    crops = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            log.warning("could not read %s", p)
            continue
        crops.append(Crop(image=img[:, :, ::-1].copy(),
                          t=t + timedelta(milliseconds=100 * i)))
    return crops


def _relative_thumb(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def inject(store, embedder, paths: list[Path], base_time: datetime,
           spacing_s: float = 60.0, protected: bool = True,
           max_identities: int | None = None,
           data_dir: str | Path | None = None) -> dict[int, list[int]]:
    """Write one tracklet per (identity, camera) group.

    Groups are spaced `spacing_s` apart so the impossible-timing filter never
    rejects a true match purely because of synthetic timestamps. Thumbnails are
    stored relative to `data_dir`, the way the media route resolves them.
    """
    root = Path(data_dir).resolve() if data_dir else None
    groups = group_by_identity_camera(paths)
    if max_identities is not None:
        keep = sorted({pid for pid, _ in groups})[:max_identities]
        groups = {k: v for k, v in groups.items() if k[0] in keep}

    mapping: dict[int, list[int]] = {}
    for slot, ((pid, cam), group) in enumerate(sorted(groups.items())):
        start = base_time + timedelta(seconds=slot * spacing_s)
        crops = _load_crops(group, start)
        if not crops:
            continue
        try:
            embedding, n_crops = embedder.embed_tracklet(crops)
        except ValueError as exc:
            log.warning("skipping %s/%s: %s", pid, cam, exc)
            continue

        rec = TrackletRecord(
            camera_id=f"m{cam}",
            started_at=start,
            ended_at=start + timedelta(seconds=max(len(crops) * 0.1, 1.0)),
            embedding=embedding,
            thumb_path=_relative_thumb(group[0], root),
            clip_path="",
            n_crops=n_crops,
        )
        mapping.setdefault(pid, []).append(store.insert_tracklet(rec, protected=protected))

    log.info("injected %d identities as %d tracklets",
             len(mapping), sum(len(v) for v in mapping.values()))
    return mapping
