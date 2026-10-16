"""Delete footage past its retention window, logging what went.

Rows marked `protected` are never deleted: the labelled evaluation set has to
outlive the retention period or the ground-truth CSV points at nothing.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tracer.config import Config
from tracer.store import Store

log = logging.getLogger(__name__)
MEDIA_DIRS = ("clips", "thumbs")


def _unlink(path: Path) -> int:
    try:
        path.unlink()
        return 1
    except FileNotFoundError:
        return 0
    except OSError as exc:
        log.warning("could not delete %s: %s", path, exc)
        return 0


def sweep(store: Store, cfg: Config, now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=cfg.retention_days)
    root = Path(cfg.data_dir)

    expired = store.expired_tracklets(cutoff)
    files = sum(_unlink(root / rel)
                for row in expired
                for rel in (row["thumb_path"], row["clip_path"]) if rel)
    deleted = store.delete_tracklets([row["id"] for row in expired])

    # Orphans are only swept once they are themselves past the retention
    # window. Ingest writes a clip before inserting its row, and a truncated
    # database would otherwise make every file on disk an orphan.
    known = store.all_media_paths()
    orphans = 0
    cutoff_ts = cutoff.timestamp()
    for sub in MEDIA_DIRS:
        for path in (root / sub).glob("*"):
            if not path.is_file() or str(path.relative_to(root)) in known:
                continue
            if path.stat().st_mtime >= cutoff_ts:
                continue
            orphans += _unlink(path)

    photos = 0
    rows = store.conn.execute(
        "SELECT id, photo_path FROM search WHERE closed_at IS NOT NULL "
        "AND photo_path IS NOT NULL").fetchall()
    for search_id, rel in rows:
        if _unlink(root / rel):
            photos += 1
            store.conn.execute("UPDATE search SET photo_path = NULL WHERE id = %s",
                               (search_id,))

    counts = {"tracklets": deleted, "files": files, "orphans": orphans, "photos": photos}
    if any(counts.values()):
        log.info("retention sweep (cutoff %s): %s", cutoff.isoformat(), counts)
    return counts


async def retention_loop(store: Store, cfg: Config, interval_s: float = 3600.0) -> None:
    """Hourly sweep inside the app process: no extra container, no host cron.

    Sleeps before its first sweep, so a short-lived app start never deletes
    anything on the way up.
    """
    while True:
        await asyncio.sleep(interval_s)
        try:
            sweep(store, cfg)
        except Exception:
            log.exception("retention sweep failed; continuing")
