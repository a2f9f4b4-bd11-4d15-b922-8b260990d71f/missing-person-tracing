from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from tracer.models import Candidate, Match, TrackletRecord

VALID_STATUS = ("pending", "confirmed", "rejected")
_EMB_COLUMN = {"osnet": "embedding", "clip": "clip_emb"}


class Store:
    def __init__(self, dsn: str) -> None:
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def close(self) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        sql = Path(__file__).resolve().parent.parent / "sql" / "001_schema.sql"
        self.conn.execute(sql.read_text())

    def reset(self) -> None:
        self.conn.execute("TRUNCATE match, search, tracklet RESTART IDENTITY CASCADE")

    def insert_tracklet(self, rec: TrackletRecord, protected: bool = False) -> int:
        row = self.conn.execute(
            """INSERT INTO tracklet (camera_id, started_at, ended_at, embedding,
                                     thumb_path, clip_path, n_crops, protected)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (rec.camera_id, rec.started_at, rec.ended_at, rec.embedding,
             rec.thumb_path, rec.clip_path, rec.n_crops, protected),
        ).fetchone()
        return int(row[0])

    def get_tracklet(self, tracklet_id: int) -> dict:
        row = self.conn.execute(
            """SELECT id, camera_id, started_at, ended_at, embedding,
                      thumb_path, clip_path, n_crops, protected
               FROM tracklet WHERE id = %s""",
            (tracklet_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no tracklet {tracklet_id}")
        keys = ("id", "camera_id", "started_at", "ended_at", "embedding",
                "thumb_path", "clip_path", "n_crops", "protected")
        return dict(zip(keys, row))

    def update_tracklet(self, tracklet_id: int, rec: TrackletRecord, n_crops: int) -> None:
        self.conn.execute(
            """UPDATE tracklet SET started_at = %s, ended_at = %s, embedding = %s,
                                   n_crops = %s
               WHERE id = %s""",
            (rec.started_at, rec.ended_at, rec.embedding, n_crops, tracklet_id),
        )

    def set_clip_embedding(self, tracklet_id: int, embedding: np.ndarray) -> None:
        self.conn.execute("UPDATE tracklet SET clip_emb = %s WHERE id = %s",
                          (embedding, tracklet_id))

    def candidates(self, embedding: np.ndarray, space: str,
                   window: tuple[datetime | None, datetime | None],
                   cameras: list[str], limit: int = 50) -> list[Candidate]:
        """Exact cosine scan with filters.

        Deliberately not using the HNSW index: pgvector post-filters, so an
        indexed LIMIT can return far fewer than `limit` true candidates on a
        narrow time window.
        """
        col = _EMB_COLUMN[space]
        frm, to = window
        rows = self.conn.execute(
            f"""SELECT id, camera_id, started_at, ended_at,
                       1 - ({col} <=> %s) AS sim
                FROM tracklet
                WHERE {col} IS NOT NULL
                  AND (%s::timestamptz IS NULL OR started_at >= %s)
                  AND (%s::timestamptz IS NULL OR started_at <= %s)
                  AND camera_id = ANY(%s)
                ORDER BY {col} <=> %s
                LIMIT %s""",
            (embedding, frm, frm, to, to, list(cameras), embedding, limit),
        ).fetchall()
        return [Candidate(tracklet_id=r[0], camera_id=r[1], started_at=r[2],
                          ended_at=r[3], score=float(r[4])) for r in rows]

    def create_search(self, photo_path: str | None, query_text: str | None,
                      embedding: np.ndarray, space: str,
                      window: tuple[datetime | None, datetime | None],
                      cameras: list[str] | None = None) -> int:
        row = self.conn.execute(
            """INSERT INTO search (photo_path, query_text, embedding, space,
                                   cameras, window_from, window_to)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (photo_path, query_text, embedding, space,
             list(cameras or []), window[0], window[1]),
        ).fetchone()
        return int(row[0])

    def record_matches(self, search_id: int, scored: list[tuple[int, float]]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO match (search_id, tracklet_id, score)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (search_id, tracklet_id) DO UPDATE SET score = EXCLUDED.score""",
                [(search_id, tid, float(score)) for tid, score in scored],
            )

    def set_match_status(self, search_id: int, tracklet_id: int, status: str) -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"status must be one of {VALID_STATUS}, got {status!r}")
        self.conn.execute(
            "UPDATE match SET status = %s WHERE search_id = %s AND tracklet_id = %s",
            (status, search_id, tracklet_id),
        )

    def _match_rows(self, search_id: int, exclude_rejected: bool, limit: int | None):
        sql = """SELECT m.tracklet_id, t.camera_id, t.started_at, t.ended_at,
                        m.score, t.thumb_path, t.clip_path, m.status
                 FROM match m JOIN tracklet t ON t.id = m.tracklet_id
                 WHERE m.search_id = %s"""
        if exclude_rejected:
            sql += " AND m.status <> 'rejected'"
        sql += " ORDER BY m.score DESC"
        params: tuple = (search_id,)
        if limit is not None:
            sql += " LIMIT %s"
            params = (search_id, limit)
        return self.conn.execute(sql, params).fetchall()

    @staticmethod
    def _to_match(r) -> Match:
        return Match(tracklet_id=r[0], camera_id=r[1], started_at=r[2], ended_at=r[3],
                     score=float(r[4]), thumb_url=f"/media/{r[5]}",
                     clip_url=f"/media/{r[6]}", status=r[7])

    def get_matches(self, search_id: int, limit: int = 10) -> list[Match]:
        return [self._to_match(r) for r in self._match_rows(search_id, False, limit)]

    def timeline(self, search_id: int) -> dict[str, list[Match]]:
        rows = self._match_rows(search_id, exclude_rejected=True, limit=None)
        lanes: dict[str, list[Match]] = {}
        for r in rows:
            lanes.setdefault(r[1], []).append(self._to_match(r))
        for cam in lanes:
            lanes[cam].sort(key=lambda m: m.started_at)
        return {cam: lanes[cam] for cam in sorted(lanes)}

    def expired_tracklets(self, cutoff: datetime) -> list[dict]:
        rows = self.conn.execute(
            """SELECT id, thumb_path, clip_path FROM tracklet
               WHERE ended_at < %s AND NOT protected""",
            (cutoff,),
        ).fetchall()
        return [{"id": r[0], "thumb_path": r[1], "clip_path": r[2]} for r in rows]

    def delete_tracklets(self, ids: list[int]) -> int:
        if not ids:
            return 0
        cur = self.conn.execute("DELETE FROM tracklet WHERE id = ANY(%s)", (ids,))
        return cur.rowcount

    def all_media_paths(self) -> set[str]:
        rows = self.conn.execute("SELECT thumb_path, clip_path FROM tracklet").fetchall()
        return {p for r in rows for p in r}
