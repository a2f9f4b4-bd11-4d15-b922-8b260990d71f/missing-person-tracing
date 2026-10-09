CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS camera (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('file', 'rtsp')),
  source_url  TEXT,
  location    TEXT
);

CREATE TABLE IF NOT EXISTS tracklet (
  id          BIGSERIAL PRIMARY KEY,
  camera_id   TEXT NOT NULL,
  started_at  TIMESTAMPTZ NOT NULL,
  ended_at    TIMESTAMPTZ NOT NULL,
  embedding   VECTOR(512) NOT NULL,
  clip_emb    VECTOR(512),
  thumb_path  TEXT NOT NULL,
  clip_path   TEXT NOT NULL,
  n_crops     INT NOT NULL,
  protected   BOOLEAN NOT NULL DEFAULT false,
  CHECK (ended_at >= started_at)
);

-- Defined for the growth story only. Queries use an exact scan: pgvector
-- applies WHERE filters after the index scan, which under-returns on narrow
-- time windows.
CREATE INDEX IF NOT EXISTS tracklet_embedding_hnsw
  ON tracklet USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS tracklet_camera_time
  ON tracklet (camera_id, started_at);

CREATE TABLE IF NOT EXISTS search (
  id          BIGSERIAL PRIMARY KEY,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  photo_path  TEXT,
  query_text  TEXT,
  embedding   VECTOR(512) NOT NULL,
  space       TEXT NOT NULL DEFAULT 'osnet' CHECK (space IN ('osnet', 'clip')),
  cameras     TEXT[] NOT NULL DEFAULT '{}',
  window_from TIMESTAMPTZ,
  window_to   TIMESTAMPTZ,
  closed_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS match (
  search_id   BIGINT NOT NULL REFERENCES search(id) ON DELETE CASCADE,
  tracklet_id BIGINT NOT NULL REFERENCES tracklet(id) ON DELETE CASCADE,
  score       REAL NOT NULL,
  status      TEXT NOT NULL DEFAULT 'pending'
              CHECK (status IN ('pending', 'confirmed', 'rejected')),
  PRIMARY KEY (search_id, tracklet_id)
);
