# Missing Person Tracing — Technical Spec

Oct 8, 2026 · @Thai

## Scope and assumptions

All AI runs on the desktop server; the Raspberry Pi 4 only captures and streams video. This spec implements requirements R1–R13 from the product spec.

- **Scale:** 2–3 cameras, a few hours of footage, a few thousand sightings. One machine, no clustering.
- **Models:** pretrained, used as-is first — YOLO26 for people, ByteTrack for tracking, OSNet for appearance. Fine-tuning only if accuracy on own footage is too low.
- **Cameras:** a USB webcam on the Pi 4 (camera 1) and a phone streaming RTSP (camera 2). Until the webcam arrives, phone-recorded clips stand in.
- **Time:** camera clocks are synced with NTP; each recorded file carries its start time.
- **Language:** Python 3.12 for everything.

## System architecture

&#91;embedded content: system architecture · 3 sources, 6 services\]

Video flows left to right: sources feed the ingest worker, each person becomes a tracklet with an embedding in Postgres, and the reviewer's photo is matched against those embeddings through the search API.

## Hardware and environment

| Machine | Role | Specs | Software |
| --- | --- | --- | --- |
| pepper (desktop) | Processing server, database, web UI | Ryzen 9 7900X, 32 GB RAM, RX 6800 XT, Arch Linux | Docker, Python 3.12, PyTorch (ROCm, CPU fallback), Postgres 16 + pgvector |
| Raspberry Pi 4 | Camera node 1 | USB 3 port for the webcam, Ethernet preferred | Raspberry Pi OS Lite 64-bit, MediaMTX, ffmpeg |
| USB webcam | Camera 1 | UVC, 1080p30, MJPEG, 78–90° field of view | — |
| Phone | Camera 2 | Any recent phone | RTSP streaming app (IP Webcam on Android, Larix on iOS) |

**Pi 4 notes:** official 3 A power supply; heatsink or fan for continuous encoding; the webcam goes in a blue USB 3 port.

**GPU note:** the RX 6800 XT is not on AMD's official ROCm list for consumer cards but generally works on Linux. Spend at most an hour on it before 11/10; the CPU runs OSNet fast enough at this scale.

## Components

### 1. Camera node (Pi 4)

Captures the webcam and publishes an H.264 RTSP stream using the Pi 4's hardware encoder. MediaMTX runs the ffmpeg command at boot.

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
  -i /dev/video0 -c:v h264_v4l2m2m -b:v 3M \
  -f rtsp rtsp://localhost:8554/cam1
```

If the webcam outputs H.264 itself, use `-c:v copy`. 720p is enough: re-ID works on small person crops.

### 2. Ingest worker (pepper)

One worker per source. A source is either a video file (MVP) or an RTSP URL (final). It reads frames with OpenCV, samples at 10 fps to save compute, and tags each frame with camera ID and wall-clock time (file start time + frame offset, or arrival time for live).

### 3. Detection and tracking

- **Detector:** Ultralytics YOLO26, class `person` only, confidence ≥ 0.5. Small model (`yolo26s`) on GPU, nano (`yolo26n`) on CPU.
- **Tracker:** ByteTrack, built into Ultralytics (`model.track(..., tracker='bytetrack.yaml', persist=True)`).
- **Tracklet:** all boxes with one track ID in one camera. A tracklet closes when its track is lost for 2 s. Tracklets shorter than 1 s are dropped as noise.
- **Clip:** for each closed tracklet, save the frames from start − 1 s to end + 1 s as an MP4 for review.

### 4. Re-identification

- **Model:** OSNet (`osnet_x1_0`, pretrained on MSMT17) via `torchreid`, input crops resized to 256 × 128.
- **Crop selection:** up to 10 crops per tracklet, spread evenly over time, skipping crops smaller than 64 px tall or cut off at the frame edge.
- **Tracklet embedding:** mean of the crop embeddings, L2-normalised, 512 dimensions.
- **Thumbnail:** the largest crop, saved as JPEG.

### 5. Search service

FastAPI app. Takes a reference photo, detects the person, embeds the crop the same way, runs a cosine nearest-neighbour query in pgvector with filters, applies the timing check, and returns ranked sightings (see Matching and scoring).

### 6. Web UI

Server-rendered pages (FastAPI + Jinja2 + HTMX). Pages: upload and scope, ranked results with confirm/reject, timeline with one lane per camera, and a sources page for adding files or RTSP URLs.

### 7. Retention job

A daily task deletes clips, thumbnails and tracklets older than the configured number of days, and reference photos of closed searches.

## Data model

Four tables in Postgres; clips, thumbnails and photos live on disk under `data/`, referenced by path.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE camera (
  id          TEXT PRIMARY KEY,          -- 'cam1', 'phone'
  name        TEXT NOT NULL,
  source_url  TEXT NOT NULL,             -- file path or rtsp:// URL
  location    TEXT                        -- 'front door', 'corridor'
);

CREATE TABLE tracklet (
  id          BIGSERIAL PRIMARY KEY,
  camera_id   TEXT NOT NULL REFERENCES camera(id),
  started_at  TIMESTAMPTZ NOT NULL,
  ended_at    TIMESTAMPTZ NOT NULL,
  embedding   VECTOR(512) NOT NULL,      -- OSNet, L2-normalised
  face_emb    VECTOR(128),               -- optional, R12
  thumb_path  TEXT NOT NULL,
  clip_path   TEXT NOT NULL,
  n_crops     INT NOT NULL
);
CREATE INDEX ON tracklet USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON tracklet (camera_id, started_at);

CREATE TABLE search (
  id          BIGSERIAL PRIMARY KEY,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  photo_path  TEXT,
  query_text  TEXT,                       -- R13
  embedding   VECTOR(512) NOT NULL,
  window_from TIMESTAMPTZ,
  window_to   TIMESTAMPTZ,
  closed_at   TIMESTAMPTZ
);

CREATE TABLE match (
  search_id   BIGINT REFERENCES search(id) ON DELETE CASCADE,
  tracklet_id BIGINT REFERENCES tracklet(id) ON DELETE CASCADE,
  score       REAL NOT NULL,
  status      TEXT NOT NULL DEFAULT 'pending',  -- pending | confirmed | rejected
  PRIMARY KEY (search_id, tracklet_id)
);
```

## API

JSON endpoints under `/api`; the web UI calls the same endpoints.

| Method | Path | Purpose | Release |
| --- | --- | --- | --- |
| POST | `/api/cameras` | Add a camera: id, name, source URL or file, start time for files | MVP |
| GET | `/api/cameras` | List cameras with ingest status | MVP |
| POST | `/api/searches` | Multipart: photo (or `text`), `from`, `to`, `cameras[]`. Returns search id + detected person crop | MVP |
| GET | `/api/searches/{id}/matches?limit=10` | Ranked matches: tracklet id, camera, times, score, thumb and clip URLs | MVP |
| PATCH | `/api/searches/{id}/matches/{tracklet_id}` | Set status to `confirmed` or `rejected` | Final |
| GET | `/api/searches/{id}/timeline` | Confirmed and top matches grouped by camera, time-ordered | Final |
| GET | `/api/searches/{id}/events` | Server-sent events: new matches during a live search | Final, stretch |
| GET | `/media/{path}` | Thumbnails and clips | MVP |

Example match:

```json
{
  "tracklet_id": 412,
  "camera_id": "cam1",
  "started_at": "2026-10-13T14:05:12+07:00",
  "ended_at": "2026-10-13T14:05:19+07:00",
  "score": 0.81,
  "thumb_url": "/media/thumbs/412.jpg",
  "clip_url": "/media/clips/412.mp4",
  "status": "pending"
}
```

## Matching and scoring

A sighting's score is its appearance similarity to the photo, optionally blended with face similarity; impossible timings are then removed.

1. **Candidates:** pgvector cosine search, top 50, filtered by time window and cameras.

   ```sql
   SELECT id, 1 - (embedding <=> :q) AS sim
   FROM tracklet
   WHERE started_at BETWEEN :from AND :to AND camera_id = ANY(:cams)
   ORDER BY embedding <=> :q LIMIT 50;
   ```
2. **Threshold:** drop candidates with appearance similarity below τ. Start with τ = 0.5 and tune on own footage on 15/10.
3. **Face blend (R12, when both sides have a face):**

```latex
s = 0.7 \cdot s_{\text{body}} + 0.3 \cdot s_{\text{face}}
```

4. **Timing check:** sort candidates by time. Keep the best-scoring sighting as an anchor. For each other sighting on a different camera, drop it if the gap to an already-kept sighting is shorter than the minimum walk time between those cameras (set per camera pair in config; default 10 s). Two sightings at the same time on different cameras cannot both be the person.
5. **Rank:** remaining sightings by score, top 10 returned. The timeline view shows the same set in time order.

Reviewer feedback (confirm/reject) changes only the displayed route in this version; it does not retrain anything.

## Repository layout and deployment

```
tracer/
├── docker-compose.yml        # db (pgvector), app, worker
├── config.yaml               # cameras, walk times, τ, retention days
├── tracer/
│   ├── ingest.py             # file / RTSP reader, frame sampling, timestamps
│   ├── detect_track.py       # YOLO26 + ByteTrack → tracklets, clips
│   ├── reid.py               # OSNet crop selection + embeddings
│   ├── face.py               # YuNet + SFace (R12)
│   ├── store.py              # Postgres / pgvector access
│   ├── search.py             # candidates, threshold, timing check
│   ├── api.py                # FastAPI routes
│   ├── web/                  # Jinja2 templates, HTMX
│   └── retention.py
├── scripts/
│   ├── ingest_file.py        # CLI: python -m scripts.ingest_file clip.mp4 --camera cam1 --start ...
│   ├── search_cli.py         # CLI search for the MVP
│   └── eval_market1501.py
├── pi/
│   ├── mediamtx.yml          # runOnInit ffmpeg command
│   └── setup.sh
├── tests/
└── data/                     # clips, thumbs, photos (git-ignored)
```

**Deployment**

- **pepper:** `docker compose up` starts Postgres, the API/UI on port 8000 and one worker per camera from `config.yaml`. The worker container gets `/dev/kfd` and `/dev/dri` for ROCm; set `DEVICE=cpu` to skip the GPU.
- **Pi 4:** `pi/setup.sh` installs MediaMTX and ffmpeg, copies `mediamtx.yml`, and enables the systemd service so the stream starts at boot.
- **Phone:** start the streaming app and add its RTSP URL as a camera.

## Evaluation plan

Two measurements: a public benchmark for the re-ID model alone, and end-to-end search on own footage.

**Benchmark (Market-1501).** Run `scripts/eval_market1501.py` with the pretrained OSNet; report rank-1, rank-5 and mAP. This checks the model works as published; it says nothing about own cameras.

**Own footage.**

- 3–4 volunteers, 2 camera spots, about 10 clips: different walking orders, one clip each with a jacket on or off, one in dimmer light.
- Label ground truth by hand: which tracklets belong to which person (a small CSV: `tracklet_id, person`).
- For each person, run searches with 2 reference photos taken separately from the clips.
- Report top-1 and top-5 accuracy, and how many wrong matches pass τ.
- Repeat after tuning τ and the timing check; report both rows.

**Live (final).** Time from a volunteer leaving the frame to the sighting appearing in search; target ≤ 10 s.

| Run | Rank-1 / Top-1 | Top-5 | mAP | Notes |
| --- | --- | --- | --- | --- |
| Market-1501, OSNet pretrained |  |  |  |  |
| Own footage, τ = 0.5 |  |  | — |  |
| Own footage, tuned |  |  | — |  |

## Implementation plan

Four days to the MVP and seven more to the final, starting Thursday 8 October. Tick tasks off as they land.

**MVP — recorded clips, search, minimal UI (due 11/10)**

- [ ] Thu 8/10 — Repo skeleton, `docker-compose.yml` with pgvector, Python env with `ultralytics` and `torchreid`; YOLO26 on a sample video (ROCm: 1 hour max, else CPU)
- [ ] Thu 8/10 — Order the webcam
- [ ] Fri 9/10 — Record test clips on the phone: 2 spots, 3–4 volunteers
- [ ] Fri 9/10 — `ingest.py` + `detect_track.py`: file → tracklets, thumbnails, clips
- [ ] Sat 10/10 — `reid.py`: tracklet embeddings; `store.py`: write to Postgres
- [ ] Sat 10/10 — `search.py` + `search_cli.py`: photo → top 10, time filter, timing check
- [ ] Sun 11/10 — Minimal web UI: upload, results with thumbnails and clips
- [ ] Sun 11/10 — Market-1501 baseline; MVP acceptance check; record demo; flash the Pi

**Final — live cameras, timeline (due 18/10)**

- [ ] Mon 12/10 — Pi 4: MediaMTX + ffmpeg with the webcam; stream visible on pepper; phone as camera 2
- [ ] Tue 13/10 — Live ingest: RTSP workers, tracklets close and become searchable within 10 s
- [ ] Wed 14/10 — Timeline view; confirm/reject
- [ ] Thu 15/10 — Live walk-throughs; tune τ, tracklet length and walk times
- [ ] Fri 16/10 — Evaluation on own footage (ground-truth CSV, table filled); retention job
- [ ] Sat 17/10 — One stretch feature (face or text search); Docker Compose for everything; README / report
- [ ] Sun 18/10 — Buffer; final acceptance check; demo video

If behind on Wednesday 14/10, drop the stretch feature first, then own-footage tuning; never the live demo or timeline.

## Technical risks and fallbacks

| Risk | Sign | Fallback |
| --- | --- | --- |
| ROCm does not work on the RX 6800 XT | PyTorch sees no GPU after an hour | Run on CPU: `yolo26n`, 5 fps sampling; enough for 2 cameras |
| Re-ID confuses similar clothing | Top-1 < 80% on own footage | More crops per tracklet; tune τ; add face blend; try CLIP-ReID |
| Tracker splits one walk into many tracklets | Several short sightings per pass | Raise lost-track timeout; merge same-camera tracklets with gap < 3 s and high similarity |
| Pi 4 stream stutters | Dropped frames, RTSP reconnects | Ethernet; 720p; `-c:v copy` if the webcam outputs H.264; heatsink |
| Phone stream drops | Camera 2 offline | Use recorded phone clips for camera 2 in the demo |
| Clock drift between cameras | Timing check rejects true matches | NTP on the Pi; phone uses network time; widen walk-time margin |
