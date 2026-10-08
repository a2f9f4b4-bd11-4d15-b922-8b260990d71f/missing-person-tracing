# Missing Person Tracing Implementation Plan

**Goal:** From one reference photo, find and rank a person's sightings across two or more cameras, on recorded files and on live streams, shown as a per-camera timeline with clips.

**Architecture:** Ingest workers (one OS process per camera) turn video into *tracklets* — one continuous sighting with a 512-d OSNet appearance embedding — written to Postgres/pgvector. A FastAPI app embeds a reference photo through the identical crop pipeline, runs an exact cosine scan with time/camera filters, applies a similarity threshold and a physical-timing check, and serves ranked matches plus a timeline over a Jinja2/HTMX UI. The `TrackletRecord` dataclass is the only interface between the vision half and the database half, so both halves are built in parallel.

**Tech Stack:** Python 3.12, PyTorch 2.9.1 (ROCm 6.4 / CPU), Ultralytics 8.4.176 (YOLO26 + ByteTrack), torchreid (OSNet x1_0, MSMT17 weights), Postgres 16 + pgvector, psycopg 3, FastAPI, Jinja2, HTMX, MediaMTX, ffmpeg, Docker Compose, pytest.

**Spec:** `docs/design.md`

## Global Constraints

- **Python 3.12 only.** The host has 3.14; PyTorch supports ≤3.13. All code runs in containers.
- **ROCm wheels:** `--index-url https://download.pytorch.org/whl/rocm6.4`. CPU wheels: `--index-url https://download.pytorch.org/whl/cpu`.
- **GPU containers require:** `--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined` and `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- **Pin `lap>=0.5.12`** in the image. Ultralytics otherwise AutoUpdates it at runtime.
- **`torchreid` installs only as:** `build-essential` present, these 16 packages preinstalled — `numpy Cython setuptools scipy h5py Pillow six opencv-python-headless matplotlib gdown yacs imageio chardet future tabulate tensorboard` — then `pip install --no-build-isolation --no-deps git+https://github.com/KaiyangZhou/deep-person-reid.git`.
- **OSNet weights** (never `pretrained=True`, which gives ImageNet weights): `https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`
- **AIN variant for the A/B:** same repo, `osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth`
- **Embeddings:** float32, shape `(512,)`, L2-normalised to `1.0 ± 1e-5`.
- **Thresholds:** `tau.osnet = 0.78`, `tau.clip = 0.25`. Never 0.5 — measured to admit 21/21 different-person pairs.
- **Crop rules:** up to 10 crops per tracklet, evenly spread in time, skip crops under 64 px tall or touching a frame edge. OSNet input 256×128.
- **Tracklets:** close after 2.0 s of lost track; drop if shorter than 1.0 s; force-close at 60.0 s; merge same-camera tracklets with gap < 3.0 s and cosine ≥ 0.8.
- **Retention:** 3 days default; rows with `protected = true` are never deleted.
- **Sampling:** 10 fps default, `imgsz` configurable; 5 fps is the documented fallback.
- **Vector queries use an exact scan**, not the HNSW index — pgvector post-filters, which under-returns on narrow time windows. The index stays defined.
- **All timestamps are timezone-aware UTC.** Naive datetimes are a hard error, never coerced.
- **Tests run in the container:** `docker compose run --rm app pytest -v`.

## Requirement Traceability

Every requirement from the product spec, and the task that implements it.

| ID | Requirement | Task(s) |
|---|---|---|
| R1 | Import recorded files tagged with camera and start time | 13, 14 |
| R2 | Detect and track every person; save each sighting with a thumbnail | 11, 12, 14 |
| R3 | Reference photo returns the top 10 matches, ranked | 6, 15 |
| R4 | Each result shows camera, time, score, thumbnail, clip link | 7, 8 |
| R5 | Filter results by time window and camera | 4, 6, 7 |
| R6 | Drop matches impossible in time | 5, 16 |
| R7 | Live video from the Pi webcam and a phone at the same time | 17, 18 |
| R8 | New sightings searchable within 10 s | 18, 24 |
| R9 | Timeline view, one lane per camera, click to play | 8 |
| R10 | Reviewer can confirm or reject; rejected matches leave the timeline | 7, 8 |
| R11 | Footage and sightings auto-delete after N days | 19 |
| R12 | Face matching as an extra score | **Out of scope** — R13 was chosen instead (design decision 6) |
| R13 | Search by text description | 22 |

## Review Focus

These five input classes are implied by the spec but easy to leave untested. Each has a test assigned to the task that owns the code.

1. **Reference photo containing no person, or several people** — a reviewer uploads a crowd photo or a landscape. Expected: 422 with a readable message for none; all detections ranked by box area with the largest pre-selected for several. *(Tasks 7, 15)*
2. **Naive (timezone-less) timestamps from a camera or CLI** — clock skew and naive datetimes silently shift every gap and make the timing check reject true matches. Expected: rejected loudly at the boundary, never coerced. *(Tasks 3, 13)*
3. **A degenerate crop producing a zero or NaN embedding** — a fully black frame or a 1-pixel box yields a zero vector, and L2 normalisation then emits NaN that propagates into pgvector and silently corrupts every ranking. Expected: detected and skipped with a logged warning. *(Task 10)*
4. **A reversed or empty search scope** — `from` after `to`, or an empty camera list. Expected: an empty result with a stated reason, not a crash and not a full-table scan. *(Task 7)*
5. **Path traversal through `/media/{path}`** — `../../etc/passwd` served by the media route. Expected: refused with 404, resolved strictly inside `data/`. *(Task 8)*

---

## File Structure

| File | Responsibility |
|---|---|
| `Dockerfile` | Python 3.12 base, torch (ROCm or CPU via build arg), ultralytics, torchreid, app deps |
| `docker-compose.yml` | `db` (pgvector), `app` (API/UI/retention), `mediamtx`, `worker-*` per camera |
| `config.yaml` | cameras, camera-pair walk times and overlap flags, τ per space, retention, sampling |
| `sql/001_schema.sql` | Tables, indexes, constraints |
| `tracer/config.py` | Config loading and `min_walk()` lookup |
| `tracer/models.py` | `TrackletRecord`, `Candidate`, `Match`, invariants |
| `tracer/fixtures.py` | `make_tracklet()` — valid random records so Lane 2 needs no models |
| `tracer/store.py` | All Postgres/pgvector access |
| `tracer/timing.py` | `compatible()` and the greedy impossible-timing filter (pure logic) |
| `tracer/search.py` | Candidates → τ → timing → top-N, with empty-result reasons |
| `tracer/reid.py` | Crop selection and the OSNet embedder |
| `tracer/detect_track.py` | Detector protocol, `TrackletBuilder`, same-camera merging |
| `tracer/clips.py` | Rolling frame buffer, clip and thumbnail writing |
| `tracer/ingest.py` | File and RTSP frame sources, sampling, timestamps, reconnection |
| `tracer/worker.py` | Per-camera worker entrypoint wiring the above together |
| `tracer/api.py` | JSON API routes |
| `tracer/web/` | Jinja2 templates and HTMX fragments |
| `tracer/retention.py` | Hourly deletion task and orphan sweep |
| `tracer/textsearch.py` | CLIP encoder for R13 |
| `scripts/` | `ingest_file.py`, `search_cli.py`, `inject_market1501.py`, `eval_market1501.py`, `eval_table.py` |
| `pi/`, `mediamtx/` | Pi camera node config and setup; MediaMTX config for pepper |

---

# Phase 0 — Foundation

### Task 1: Container image and compose stack

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `.dockerignore`, `.gitignore`
- Test: `tests/test_environment.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a runnable `app` service with `pytest`, `torch`, `ultralytics`, `torchreid` importable, and a `db` service reachable at DSN `postgresql://tracer:tracer@db:5432/tracer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_environment.py
import os
import numpy as np
import pytest


def test_python_is_312():
    import sys
    assert sys.version_info[:2] == (3, 12)


def test_torch_imports_and_reports_device():
    import torch
    assert torch.__version__.startswith("2.")
    # DEVICE=cuda is only expected when the compose file passed the GPU through
    if os.environ.get("DEVICE") == "cuda":
        assert torch.cuda.is_available()


def test_ultralytics_and_lap_pinned():
    import ultralytics, lap  # noqa: F401
    assert ultralytics.__version__.startswith("8.4")


def test_torchreid_builds_osnet():
    import torchreid
    m = torchreid.models.build_model("osnet_x1_0", num_classes=1000, pretrained=False)
    assert sum(p.numel() for p in m.parameters()) == 2193616


def test_database_is_reachable_with_pgvector():
    import psycopg
    from pgvector.psycopg import register_vector
    with psycopg.connect(os.environ["TRACER_DSN"]) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        got = conn.execute("SELECT '[1,0,0]'::vector").fetchone()[0]
        assert np.allclose(got, [1.0, 0.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_environment.py -v`
Expected: FAIL — no image exists yet, so compose errors out before pytest runs.

- [ ] **Step 3: Write the Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgl1 libglib2.0-0 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# torch first, from the chosen index (CPU or ROCm)
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

# torchreid's setup.py imports the package to read __version__, so every
# runtime dependency must exist at build time, and build isolation must be off.
RUN pip install --no-cache-dir numpy Cython setuptools scipy h5py Pillow six \
      opencv-python-headless matplotlib gdown yacs imageio chardet future \
      tabulate tensorboard \
 && pip install --no-cache-dir --no-build-isolation --no-deps \
      git+https://github.com/KaiyangZhou/deep-person-reid.git

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
```

```text
# requirements.txt
ultralytics==8.4.176
lap>=0.5.12
fastapi==0.115.6
uvicorn[standard]==0.34.0
jinja2==3.1.5
python-multipart==0.0.20
psycopg[binary]==3.2.3
pgvector==0.3.6
PyYAML==6.0.2
pytest==8.3.4
httpx==0.28.1
```

- [ ] **Step 4: Write the compose file**

```yaml
# docker-compose.yml
x-gpu: &gpu
  devices:
    - /dev/kfd
    - /dev/dri
  group_add: ["video"]
  security_opt: ["seccomp:unconfined"]

services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: tracer
      POSTGRES_PASSWORD: tracer
      POSTGRES_DB: tracer
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tracer"]
      interval: 3s
      retries: 20

  app:
    build:
      context: .
      args:
        TORCH_INDEX: ${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
    <<: *gpu
    environment:
      TRACER_DSN: postgresql://tracer:tracer@db:5432/tracer
      TRACER_CONFIG: /app/config.yaml
      DEVICE: ${DEVICE:-cpu}
      HSA_OVERRIDE_GFX_VERSION: "10.3.0"
    volumes:
      - .:/app
      - ./data:/app/data
      - modelcache:/root/.cache/torch
    ports: ["8000:8000"]
    depends_on:
      db: {condition: service_healthy}
    command: uvicorn tracer.api:app --host 0.0.0.0 --port 8000

volumes:
  pgdata:
  modelcache:
```

- [ ] **Step 5: Add ignore files**

```text
# .gitignore
data/
__pycache__/
*.pyc
.pytest_cache/
*.pth
*.pt
```

```text
# .dockerignore
data/
.git/
__pycache__/
*.pth
*.pt
```

- [ ] **Step 6: Build and run the tests**

Run: `docker compose build && docker compose run --rm app pytest tests/test_environment.py -v`
Expected: PASS, 5 tests. `test_torch_imports_and_reports_device` skips the GPU assertion unless `DEVICE=cuda`.

- [ ] **Step 7: Verify the GPU path separately**

Run: `DEVICE=cuda TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4 docker compose build app && DEVICE=cuda docker compose run --rm app pytest tests/test_environment.py::test_torch_imports_and_reports_device -v`
Expected: PASS with `torch.cuda.is_available()` true. The ROCm image is ~22 GB; build it once.

- [ ] **Step 8: Commit**

```bash
git add Dockerfile docker-compose.yml requirements.txt .gitignore .dockerignore tests/test_environment.py
git commit -m "build: python 3.12 container with torch, ultralytics, torchreid and pgvector stack"
```

---

### Task 2: Configuration

**Files:**
- Create: `tracer/__init__.py`, `tracer/config.py`, `config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_config(path: str) -> Config`; `Config.min_walk(a: str, b: str) -> float`; attributes `cameras: dict[str, CameraCfg]`, `tau: dict[str, float]`, `retention_days: int`, `sample_fps: float`, `imgsz: int`, `device: str`, `max_tracklet_s: float`, `lost_track_s: float`, `min_tracklet_s: float`, `merge_gap_s: float`, `merge_sim: float`, `data_dir: Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from tracer.config import load_config, Config

YAML = """
data_dir: /app/data
device: cpu
sample_fps: 10
imgsz: 640
retention_days: 3
tau:
  osnet: 0.78
  clip: 0.25
tracklet:
  lost_track_s: 2.0
  min_tracklet_s: 1.0
  max_tracklet_s: 60.0
  merge_gap_s: 3.0
  merge_sim: 0.8
default_min_walk_s: 10.0
cameras:
  cam1:
    name: Corridor
    kind: rtsp
    source_url: rtsp://localhost:8554/cam1
    location: corridor
  cam2:
    name: Doorway
    kind: rtsp
    source_url: rtsp://localhost:8554/cam2
    location: doorway
pairs:
  - cameras: [cam1, cam2]
    min_walk_s: 4.0
  - cameras: [cam1, cam3]
    overlapping: true
"""


@pytest.fixture
def cfg(tmp_path) -> Config:
    p = tmp_path / "config.yaml"
    p.write_text(YAML)
    return load_config(str(p))


def test_data_dir_is_always_a_path(cfg):
    from pathlib import Path
    assert isinstance(cfg.data_dir, Path)
    assert (cfg.data_dir / "clips").name == "clips"


def test_loads_cameras_and_scalars(cfg):
    assert set(cfg.cameras) == {"cam1", "cam2"}
    assert cfg.cameras["cam1"].kind == "rtsp"
    assert cfg.tau["osnet"] == 0.78
    assert cfg.retention_days == 3
    assert cfg.max_tracklet_s == 60.0


def test_min_walk_is_zero_for_the_same_camera(cfg):
    # A person can leave and re-enter one camera with no delay at all.
    assert cfg.min_walk("cam1", "cam1") == 0.0


def test_min_walk_uses_the_configured_pair_value(cfg):
    assert cfg.min_walk("cam1", "cam2") == 4.0


def test_min_walk_is_symmetric(cfg):
    assert cfg.min_walk("cam2", "cam1") == cfg.min_walk("cam1", "cam2")


def test_overlapping_pairs_have_no_minimum(cfg):
    # Overlapping fields of view see the same person simultaneously, which is
    # legitimate, so the timing check must not reject it.
    assert cfg.min_walk("cam1", "cam3") == 0.0


def test_unconfigured_pairs_fall_back_to_the_default(cfg):
    assert cfg.min_walk("cam2", "cam9") == 10.0


def test_tau_of_an_unknown_space_is_an_error(cfg):
    with pytest.raises(KeyError):
        cfg.tau["nonexistent"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.config'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraCfg:
    id: str
    name: str
    kind: str                 # 'file' | 'rtsp'
    source_url: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("file", "rtsp"):
            raise ValueError(f"camera {self.id}: kind must be 'file' or 'rtsp', got {self.kind!r}")


@dataclass(frozen=True)
class PairCfg:
    min_walk_s: float = 0.0
    overlapping: bool = False


@dataclass(frozen=True)
class Config:
    """Loaded configuration. `data_dir` is always a Path, even when a caller
    or a test passes a string, so `cfg.data_dir / "clips"` never breaks."""
    data_dir: Path
    device: str
    sample_fps: float
    imgsz: int
    retention_days: int
    tau: dict[str, float]
    default_min_walk_s: float
    cameras: dict[str, CameraCfg]
    pairs: dict[frozenset[str], PairCfg] = field(default_factory=dict)
    lost_track_s: float = 2.0
    min_tracklet_s: float = 1.0
    max_tracklet_s: float = 60.0
    merge_gap_s: float = 3.0
    merge_sim: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, Path):
            object.__setattr__(self, "data_dir", Path(self.data_dir))

    def min_walk(self, a: str, b: str) -> float:
        """Minimum seconds a person needs to get from camera `a` to camera `b`."""
        if a == b:
            return 0.0
        pair = self.pairs.get(frozenset((a, b)))
        if pair is None:
            return self.default_min_walk_s
        if pair.overlapping:
            return 0.0
        return pair.min_walk_s


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    cameras = {
        cid: CameraCfg(
            id=cid,
            name=c["name"],
            kind=c["kind"],
            source_url=c.get("source_url"),
            location=c.get("location"),
        )
        for cid, c in (raw.get("cameras") or {}).items()
    }

    pairs: dict[frozenset[str], PairCfg] = {}
    for p in raw.get("pairs") or []:
        a, b = p["cameras"]
        pairs[frozenset((a, b))] = PairCfg(
            min_walk_s=float(p.get("min_walk_s", 0.0)),
            overlapping=bool(p.get("overlapping", False)),
        )

    t = raw.get("tracklet") or {}
    return Config(
        data_dir=Path(raw["data_dir"]),
        device=raw.get("device", "cpu"),
        sample_fps=float(raw.get("sample_fps", 10)),
        imgsz=int(raw.get("imgsz", 640)),
        retention_days=int(raw.get("retention_days", 3)),
        tau=dict(raw["tau"]),
        default_min_walk_s=float(raw.get("default_min_walk_s", 10.0)),
        cameras=cameras,
        pairs=pairs,
        lost_track_s=float(t.get("lost_track_s", 2.0)),
        min_tracklet_s=float(t.get("min_tracklet_s", 1.0)),
        max_tracklet_s=float(t.get("max_tracklet_s", 60.0)),
        merge_gap_s=float(t.get("merge_gap_s", 3.0)),
        merge_sim=float(t.get("merge_sim", 0.8)),
    )
```

- [ ] **Step 4: Create the real config file**

```yaml
# config.yaml
data_dir: /app/data
device: cpu             # the DEVICE env var overrides this in containers
sample_fps: 10          # drop to 5 if GPU headroom runs out
imgsz: 640
retention_days: 3
tau:
  osnet: 0.78           # measured: 0.5 admits every pair and filters nothing
  clip: 0.25            # calibrated in the R13 evaluation
tracklet:
  lost_track_s: 2.0
  min_tracklet_s: 1.0
  max_tracklet_s: 60.0
  merge_gap_s: 3.0
  merge_sim: 0.8
default_min_walk_s: 10.0
cameras:
  cam1:
    name: Pi webcam
    kind: rtsp
    source_url: rtsp://mediamtx:8554/cam1
    location: spot 1
  cam2:
    name: iPhone
    kind: rtsp
    source_url: rtsp://mediamtx:8554/cam2
    location: spot 2
pairs:
  - cameras: [cam1, cam2]
    min_walk_s: 4.0     # replace with the measured walk time between spots
```

Note: `yaml.safe_load` does not expand environment variables, so this file carries a literal value. The worker and embedder prefer `os.environ["DEVICE"]` when it is set, which is how compose switches the stack to the GPU without editing this file.

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_config.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add tracer/__init__.py tracer/config.py config.yaml tests/test_config.py
git commit -m "feat: configuration with per-pair walk times and overlap handling"
```

---

### Task 3: The tracklet contract and fixtures

**Files:**
- Create: `tracer/models.py`, `tracer/fixtures.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TrackletRecord(camera_id, started_at, ended_at, embedding, thumb_path, clip_path, n_crops)` validating on construction; `Candidate(tracklet_id, camera_id, started_at, ended_at, score)`; `Match(tracklet_id, camera_id, started_at, ended_at, score, thumb_url, clip_url, status)`; `unit_vector(v: np.ndarray) -> np.ndarray`; `make_tracklet(**overrides) -> TrackletRecord`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.models import Candidate, Match, TrackletRecord, unit_vector
from tracer.fixtures import make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def test_fixture_builds_a_valid_record():
    rec = make_tracklet()
    assert rec.embedding.shape == (512,)
    assert rec.embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(rec.embedding), 1.0, atol=1e-5)


def test_fixture_accepts_overrides():
    rec = make_tracklet(camera_id="cam2", n_crops=3)
    assert rec.camera_id == "cam2"
    assert rec.n_crops == 3


def test_naive_timestamps_are_rejected():
    # Review Focus #2: naive datetimes silently shift every gap the timing
    # check depends on, so they must fail loudly at the boundary.
    with pytest.raises(ValueError, match="timezone-aware"):
        make_tracklet(started_at=datetime(2026, 10, 13, 14, 0, 0))


def test_end_before_start_is_rejected():
    with pytest.raises(ValueError, match="ended_at"):
        make_tracklet(started_at=T0, ended_at=T0 - timedelta(seconds=1))


def test_tracklets_shorter_than_one_second_are_rejected():
    with pytest.raises(ValueError, match="at least"):
        make_tracklet(started_at=T0, ended_at=T0 + timedelta(milliseconds=400))


def test_wrong_embedding_dimension_is_rejected():
    with pytest.raises(ValueError, match="512"):
        make_tracklet(embedding=np.ones(128, dtype=np.float32))


def test_unnormalised_embedding_is_rejected():
    with pytest.raises(ValueError, match="normalis"):
        make_tracklet(embedding=np.full(512, 0.5, dtype=np.float32))


def test_crop_count_must_be_between_one_and_ten():
    with pytest.raises(ValueError, match="n_crops"):
        make_tracklet(n_crops=0)
    with pytest.raises(ValueError, match="n_crops"):
        make_tracklet(n_crops=11)


def test_unit_vector_normalises():
    v = unit_vector(np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32))
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)
    assert v.dtype == np.float32


def test_unit_vector_rejects_a_zero_vector():
    # Review Focus #3: a degenerate crop, or a model running in reduced
    # precision, can produce a zero vector. Normalising it yields NaN, which
    # would propagate into pgvector and silently corrupt every later ranking.
    with pytest.raises(ValueError, match="zero"):
        unit_vector(np.zeros(512, dtype=np.float32))


def test_unit_vector_rejects_nan():
    v = np.ones(512, dtype=np.float32)
    v[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        unit_vector(v)


def test_candidate_and_match_carry_the_expected_fields():
    c = Candidate(tracklet_id=1, camera_id="cam1", started_at=T0,
                  ended_at=T0 + timedelta(seconds=5), score=0.9)
    assert c.duration_s == 5.0
    m = Match(tracklet_id=1, camera_id="cam1", started_at=T0,
              ended_at=T0 + timedelta(seconds=5), score=0.9,
              thumb_url="/media/thumbs/1.jpg", clip_url="/media/clips/1.mp4",
              status="pending")
    assert m.status == "pending"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.models'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/models.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

EMBEDDING_DIM = 512
MIN_TRACKLET_S = 1.0
MAX_CROPS = 10


def unit_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalise to float32, refusing inputs that would produce NaN."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(v)):
        raise ValueError("embedding must be finite; got NaN or inf")
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        raise ValueError("cannot normalise a zero vector")
    return (v / norm).astype(np.float32)


def _require_aware(name: str, t: datetime) -> None:
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class TrackletRecord:
    """One continuous sighting of one person on one camera.

    This is the only interface between the vision pipeline and the database.
    """
    camera_id: str
    started_at: datetime
    ended_at: datetime
    embedding: np.ndarray
    thumb_path: str
    clip_path: str
    n_crops: int

    def __post_init__(self) -> None:
        _require_aware("started_at", self.started_at)
        _require_aware("ended_at", self.ended_at)
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if self.duration_s < MIN_TRACKLET_S:
            raise ValueError(f"tracklet must last at least {MIN_TRACKLET_S}s, got {self.duration_s:.3f}s")

        emb = np.asarray(self.embedding)
        if emb.shape != (EMBEDDING_DIM,):
            raise ValueError(f"embedding must have shape ({EMBEDDING_DIM},), got {emb.shape}")
        if emb.dtype != np.float32:
            raise ValueError(f"embedding must be float32, got {emb.dtype}")
        if not np.all(np.isfinite(emb)):
            raise ValueError("embedding must be finite")
        if not np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5):
            raise ValueError("embedding must be L2-normalised to 1.0 +/- 1e-5")

        if not 1 <= self.n_crops <= MAX_CROPS:
            raise ValueError(f"n_crops must be between 1 and {MAX_CROPS}, got {self.n_crops}")

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class Candidate:
    """A scored tracklet under consideration by the search."""
    tracklet_id: int
    camera_id: str
    started_at: datetime
    ended_at: datetime
    score: float

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class Match:
    """A candidate that survived filtering, as presented to a reviewer."""
    tracklet_id: int
    camera_id: str
    started_at: datetime
    ended_at: datetime
    score: float
    thumb_url: str
    clip_url: str
    status: str
```

```python
# tracer/fixtures.py
"""Valid random tracklets, so the search/API/UI lane needs no vision models."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from tracer.models import TrackletRecord, unit_vector

_T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
_counter = 0


def make_tracklet(**overrides) -> TrackletRecord:
    """Build a valid TrackletRecord. Any field can be overridden, including
    with invalid values, so validation itself can be tested."""
    global _counter
    _counter += 1
    rng = np.random.default_rng(_counter)
    defaults = dict(
        camera_id="cam1",
        started_at=_T0 + timedelta(seconds=10 * _counter),
        ended_at=_T0 + timedelta(seconds=10 * _counter + 5),
        embedding=unit_vector(rng.standard_normal(512).astype(np.float32)),
        thumb_path=f"thumbs/{_counter}.jpg",
        clip_path=f"clips/{_counter}.mp4",
        n_crops=10,
    )
    defaults.update(overrides)
    return TrackletRecord(**defaults)


def make_similar_embedding(base: np.ndarray, similarity: float) -> np.ndarray:
    """An embedding with approximately `similarity` cosine to `base`.

    Used to build search fixtures with known ranking, without any model.
    """
    base = unit_vector(base)
    rng = np.random.default_rng(abs(hash(float(similarity))) % (2**32))
    noise = rng.standard_normal(base.shape[0]).astype(np.float32)
    noise -= base * float(noise @ base)          # make it orthogonal to base
    noise = unit_vector(noise)
    return unit_vector(similarity * base + float(np.sqrt(1 - similarity**2)) * noise)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_models.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Add a test for the similarity helper the search tests depend on**

```python
# append to tests/test_models.py
from tracer.fixtures import make_similar_embedding


def test_make_similar_embedding_hits_the_requested_cosine():
    base = make_tracklet().embedding
    for target in (0.95, 0.8, 0.6):
        v = make_similar_embedding(base, target)
        assert np.isclose(float(v @ base), target, atol=1e-3)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
```

- [ ] **Step 6: Run it**

Run: `docker compose run --rm app pytest tests/test_models.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 7: Commit**

```bash
git add tracer/models.py tracer/fixtures.py tests/test_models.py
git commit -m "feat: TrackletRecord contract with invariants, plus model-free fixtures"
```

---

### Task 4: Schema and store round-trip

**Files:**
- Create: `sql/001_schema.sql`, `tracer/store.py`
- Test: `tests/conftest.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `TrackletRecord`, `Candidate` (Task 3); `Config` (Task 2).
- Produces: `Store(dsn)` with `insert_tracklet(rec, protected=False) -> int`, `get_tracklet(id) -> dict`, `candidates(embedding, space, window, cameras, limit) -> list[Candidate]`, `create_search(photo_path, query_text, embedding, space, window, cameras=None) -> int`, `record_matches(search_id, scored) -> None`, `set_match_status(search_id, tracklet_id, status) -> None`, `get_matches(search_id, limit) -> list[Match]`, `timeline(search_id) -> dict[str, list[Match]]`, `delete_expired(cutoff) -> dict[str, int]`, `reset()`.

- [ ] **Step 1: Write the schema**

```sql
-- sql/001_schema.sql
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
  embedding   VECTOR(512) NOT NULL,         -- OSNet, L2-normalised
  clip_emb    VECTOR(512),                  -- CLIP space, R13, filled later
  thumb_path  TEXT NOT NULL,
  clip_path   TEXT NOT NULL,
  n_crops     INT NOT NULL,
  protected   BOOLEAN NOT NULL DEFAULT false,
  CHECK (ended_at >= started_at)
);

-- Defined for the growth story. Queries use an exact scan: pgvector applies
-- WHERE filters after the index scan, which under-returns on narrow windows.
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
  cameras     TEXT[] NOT NULL DEFAULT '{}',   -- scope chosen by the reviewer
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
```

- [ ] **Step 2: Write the failing test**

```python
# tests/conftest.py
import os
import pytest
from tracer.store import Store


@pytest.fixture
def store() -> Store:
    s = Store(os.environ["TRACER_DSN"])
    s.ensure_schema()
    s.reset()
    yield s
    s.close()
```

```python
# tests/test_store.py
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.fixtures import make_similar_embedding, make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
WIDE = (T0 - timedelta(hours=1), T0 + timedelta(hours=1))


def test_insert_returns_an_id_and_round_trips(store):
    rec = make_tracklet(camera_id="cam1")
    tid = store.insert_tracklet(rec)
    assert tid > 0
    got = store.get_tracklet(tid)
    assert got["camera_id"] == "cam1"
    assert got["started_at"] == rec.started_at
    assert np.allclose(got["embedding"], rec.embedding, atol=1e-6)
    assert got["protected"] is False


def test_candidates_rank_by_cosine_similarity(store):
    base = make_tracklet().embedding
    ids = {}
    for sim in (0.95, 0.80, 0.60):
        rec = make_tracklet(camera_id="cam1", embedding=make_similar_embedding(base, sim),
                            started_at=T0, ended_at=T0 + timedelta(seconds=5))
        ids[sim] = store.insert_tracklet(rec)

    got = store.candidates(base, space="osnet", window=WIDE, cameras=["cam1"], limit=50)
    assert [c.tracklet_id for c in got] == [ids[0.95], ids[0.80], ids[0.60]]
    assert got[0].score > got[1].score > got[2].score
    assert np.isclose(got[0].score, 0.95, atol=1e-3)


def test_candidates_respect_the_time_window(store):
    inside = store.insert_tracklet(make_tracklet(
        started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    store.insert_tracklet(make_tracklet(
        started_at=T0 + timedelta(days=2), ended_at=T0 + timedelta(days=2, seconds=5)))
    got = store.candidates(make_tracklet().embedding, space="osnet",
                           window=(T0 - timedelta(minutes=1), T0 + timedelta(minutes=1)),
                           cameras=["cam1"], limit=50)
    assert [c.tracklet_id for c in got] == [inside]


def test_candidates_respect_the_camera_filter(store):
    a = store.insert_tracklet(make_tracklet(camera_id="cam1"))
    store.insert_tracklet(make_tracklet(camera_id="cam2"))
    got = store.candidates(make_tracklet().embedding, space="osnet",
                           window=WIDE, cameras=["cam1"], limit=50)
    assert [c.tracklet_id for c in got] == [a]


def test_candidates_on_the_clip_space_skip_rows_without_a_clip_embedding(store):
    store.insert_tracklet(make_tracklet())            # clip_emb is NULL
    got = store.candidates(make_tracklet().embedding, space="clip",
                           window=WIDE, cameras=["cam1"], limit=50)
    assert got == []


def test_search_matches_and_status_transitions(store):
    tid = store.insert_tracklet(make_tracklet())
    sid = store.create_search(photo_path="photos/a.jpg", query_text=None,
                              embedding=make_tracklet().embedding, space="osnet",
                              window=WIDE)
    store.record_matches(sid, [(tid, 0.91)])
    matches = store.get_matches(sid, limit=10)
    assert len(matches) == 1 and matches[0].status == "pending"
    assert matches[0].thumb_url.startswith("/media/")

    store.set_match_status(sid, tid, "confirmed")
    assert store.get_matches(sid, limit=10)[0].status == "confirmed"

    with pytest.raises(ValueError, match="status"):
        store.set_match_status(sid, tid, "maybe")


def test_timeline_groups_by_camera_in_time_order(store):
    t1 = store.insert_tracklet(make_tracklet(camera_id="cam1", started_at=T0,
                                             ended_at=T0 + timedelta(seconds=5)))
    t2 = store.insert_tracklet(make_tracklet(camera_id="cam2",
                                             started_at=T0 + timedelta(seconds=30),
                                             ended_at=T0 + timedelta(seconds=35)))
    t3 = store.insert_tracklet(make_tracklet(camera_id="cam1",
                                             started_at=T0 + timedelta(seconds=60),
                                             ended_at=T0 + timedelta(seconds=65)))
    sid = store.create_search(None, None, make_tracklet().embedding, "osnet", WIDE)
    store.record_matches(sid, [(t1, 0.9), (t2, 0.85), (t3, 0.8)])
    lanes = store.timeline(sid)
    assert list(lanes) == ["cam1", "cam2"]
    assert [m.tracklet_id for m in lanes["cam1"]] == [t1, t3]


def test_rejected_matches_leave_the_timeline(store):
    tid = store.insert_tracklet(make_tracklet())
    sid = store.create_search(None, None, make_tracklet().embedding, "osnet", WIDE)
    store.record_matches(sid, [(tid, 0.9)])
    store.set_match_status(sid, tid, "rejected")
    assert store.timeline(sid) == {}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.store'`

- [ ] **Step 4: Write the implementation**

```python
# tracer/store.py
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

    # ---------- tracklets ----------

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

    def set_clip_embedding(self, tracklet_id: int, embedding: np.ndarray) -> None:
        self.conn.execute("UPDATE tracklet SET clip_emb = %s WHERE id = %s",
                          (embedding, tracklet_id))

    # ---------- candidate retrieval ----------

    def candidates(self, embedding: np.ndarray, space: str,
                   window: tuple[datetime | None, datetime | None],
                   cameras: list[str], limit: int = 50) -> list[Candidate]:
        """Exact cosine scan with filters. Deliberately not using the HNSW
        index: pgvector post-filters, so an indexed LIMIT can return far fewer
        than `limit` true candidates on a narrow time window."""
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

    def max_similarity(self, embedding: np.ndarray, space: str,
                       window: tuple[datetime | None, datetime | None],
                       cameras: list[str]) -> float | None:
        """Best similarity in scope regardless of threshold, so an empty
        result can tell a reviewer *why* it is empty."""
        got = self.candidates(embedding, space, window, cameras, limit=1)
        return got[0].score if got else None

    # ---------- searches and matches ----------

    def create_search(self, photo_path: str | None, query_text: str | None,
                      embedding: np.ndarray, space: str,
                      window: tuple[datetime | None, datetime | None],
                      cameras: list[str] | None = None) -> int:
        """The camera scope is stored, not just the window: the matches
        endpoint re-runs the search later and must honour the same scope."""
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
        """Confirmed and pending matches grouped by camera, each lane in time
        order. Rejected matches leave the timeline (R10)."""
        rows = self._match_rows(search_id, exclude_rejected=True, limit=None)
        lanes: dict[str, list[Match]] = {}
        for r in rows:
            lanes.setdefault(r[1], []).append(self._to_match(r))
        for cam in lanes:
            lanes[cam].sort(key=lambda m: m.started_at)
        return {cam: lanes[cam] for cam in sorted(lanes)}

    # ---------- retention ----------

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
```

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm app pytest tests/test_store.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add sql/001_schema.sql tracer/store.py tests/conftest.py tests/test_store.py
git commit -m "feat: schema and pgvector store with exact-scan candidate retrieval"
```

---

# Phase 1 — Search path (no vision models required)

Everything in this phase is built against `make_tracklet()` fixtures. It can proceed in parallel with Phase 2.

### Task 5: Impossible-timing filter

**Files:**
- Create: `tracer/timing.py`
- Test: `tests/test_timing.py`

**Interfaces:**
- Consumes: `Candidate` (Task 3), `Config.min_walk` (Task 2).
- Produces: `compatible(a: Candidate, b: Candidate, cfg: Config) -> bool`; `filter_impossible(cands: Sequence[Candidate], cfg: Config) -> list[Candidate]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_timing.py
from datetime import datetime, timedelta, timezone

import pytest

from tracer.config import Config, CameraCfg, PairCfg
from tracer.models import Candidate
from tracer.timing import compatible, filter_impossible

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={c: CameraCfg(id=c, name=c, kind="rtsp") for c in ("cam1", "cam2", "cam3")},
        pairs={
            frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0),
            frozenset(("cam1", "cam3")): PairCfg(overlapping=True),
        },
    )


def cand(cam: str, start_s: float, dur_s: float = 5.0, score: float = 0.9) -> Candidate:
    return Candidate(tracklet_id=int(start_s * 1000) + hash(cam) % 1000, camera_id=cam,
                     started_at=T0 + timedelta(seconds=start_s),
                     ended_at=T0 + timedelta(seconds=start_s + dur_s), score=score)


def test_same_camera_is_always_compatible(cfg):
    # A person can leave one camera and walk straight back into it.
    assert compatible(cand("cam1", 0), cand("cam1", 1), cfg)


def test_gap_longer_than_the_walk_time_is_compatible(cfg):
    # cam1 ends at +5, cam2 starts at +10 => 5s gap, min_walk is 4s.
    assert compatible(cand("cam1", 0), cand("cam2", 10), cfg)


def test_gap_shorter_than_the_walk_time_is_impossible(cfg):
    # cam1 ends at +5, cam2 starts at +6 => 1s gap, min_walk is 4s.
    assert not compatible(cand("cam1", 0), cand("cam2", 6), cfg)


def test_gap_exactly_equal_to_the_walk_time_is_compatible(cfg):
    # cam1 ends at +5, cam2 starts at +9 => exactly 4s.
    assert compatible(cand("cam1", 0), cand("cam2", 9), cfg)


def test_overlapping_intervals_on_different_cameras_are_impossible(cfg):
    # Negative gap: the person would have to be in two places at once.
    assert not compatible(cand("cam1", 0, dur_s=10), cand("cam2", 5), cfg)


def test_argument_order_does_not_matter(cfg):
    a, b = cand("cam1", 0), cand("cam2", 6)
    assert compatible(a, b, cfg) == compatible(b, a, cfg)


def test_cameras_configured_as_overlapping_allow_simultaneous_sightings(cfg):
    # cam1 and cam3 see the same space, so the same person legitimately
    # appears on both at once and must not be filtered out.
    assert compatible(cand("cam1", 0, dur_s=10), cand("cam3", 2), cfg)


def test_unconfigured_pairs_use_the_default_walk_time(cfg):
    assert not compatible(cand("cam2", 0), cand("cam3", 8), cfg)   # 3s gap < 10s
    assert compatible(cand("cam2", 0), cand("cam3", 20), cfg)      # 15s gap


def test_filter_keeps_the_highest_scorer_and_drops_what_conflicts_with_it(cfg):
    best = cand("cam1", 0, score=0.95)
    impossible = cand("cam2", 6, score=0.90)       # 1s after best, needs 4s
    fine = cand("cam3", 60, score=0.80)            # 55s after best, needs 10s
    kept = filter_impossible([best, impossible, fine], cfg)
    assert [c.tracklet_id for c in kept] == [best.tracklet_id, fine.tracklet_id]


def test_filter_output_is_ordered_by_score(cfg):
    kept = filter_impossible([cand("cam1", 0, score=0.7), cand("cam1", 30, score=0.9)], cfg)
    assert [c.score for c in kept] == [0.9, 0.7]


def test_filter_is_independent_of_input_order(cfg):
    cands = [cand("cam1", 0, score=0.95), cand("cam2", 6, score=0.90),
             cand("cam3", 60, score=0.80)]
    forward = [c.tracklet_id for c in filter_impossible(cands, cfg)]
    backward = [c.tracklet_id for c in filter_impossible(list(reversed(cands)), cfg)]
    assert forward == backward


def test_a_candidate_conflicting_only_with_a_dropped_one_survives(cfg):
    # 0.90 is dropped against the 0.95 anchor. 0.85 conflicts with the dropped
    # 0.90 but not with the anchor, so it must still be kept.
    anchor = cand("cam1", 0, score=0.95)
    dropped = cand("cam2", 6, score=0.90)
    survivor = cand("cam2", 30, score=0.85)
    kept = filter_impossible([anchor, dropped, survivor], cfg)
    assert [c.tracklet_id for c in kept] == [anchor.tracklet_id, survivor.tracklet_id]


def test_empty_input_gives_empty_output(cfg):
    assert filter_impossible([], cfg) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_timing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.timing'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/timing.py
"""Reject sightings that are physically impossible given camera geometry.

Two sightings on different cameras cannot both be the same person if the gap
between them is shorter than the time needed to walk between those cameras.
Gaps are measured end-to-start, not centre-to-centre.
"""
from __future__ import annotations

from typing import Sequence

from tracer.config import Config
from tracer.models import Candidate


def compatible(a: Candidate, b: Candidate, cfg: Config) -> bool:
    if a.camera_id == b.camera_id:
        return True                      # a person can reappear on one camera
    first, second = sorted((a, b), key=lambda c: c.started_at)
    gap = (second.started_at - first.ended_at).total_seconds()
    return gap >= cfg.min_walk(a.camera_id, b.camera_id)


def filter_impossible(cands: Sequence[Candidate], cfg: Config) -> list[Candidate]:
    """Greedy maximal compatible set, taken in descending score order.

    Deterministic regardless of input order: the highest-scoring sighting
    anchors the route, and each lower-scoring one is kept only if it is
    compatible with everything already kept.
    """
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (-c.score, c.tracklet_id)):
        if all(compatible(c, k, cfg) for k in kept):
            kept.append(c)
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_timing.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Commit**

```bash
git add tracer/timing.py tests/test_timing.py
git commit -m "feat: deterministic impossible-timing filter with end-to-start gaps"
```

---

### Task 6: Search orchestration

**Files:**
- Create: `tracer/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `Store.candidates`, `Store.max_similarity` (Task 4); `filter_impossible` (Task 5); `Config.tau` (Task 2).
- Produces: `SearchResult(matches: list[Candidate], reason: str | None, best_score: float | None)`; `run_search(store, cfg, embedding, space, window, cameras, limit=10) -> SearchResult`. Reason codes: `"no_tracklets_in_scope"`, `"none_above_tau"`, `"all_rejected_by_timing"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py
from datetime import datetime, timedelta, timezone

import pytest

from tracer.config import CameraCfg, Config, PairCfg
from tracer.fixtures import make_similar_embedding, make_tracklet
from tracer.search import run_search

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
WIDE = (T0 - timedelta(hours=1), T0 + timedelta(hours=1))
CAMS = ["cam1", "cam2", "cam3"]


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={c: CameraCfg(id=c, name=c, kind="rtsp") for c in CAMS},
        pairs={frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0)},
    )


def test_empty_database_reports_no_tracklets_in_scope(store, cfg):
    q = make_tracklet().embedding
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert res.matches == []
    assert res.reason == "no_tracklets_in_scope"
    assert res.best_score is None


def test_candidates_below_tau_report_the_best_score_they_saw(store, cfg):
    q = make_tracklet().embedding
    store.insert_tracklet(make_tracklet(embedding=make_similar_embedding(q, 0.71),
                                        started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert res.matches == []
    assert res.reason == "none_above_tau"
    assert res.best_score == pytest.approx(0.71, abs=1e-3)


def test_matches_above_tau_are_returned_best_first(store, cfg):
    q = make_tracklet().embedding
    ids = {}
    for sim, start in ((0.95, 0), (0.85, 30), (0.80, 60)):
        ids[sim] = store.insert_tracklet(make_tracklet(
            camera_id="cam1", embedding=make_similar_embedding(q, sim),
            started_at=T0 + timedelta(seconds=start),
            ended_at=T0 + timedelta(seconds=start + 5)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert [m.tracklet_id for m in res.matches] == [ids[0.95], ids[0.85], ids[0.80]]
    assert res.reason is None


def test_the_limit_is_applied_after_filtering(store, cfg):
    q = make_tracklet().embedding
    for i in range(15):
        store.insert_tracklet(make_tracklet(
            camera_id="cam1", embedding=make_similar_embedding(q, 0.95 - i * 0.005),
            started_at=T0 + timedelta(seconds=i * 30),
            ended_at=T0 + timedelta(seconds=i * 30 + 5)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS, limit=10)
    assert len(res.matches) == 10


def test_impossible_timing_removes_a_high_scoring_match(store, cfg):
    q = make_tracklet().embedding
    keep = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.95),
        started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    # On cam2 one second later; min_walk between cam1 and cam2 is 4s.
    store.insert_tracklet(make_tracklet(
        camera_id="cam2", embedding=make_similar_embedding(q, 0.93),
        started_at=T0 + timedelta(seconds=6), ended_at=T0 + timedelta(seconds=11)))
    res = run_search(store, cfg, q, "osnet", WIDE, CAMS)
    assert [m.tracklet_id for m in res.matches] == [keep]


def test_the_camera_filter_is_honoured(store, cfg):
    q = make_tracklet().embedding
    store.insert_tracklet(make_tracklet(camera_id="cam2",
                                        embedding=make_similar_embedding(q, 0.95)))
    res = run_search(store, cfg, q, "osnet", WIDE, ["cam1"])
    assert res.matches == []
    assert res.reason == "no_tracklets_in_scope"


def test_the_clip_space_uses_its_own_threshold(store, cfg):
    # 0.3 is below tau.osnet (0.78) but above tau.clip (0.25): a shared
    # threshold would wrongly reject every text-search result.
    q = make_tracklet().embedding
    tid = store.insert_tracklet(make_tracklet(started_at=T0, ended_at=T0 + timedelta(seconds=5)))
    store.set_clip_embedding(tid, make_similar_embedding(q, 0.30))
    res = run_search(store, cfg, q, "clip", WIDE, CAMS)
    assert [m.tracklet_id for m in res.matches] == [tid]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.search'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/search.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tracer.config import Config
from tracer.models import Candidate
from tracer.store import Store
from tracer.timing import filter_impossible

CANDIDATE_POOL = 50


@dataclass(frozen=True)
class SearchResult:
    matches: list[Candidate]
    reason: str | None = None        # set only when matches is empty
    best_score: float | None = None  # best similarity seen, for the reason text


def run_search(store: Store, cfg: Config, embedding: np.ndarray, space: str,
               window: tuple[datetime | None, datetime | None],
               cameras: list[str], limit: int = 10) -> SearchResult:
    """Candidates -> threshold -> impossible-timing filter -> top `limit`."""
    pool = store.candidates(embedding, space, window, cameras, limit=CANDIDATE_POOL)
    if not pool:
        return SearchResult([], reason="no_tracklets_in_scope")

    best = max(c.score for c in pool)
    tau = cfg.tau[space]
    above = [c for c in pool if c.score >= tau]
    if not above:
        return SearchResult([], reason="none_above_tau", best_score=best)

    kept = filter_impossible(above, cfg)
    if not kept:
        return SearchResult([], reason="all_rejected_by_timing", best_score=best)

    return SearchResult(kept[:limit], reason=None, best_score=best)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_search.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add tracer/search.py tests/test_search.py
git commit -m "feat: search pipeline with per-space thresholds and empty-result reasons"
```

---

### Task 7: JSON API

**Files:**
- Create: `tracer/api.py`, `tracer/deps.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `run_search` (Task 6), `Store` (Task 4), `Config` (Task 2).
- Produces: FastAPI `app`; routes `POST /api/cameras`, `GET /api/cameras`, `POST /api/searches`, `GET /api/searches/{id}/matches`, `PATCH /api/searches/{id}/matches/{tracklet_id}`, `GET /api/searches/{id}/timeline`. Dependency overrides `get_store`, `get_config`, `get_embedder` for tests. An embedder satisfies `embed_image(data: bytes) -> tuple[np.ndarray, list[dict]]`.

The embedder is injected so this task needs no vision models. Task 16 supplies the real one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tracer.api import app
from tracer.config import CameraCfg, Config, PairCfg
from tracer.deps import get_config, get_embedder, get_store
from tracer.fixtures import make_similar_embedding, make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
QUERY = make_tracklet().embedding


class FakeEmbedder:
    """Stands in for YOLO + OSNet. Returns a fixed embedding and a
    configurable list of detected people."""
    def __init__(self, detections=1):
        self.detections = detections

    def embed_image(self, data: bytes):
        if self.detections == 0:
            raise ValueError("no person detected in the reference photo")
        boxes = [{"xyxy": [0, 0, 100, 300 - 50 * i], "area": 100 * (300 - 50 * i)}
                 for i in range(self.detections)]
        return QUERY, boxes


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={c: CameraCfg(id=c, name=c, kind="rtsp") for c in ("cam1", "cam2")},
        pairs={frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0)},
    )


@pytest.fixture
def client(store, cfg):
    embedder = FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_config] = lambda: cfg
    app.dependency_overrides[get_embedder] = lambda: embedder
    c = TestClient(app)
    c.embedder = embedder
    yield c
    app.dependency_overrides.clear()


def photo() -> dict:
    return {"photo": ("ref.jpg", io.BytesIO(b"not-a-real-jpeg"), "image/jpeg")}


def form(**extra) -> dict:
    data = {"from": T0.isoformat(), "to": (T0 + timedelta(hours=1)).isoformat(),
            "cameras": ["cam1", "cam2"]}
    data.update(extra)
    return data


def test_listing_cameras_returns_the_configured_set(client):
    r = client.get("/api/cameras")
    assert r.status_code == 200
    assert {c["id"] for c in r.json()} == {"cam1", "cam2"}


def test_creating_a_search_returns_an_id_and_the_detected_person(client):
    r = client.post("/api/searches", files=photo(), data=form())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["search_id"] > 0
    assert len(body["detections"]) == 1
    assert body["selected"] == 0


def test_several_detected_people_are_ranked_by_area_with_the_largest_selected(client):
    # Review Focus #1: a crowd photo must let the reviewer choose, not guess.
    client.embedder.detections = 3
    r = client.post("/api/searches", files=photo(), data=form())
    assert r.status_code == 201
    areas = [d["area"] for d in r.json()["detections"]]
    assert areas == sorted(areas, reverse=True)
    assert r.json()["selected"] == 0


def test_a_photo_with_no_person_is_a_422_with_a_readable_message(client):
    # Review Focus #1: the reviewer uploaded a landscape by mistake.
    client.embedder.detections = 0
    r = client.post("/api/searches", files=photo(), data=form())
    assert r.status_code == 422
    assert "no person" in r.json()["detail"].lower()


def test_a_reversed_time_window_is_rejected(client):
    # Review Focus #4: `from` after `to` must not silently scan everything.
    r = client.post("/api/searches", files=photo(),
                    data=form(**{"from": (T0 + timedelta(hours=1)).isoformat(),
                                 "to": T0.isoformat()}))
    assert r.status_code == 422
    assert "window" in r.json()["detail"].lower()


def test_an_empty_camera_list_is_rejected(client):
    # Review Focus #4: an empty scope is a mistake, not "search everything".
    r = client.post("/api/searches", files=photo(), data=form(cameras=[]))
    assert r.status_code == 422
    assert "camera" in r.json()["detail"].lower()


def test_an_unknown_camera_is_rejected(client):
    r = client.post("/api/searches", files=photo(), data=form(cameras=["cam9"]))
    assert r.status_code == 422
    assert "cam9" in r.json()["detail"]


def test_neither_photo_nor_text_is_rejected(client):
    r = client.post("/api/searches", data=form())
    assert r.status_code == 422


def test_matches_are_returned_with_urls_and_status(client, store):
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    r = client.get(f"/api/searches/{sid}/matches?limit=10")
    assert r.status_code == 200
    m = r.json()["matches"][0]
    assert m["tracklet_id"] == tid
    assert m["camera_id"] == "cam1"
    assert m["thumb_url"].startswith("/media/")
    assert m["clip_url"].startswith("/media/")
    assert m["status"] == "pending"
    assert 0.0 <= m["score"] <= 1.0


def test_an_empty_result_explains_itself(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.70),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    body = client.get(f"/api/searches/{sid}/matches").json()
    assert body["matches"] == []
    assert body["reason"] == "none_above_tau"
    assert body["best_score"] == pytest.approx(0.70, abs=1e-3)


def test_confirming_and_rejecting_a_match(client, store):
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    client.get(f"/api/searches/{sid}/matches")

    r = client.patch(f"/api/searches/{sid}/matches/{tid}", json={"status": "confirmed"})
    assert r.status_code == 200
    assert r.json()["status"] == "confirmed"

    r = client.patch(f"/api/searches/{sid}/matches/{tid}", json={"status": "nonsense"})
    assert r.status_code == 422


def test_the_timeline_groups_matches_into_one_lane_per_camera(client, store):
    a = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    b = store.insert_tracklet(make_tracklet(
        camera_id="cam2", embedding=make_similar_embedding(QUERY, 0.90),
        started_at=T0 + timedelta(seconds=40), ended_at=T0 + timedelta(seconds=45)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    client.get(f"/api/searches/{sid}/matches")

    lanes = client.get(f"/api/searches/{sid}/timeline").json()["lanes"]
    assert [m["tracklet_id"] for m in lanes["cam1"]] == [a]
    assert [m["tracklet_id"] for m in lanes["cam2"]] == [b]

    client.patch(f"/api/searches/{sid}/matches/{b}", json={"status": "rejected"})
    lanes = client.get(f"/api/searches/{sid}/timeline").json()["lanes"]
    assert "cam2" not in lanes


def test_the_camera_scope_chosen_at_search_time_is_honoured_later(client, store):
    """The matches endpoint re-runs the search, so the scope must be stored.
    Searching cam1 only must never surface a cam2 sighting."""
    store.insert_tracklet(make_tracklet(
        camera_id="cam2", embedding=make_similar_embedding(QUERY, 0.99),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(),
                      data=form(cameras=["cam1"])).json()["search_id"]
    body = client.get(f"/api/searches/{sid}/matches").json()
    assert body["matches"] == []
    assert body["reason"] == "no_tracklets_in_scope"


def test_a_missing_search_is_a_404(client):
    assert client.get("/api/searches/99999/matches").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.api'`

- [ ] **Step 3: Write the dependency providers**

```python
# tracer/deps.py
"""Dependency providers, overridden in tests so the API needs no models."""
from __future__ import annotations

import functools
import os

from tracer.config import Config, load_config
from tracer.store import Store


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config(os.environ.get("TRACER_CONFIG", "/app/config.yaml"))


@functools.lru_cache(maxsize=1)
def get_store() -> Store:
    store = Store(os.environ["TRACER_DSN"])
    store.ensure_schema()
    return store


@functools.lru_cache(maxsize=1)
def get_embedder():
    from tracer.reid import PhotoEmbedder     # imported lazily: loads models
    return PhotoEmbedder(get_config())
```

- [ ] **Step 4: Write the API**

```python
# tracer/api.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tracer.config import Config
from tracer.deps import get_config, get_embedder, get_store
from tracer.models import Match
from tracer.search import run_search
from tracer.store import VALID_STATUS, Store

app = FastAPI(title="Missing Person Tracing")


class StatusPatch(BaseModel):
    status: str


def _match_json(m: Match) -> dict:
    return {"tracklet_id": m.tracklet_id, "camera_id": m.camera_id,
            "started_at": m.started_at.isoformat(), "ended_at": m.ended_at.isoformat(),
            "score": round(m.score, 4), "thumb_url": m.thumb_url,
            "clip_url": m.clip_url, "status": m.status}


def _validate_scope(cfg: Config, cameras: list[str],
                    frm: datetime | None, to: datetime | None) -> None:
    if not cameras:
        raise HTTPException(422, "select at least one camera")
    unknown = [c for c in cameras if c not in cfg.cameras]
    if unknown:
        raise HTTPException(422, f"unknown camera(s): {', '.join(unknown)}")
    if frm and to and frm > to:
        raise HTTPException(422, "time window is reversed: 'from' is after 'to'")


@app.get("/api/cameras")
def list_cameras(cfg: Config = Depends(get_config)) -> list[dict]:
    return [{"id": c.id, "name": c.name, "kind": c.kind, "location": c.location}
            for c in cfg.cameras.values()]


@app.post("/api/cameras", status_code=201)
def add_camera(camera: dict, store: Store = Depends(get_store)) -> dict:
    store.conn.execute(
        """INSERT INTO camera (id, name, kind, source_url, location)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name,
             kind = EXCLUDED.kind, source_url = EXCLUDED.source_url,
             location = EXCLUDED.location""",
        (camera["id"], camera["name"], camera["kind"],
         camera.get("source_url"), camera.get("location")),
    )
    return camera


@app.post("/api/searches", status_code=201)
async def create_search(
    photo: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    frm: datetime | None = Form(default=None, alias="from"),
    to: datetime | None = Form(default=None),
    cameras: list[str] = Form(default=[]),
    cfg: Config = Depends(get_config),
    store: Store = Depends(get_store),
    embedder=Depends(get_embedder),
) -> JSONResponse:
    if photo is None and not text:
        raise HTTPException(422, "provide either a reference photo or a text description")
    _validate_scope(cfg, cameras, frm, to)

    detections: list[dict] = []
    if photo is not None:
        try:
            embedding, detections = embedder.embed_image(await photo.read())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        detections = sorted(detections, key=lambda d: d["area"], reverse=True)
        space, photo_path, query_text = "osnet", f"photos/{photo.filename}", None
    else:
        from tracer.textsearch import embed_text
        embedding = embed_text(text, cfg)
        space, photo_path, query_text = "clip", None, text

    search_id = store.create_search(photo_path, query_text, embedding, space,
                                    (frm, to), cameras)
    return JSONResponse(
        status_code=201,
        content={"search_id": search_id, "space": space,
                 "detections": detections, "selected": 0 if detections else None},
    )


def _load_search(store: Store, search_id: int) -> dict:
    row = store.conn.execute(
        """SELECT id, embedding, space, cameras, window_from, window_to
           FROM search WHERE id = %s""", (search_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no search {search_id}")
    return {"id": row[0], "embedding": row[1], "space": row[2],
            "cameras": list(row[3] or []), "window": (row[4], row[5])}


@app.get("/api/searches/{search_id}/matches")
def get_matches(search_id: int, limit: int = 10,
                cfg: Config = Depends(get_config),
                store: Store = Depends(get_store)) -> dict:
    s = _load_search(store, search_id)
    # Honour the scope the reviewer chose, not every configured camera.
    scope = s["cameras"] or list(cfg.cameras)
    result = run_search(store, cfg, s["embedding"], s["space"], s["window"],
                        scope, limit=limit)
    store.record_matches(search_id, [(m.tracklet_id, m.score) for m in result.matches])
    return {"matches": [_match_json(m) for m in store.get_matches(search_id, limit)],
            "reason": result.reason, "best_score": result.best_score}


@app.patch("/api/searches/{search_id}/matches/{tracklet_id}")
def patch_match(search_id: int, tracklet_id: int, patch: StatusPatch,
                store: Store = Depends(get_store)) -> dict:
    if patch.status not in VALID_STATUS:
        raise HTTPException(422, f"status must be one of {VALID_STATUS}")
    _load_search(store, search_id)
    store.set_match_status(search_id, tracklet_id, patch.status)
    return {"tracklet_id": tracklet_id, "status": patch.status}


@app.get("/api/searches/{search_id}/timeline")
def get_timeline(search_id: int, store: Store = Depends(get_store)) -> dict:
    _load_search(store, search_id)
    lanes = store.timeline(search_id)
    return {"lanes": {cam: [_match_json(m) for m in ms] for cam, ms in lanes.items()}}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_api.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 6: Commit**

```bash
git add tracer/api.py tracer/deps.py tests/test_api.py
git commit -m "feat: JSON API with scope validation and reviewer confirm/reject"
```

---

### Task 8: Web UI and media serving

**Files:**
- Create: `tracer/web/templates/base.html`, `search.html`, `_results.html`, `_card.html`, `_lanes.html`, `timeline.html`
- Modify: `tracer/api.py` (add the UI routes and `/media`)
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: everything from Task 7.
- Produces: `GET /` (upload and scope form), `POST /ui/searches` (form submit, returns the results fragment), `GET /ui/searches/{id}/results` (HTMX fragment), `GET /ui/searches/{id}/timeline`, `GET /media/{path}` resolved strictly inside `data/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web.py
import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tracer.api import app
from tracer.deps import get_config, get_embedder, get_store
from tracer.fixtures import make_similar_embedding, make_tracklet
from tests.test_api import QUERY, FakeEmbedder, cfg, form, photo  # noqa: F401

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def client(store, cfg, tmp_path):
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "thumbs" / "1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (tmp_path / "secret.txt").write_text("not media")
    cfg = type(cfg)(**{**cfg.__dict__, "data_dir": tmp_path})
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_config] = lambda: cfg
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_upload_page_lists_the_cameras(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "cam1" in r.text and "cam2" in r.text
    assert "<form" in r.text


def test_the_results_fragment_shows_camera_time_score_and_thumbnail(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15),
        thumb_path="thumbs/1.jpg", clip_path="clips/1.mp4"))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    r = client.get(f"/ui/searches/{sid}/results")
    assert r.status_code == 200
    assert "cam1" in r.text
    assert "/media/thumbs/1.jpg" in r.text
    assert "/media/clips/1.mp4" in r.text
    assert "0.9" in r.text
    assert "Confirm" in r.text and "Reject" in r.text


def test_an_empty_result_tells_the_reviewer_why(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.70),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    r = client.get(f"/ui/searches/{sid}/results")
    assert "0.7" in r.text
    assert "threshold" in r.text.lower()


def test_the_timeline_page_renders_one_lane_per_camera(client, store):
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(QUERY, 0.95),
        started_at=T0 + timedelta(seconds=10), ended_at=T0 + timedelta(seconds=15)))
    sid = client.post("/api/searches", files=photo(), data=form()).json()["search_id"]
    client.get(f"/api/searches/{sid}/matches")
    r = client.get(f"/ui/searches/{sid}/timeline")
    assert r.status_code == 200
    assert 'hx-trigger="every 3s"' in r.text      # live updates without SSE
    assert "cam1" in r.text


def test_media_serves_a_file_inside_the_data_directory(client):
    r = client.get("/media/thumbs/1.jpg")
    assert r.status_code == 200
    assert r.content.startswith(b"\xff\xd8")


def test_media_refuses_path_traversal(client):
    # Review Focus #5: the media route must not escape data/.
    for bad in ("/media/../secret.txt", "/media/..%2fsecret.txt",
                "/media/thumbs/../../secret.txt"):
        assert client.get(bad).status_code in (404, 400), bad


def test_media_refuses_an_absolute_path(client):
    assert client.get("/media//etc/passwd").status_code in (404, 400)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_web.py -v`
Expected: FAIL — the `/` route does not exist, returning 404.

- [ ] **Step 3: Write the templates**

```html
<!-- tracer/web/templates/base.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Missing Person Tracing{% endblock %}</title>
  <script src="/static/htmx.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 70rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr)); gap: 1rem; }
    .card { border: 1px solid #ccc; border-radius: .5rem; padding: .5rem; }
    .card img { width: 100%; height: auto; border-radius: .25rem; }
    .score { font-variant-numeric: tabular-nums; font-weight: 600; }
    .lane { display: grid; grid-template-columns: 8rem 1fr; align-items: center; gap: .5rem; margin: .5rem 0; }
    .track { position: relative; height: 2.5rem; background: #f0f0f0; border-radius: .25rem; }
    .sighting { position: absolute; top: 0; height: 100%; background: #4a7; border-radius: .25rem; }
    .sighting.confirmed { background: #2a6; }
    .note { color: #666; }
  </style>
</head>
<body>
  <h1><a href="/">Missing Person Tracing</a></h1>
  {% block content %}{% endblock %}
</body>
</html>
```

```html
<!-- tracer/web/templates/search.html -->
{% extends "base.html" %}
{% block content %}
<h2>New search</h2>
<form hx-post="/ui/searches" hx-target="#results" hx-encoding="multipart/form-data">
  <p><label>Reference photo <input type="file" name="photo" accept="image/*"></label></p>
  <p><label>or description <input type="text" name="text" placeholder="red shirt, backpack"></label></p>
  <p><label>From <input type="datetime-local" name="from"></label>
     <label>To <input type="datetime-local" name="to"></label></p>
  <fieldset>
    <legend>Cameras</legend>
    {% for c in cameras %}
      <label><input type="checkbox" name="cameras" value="{{ c.id }}" checked>
        {{ c.id }} — {{ c.name }}{% if c.location %} ({{ c.location }}){% endif %}</label><br>
    {% endfor %}
  </fieldset>
  <p><button type="submit">Search</button></p>
</form>
<div id="results"></div>
{% endblock %}
```

```html
<!-- tracer/web/templates/_results.html -->
{% if matches %}
<p><a href="/ui/searches/{{ search_id }}/timeline">Open timeline &rarr;</a></p>
<div class="grid">
  {% for m in matches %}{% include "_card.html" %}{% endfor %}
</div>
{% else %}
<p class="note">
  {% if reason == "none_above_tau" %}
    No sighting passed the similarity threshold. Best score seen:
    <span class="score">{{ "%.3f"|format(best_score) }}</span>.
  {% elif reason == "all_rejected_by_timing" %}
    Candidates were found but every one was impossible in time. Best score:
    <span class="score">{{ "%.3f"|format(best_score) }}</span>.
  {% else %}
    No sightings at all in that time window and camera selection.
  {% endif %}
</p>
{% endif %}
```

```html
<!-- tracer/web/templates/_card.html -->
<div class="card" id="match-{{ m.tracklet_id }}">
  <a href="{{ m.clip_url }}" target="_blank"><img src="{{ m.thumb_url }}" alt="sighting"></a>
  <p>{{ m.camera_id }} · {{ m.started_at[11:19] }}<br>
     <span class="score">{{ "%.3f"|format(m.score) }}</span> · {{ m.status }}</p>
  <button hx-patch="/ui/searches/{{ search_id }}/matches/{{ m.tracklet_id }}?status=confirmed"
          hx-target="#match-{{ m.tracklet_id }}" hx-swap="outerHTML">Confirm</button>
  <button hx-patch="/ui/searches/{{ search_id }}/matches/{{ m.tracklet_id }}?status=rejected"
          hx-target="#match-{{ m.tracklet_id }}" hx-swap="outerHTML">Reject</button>
</div>
```

```html
<!-- tracer/web/templates/timeline.html -->
{% extends "base.html" %}
{% block content %}
<h2>Timeline — search {{ search_id }}</h2>
<div id="lanes" hx-get="/ui/searches/{{ search_id }}/timeline?fragment=1"
     hx-trigger="every 3s" hx-swap="innerHTML">
  {% include "_lanes.html" %}
</div>
{% endblock %}
```

```html
<!-- tracer/web/templates/_lanes.html -->
{% for cam, sightings in lanes.items() %}
<div class="lane">
  <div>{{ cam }}</div>
  <div class="track">
    {% for s in sightings %}
    <a class="sighting {{ s.status }}" href="{{ s.clip_url }}" target="_blank"
       style="left: {{ s.left_pct }}%; width: {{ s.width_pct }}%"
       title="{{ s.camera_id }} {{ s.started_at }} score {{ '%.3f'|format(s.score) }}"></a>
    {% endfor %}
  </div>
</div>
{% else %}
<p class="note">No confirmed or pending sightings yet.</p>
{% endfor %}
{% if last_seen %}<p>Last known location: <strong>{{ last_seen }}</strong></p>{% endif %}
```

- [ ] **Step 4: Add the UI and media routes**

```python
# append to tracer/api.py
from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "web" / "templates"))


def _lane_geometry(lanes: dict[str, list[Match]]) -> tuple[dict, str | None]:
    """Position each sighting on a shared time axis as percentages."""
    everything = [m for ms in lanes.values() for m in ms]
    if not everything:
        return {}, None
    t0 = min(m.started_at for m in everything)
    t1 = max(m.ended_at for m in everything)
    span = max((t1 - t0).total_seconds(), 1.0)
    out: dict[str, list[dict]] = {}
    for cam, ms in lanes.items():
        out[cam] = []
        for m in ms:
            left = (m.started_at - t0).total_seconds() / span * 100.0
            width = max((m.ended_at - m.started_at).total_seconds() / span * 100.0, 1.0)
            d = _match_json(m)
            d.update(left_pct=round(left, 2), width_pct=round(min(width, 100 - left), 2))
            out[cam].append(d)
    newest = max(everything, key=lambda m: m.ended_at)
    return out, f"{newest.camera_id} at {newest.ended_at.isoformat(timespec='seconds')}"


@app.get("/", response_class=HTMLResponse)
def ui_home(request: Request, cfg: Config = Depends(get_config)):
    return TEMPLATES.TemplateResponse(
        request, "search.html", {"cameras": list(cfg.cameras.values())})


@app.get("/ui/searches/{search_id}/results", response_class=HTMLResponse)
def ui_results(request: Request, search_id: int, limit: int = 10,
               cfg: Config = Depends(get_config), store: Store = Depends(get_store)):
    body = get_matches(search_id, limit, cfg, store)
    return TEMPLATES.TemplateResponse(request, "_results.html", {
        "search_id": search_id, "matches": body["matches"],
        "reason": body["reason"], "best_score": body["best_score"]})


@app.post("/ui/searches", response_class=HTMLResponse)
async def ui_create_search(request: Request,
                           cfg: Config = Depends(get_config),
                           store: Store = Depends(get_store),
                           embedder=Depends(get_embedder)):
    """Thin wrapper: the JSON route does the work, then render the fragment.

    Dependencies are declared rather than fetched directly, so tests can
    override them the same way they do for the JSON routes.
    """
    import json

    form_data = await request.form()
    resp = await create_search(
        photo=form_data.get("photo") or None,
        text=form_data.get("text") or None,
        frm=form_data.get("from") or None,
        to=form_data.get("to") or None,
        cameras=form_data.getlist("cameras"),
        cfg=cfg, store=store, embedder=embedder,
    )
    search_id = json.loads(resp.body)["search_id"]
    return ui_results(request, search_id, 10, cfg, store)


@app.patch("/ui/searches/{search_id}/matches/{tracklet_id}", response_class=HTMLResponse)
def ui_patch_match(request: Request, search_id: int, tracklet_id: int, status: str,
                   store: Store = Depends(get_store)):
    patch_match(search_id, tracklet_id, StatusPatch(status=status), store)
    if status == "rejected":
        return HTMLResponse("")                     # the card leaves the page
    m = next(m for m in store.get_matches(search_id, 50) if m.tracklet_id == tracklet_id)
    # Just the one card: hx-swap is outerHTML on #match-<id>.
    return TEMPLATES.TemplateResponse(request, "_card.html",
                                      {"search_id": search_id, "m": _match_json(m)})


@app.get("/ui/searches/{search_id}/timeline", response_class=HTMLResponse)
def ui_timeline(request: Request, search_id: int, fragment: int = 0,
                store: Store = Depends(get_store)):
    lanes, last_seen = _lane_geometry(store.timeline(search_id))
    ctx = {"search_id": search_id, "lanes": lanes, "last_seen": last_seen}
    name = "_lanes.html" if fragment else "timeline.html"
    return TEMPLATES.TemplateResponse(request, name, ctx)


@app.get("/media/{path:path}")
def media(path: str, cfg: Config = Depends(get_config)) -> FileResponse:
    """Serve a file from data/, refusing anything that escapes it."""
    root = Path(cfg.data_dir).resolve()
    try:
        target = (root / path).resolve()
        target.relative_to(root)                    # raises if outside root
    except (ValueError, OSError):
        raise HTTPException(404, "not found")
    if not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)
```

- [ ] **Step 5: Vendor HTMX so the UI works offline**

```bash
mkdir -p tracer/web/static
curl -sL -o tracer/web/static/htmx.min.js https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js
test -s tracer/web/static/htmx.min.js && echo "htmx vendored"
```

Then mount it by adding this near the top of `tracer/api.py`, after `app` is created:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "web" / "static")),
          name="static")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_web.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 7: Verify the full suite still passes**

Run: `docker compose run --rm app pytest -v`
Expected: PASS, all tests from Tasks 1–8.

- [ ] **Step 8: Commit**

```bash
git add tracer/web tracer/api.py tests/test_web.py
git commit -m "feat: HTMX reviewer UI, timeline lanes, and traversal-safe media route"
```

**Phase 1 checkpoint:** the whole search path works end to end on fixture data. `docker compose up` serves a usable UI. No vision model has been written yet.

---

# Phase 2 — Vision pipeline

Runs in parallel with Phase 1. Needs the GPU image and the OSNet weights.

### Task 9: Crop selection

**Files:**
- Create: `tracer/reid.py` (crop selection only; the embedder arrives in Task 10)
- Test: `tests/test_crop_selection.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Crop(image: np.ndarray, t: datetime, edge_cut: bool)`; `select_crops(crops: Sequence[Crop], n: int = 10, min_h: int = 64) -> list[Crop]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crop_selection.py
from datetime import datetime, timedelta, timezone

import numpy as np

from tracer.reid import Crop, select_crops

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def crop(h: int, i: int = 0, edge_cut: bool = False) -> Crop:
    return Crop(image=np.full((h, h // 2, 3), i % 256, dtype=np.uint8),
                t=T0 + timedelta(milliseconds=100 * i), edge_cut=edge_cut)


def test_crops_shorter_than_the_minimum_height_are_dropped():
    got = select_crops([crop(63, 0), crop(64, 1), crop(200, 2)])
    assert [c.image.shape[0] for c in got] == [64, 200]


def test_edge_cut_crops_are_dropped():
    # A person half outside the frame gives a misleading appearance vector.
    got = select_crops([crop(100, 0, edge_cut=True), crop(100, 1)])
    assert len(got) == 1
    assert got[0].t == T0 + timedelta(milliseconds=100)


def test_at_most_n_crops_are_returned():
    got = select_crops([crop(100, i) for i in range(40)], n=10)
    assert len(got) == 10


def test_crops_are_spread_evenly_over_time_not_taken_from_the_start():
    crops = [crop(100, i) for i in range(40)]
    got = select_crops(crops, n=5)
    assert got[0].t == crops[0].t
    assert got[-1].t == crops[-1].t
    gaps = [(b.t - a.t).total_seconds() for a, b in zip(got, got[1:])]
    assert max(gaps) - min(gaps) < 0.25


def test_results_stay_in_time_order():
    got = select_crops([crop(100, i) for i in range(20)], n=7)
    assert [c.t for c in got] == sorted(c.t for c in got)


def test_fewer_crops_than_requested_returns_all_usable_ones():
    got = select_crops([crop(100, 0), crop(100, 1)], n=10)
    assert len(got) == 2


def test_no_usable_crops_returns_empty():
    assert select_crops([crop(10, 0), crop(100, 1, edge_cut=True)]) == []


def test_empty_input_returns_empty():
    assert select_crops([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_crop_selection.py -v`
Expected: FAIL with `ImportError: cannot import name 'Crop' from 'tracer.reid'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/reid.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

MIN_CROP_HEIGHT = 64
MAX_CROPS = 10
OSNET_INPUT = (256, 128)          # height, width


@dataclass(frozen=True)
class Crop:
    """One person crop taken from one frame, in RGB."""
    image: np.ndarray
    t: datetime
    edge_cut: bool = False

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def select_crops(crops: Sequence[Crop], n: int = MAX_CROPS,
                 min_h: int = MIN_CROP_HEIGHT) -> list[Crop]:
    """Up to `n` usable crops, spread evenly across the tracklet's lifetime.

    Crops shorter than `min_h` are too small to embed reliably, and crops
    touching a frame edge show only part of the person.
    """
    usable = [c for c in sorted(crops, key=lambda c: c.t)
              if c.height >= min_h and not c.edge_cut]
    if len(usable) <= n:
        return usable
    idx = np.linspace(0, len(usable) - 1, n).round().astype(int)
    return [usable[i] for i in dict.fromkeys(idx.tolist())]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_crop_selection.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add tracer/reid.py tests/test_crop_selection.py
git commit -m "feat: crop selection with size, edge and even-spread rules"
```

---

### Task 10: OSNet embedder

**Files:**
- Create: `scripts/fetch_weights.sh`
- Modify: `tracer/reid.py` (add `OsnetEmbedder`)
- Test: `tests/test_embedder.py`

**Interfaces:**
- Consumes: `Crop`, `select_crops` (Task 9); `unit_vector` (Task 3).
- Produces: `OsnetEmbedder(weights_path: str, device: str = "cpu", model_name: str = "osnet_x1_0", extractor=None)` with `embed_images(images: list[np.ndarray]) -> np.ndarray` returning `(N, 512)` L2-normalised rows, and `embed_tracklet(crops: Sequence[Crop]) -> tuple[np.ndarray, int]` returning the mean embedding and the crop count actually used.

- [ ] **Step 1: Write the weight fetch script**

```bash
# scripts/fetch_weights.sh
#!/usr/bin/env bash
# Download OSNet re-ID weights. NEVER use torchreid's pretrained=True, which
# fetches ImageNet weights that are inadequate for re-identification.
set -euo pipefail
DEST="${1:-data/weights}"
BASE="https://huggingface.co/kaiyangzhou/osnet/resolve/main"
mkdir -p "$DEST"

declare -A FILES=(
  [osnet_x1_0_msmt17.pth]="osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
  [osnet_ain_x1_0_msmt17.pth]="osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)

for local in "${!FILES[@]}"; do
  if [[ -s "$DEST/$local" ]]; then
    echo "have $local"
  else
    echo "fetching $local"
    curl -fsSL -o "$DEST/$local" "$BASE/${FILES[$local]}"
  fi
  python3 - "$DEST/$local" <<'PY'
import sys, pathlib
size = pathlib.Path(sys.argv[1]).stat().st_size
assert size > 5_000_000, f"{sys.argv[1]} is only {size} bytes; download failed"
print(f"  ok {sys.argv[1]} ({size/1e6:.1f} MB)")
PY
done
```

Run: `chmod +x scripts/fetch_weights.sh && ./scripts/fetch_weights.sh data/weights`
Expected: both files present, roughly 17 MB each.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_embedder.py
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from tracer.reid import Crop, OsnetEmbedder

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
WEIGHTS = Path("data/weights/osnet_x1_0_msmt17.pth")


class FakeExtractor:
    """Returns rows we control, so the degenerate-embedding guards can be
    tested deterministically without depending on model behaviour."""
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, images):
        import torch
        return torch.tensor(np.asarray(self.rows[:len(images)], dtype=np.float32))


def crop(i: int, h: int = 200) -> Crop:
    return Crop(image=np.full((h, h // 2, 3), (i * 37) % 256, dtype=np.uint8),
                t=T0 + timedelta(milliseconds=100 * i))


def test_zero_embeddings_are_skipped_not_normalised():
    # Review Focus #3: normalising a zero row emits NaN, which would
    # propagate into pgvector and silently corrupt every later ranking.
    rows = [np.zeros(512), np.ones(512), np.ones(512) * 2]
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    emb.extractor = FakeExtractor(rows)
    vec, used = emb.embed_tracklet([crop(0), crop(1), crop(2)])
    assert used == 2
    assert np.all(np.isfinite(vec))
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_nan_embeddings_are_skipped():
    bad = np.ones(512); bad[3] = np.nan
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    emb.extractor = FakeExtractor([bad, np.ones(512)])
    vec, used = emb.embed_tracklet([crop(0), crop(1)])
    assert used == 1
    assert np.all(np.isfinite(vec))


def test_all_embeddings_degenerate_is_an_error():
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    emb.extractor = FakeExtractor([np.zeros(512), np.zeros(512)])
    with pytest.raises(ValueError, match="no usable"):
        emb.embed_tracklet([crop(0), crop(1)])


def test_no_usable_crops_is_an_error():
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    emb.extractor = FakeExtractor([np.ones(512)])
    with pytest.raises(ValueError, match="no usable"):
        emb.embed_tracklet([Crop(np.zeros((10, 5, 3), np.uint8), T0)])


@pytest.mark.skipif(not WEIGHTS.exists(), reason="run scripts/fetch_weights.sh first")
class TestWithRealWeights:
    @pytest.fixture(scope="class")
    def embedder(self) -> OsnetEmbedder:
        return OsnetEmbedder(str(WEIGHTS), device="cpu")

    def test_embeddings_are_512d_unit_vectors(self, embedder):
        out = embedder.embed_images([crop(1).image, crop(2).image])
        assert out.shape == (2, 512)
        assert out.dtype == np.float32
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_the_same_image_embeds_identically(self, embedder):
        img = crop(5).image
        a, b = embedder.embed_images([img, img])
        assert float(a @ b) == pytest.approx(1.0, abs=1e-4)

    def test_different_images_are_not_identical(self, embedder):
        a, b = embedder.embed_images([crop(1).image, crop(200).image])
        assert float(a @ b) < 0.999

    def test_embed_tracklet_returns_a_unit_mean_and_the_count_used(self, embedder):
        vec, used = embedder.embed_tracklet([crop(i) for i in range(25)])
        assert used == 10                       # capped by MAX_CROPS
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_embedder.py -v`
Expected: FAIL with `ImportError: cannot import name 'OsnetEmbedder'`

- [ ] **Step 4: Write the implementation**

```python
# append to tracer/reid.py
import cv2

from tracer.models import unit_vector


class OsnetEmbedder:
    """OSNet appearance embeddings, 512-d and L2-normalised.

    The weights must be re-ID trained (MSMT17). torchreid's pretrained=True
    loads ImageNet weights, which do not work for re-identification.
    """

    def __init__(self, weights_path: str, device: str = "cpu",
                 model_name: str = "osnet_x1_0", extractor=None) -> None:
        if extractor is not None:
            self.extractor = extractor
            return
        from torchreid.utils import FeatureExtractor
        self.extractor = FeatureExtractor(
            model_name=model_name, model_path=weights_path, device=device)

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h < 2 or w < 2:
            raise ValueError(f"crop too small to resize: {image.shape}")
        return cv2.resize(image, (OSNET_INPUT[1], OSNET_INPUT[0]),
                          interpolation=cv2.INTER_LINEAR)

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """(N, 512) L2-normalised rows. Degenerate rows are returned as NaN
        so callers can decide; embed_tracklet drops them."""
        prepared = [self._prepare(im) for im in images]
        raw = self.extractor(prepared)
        out = np.asarray(raw.cpu() if hasattr(raw, "cpu") else raw, dtype=np.float32)
        out = np.atleast_2d(out)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (out / norms).astype(np.float32)

    def embed_tracklet(self, crops: Sequence[Crop]) -> tuple[np.ndarray, int]:
        """Mean of up to 10 evenly spread crop embeddings, L2-normalised."""
        chosen = select_crops(crops)
        if not chosen:
            raise ValueError("no usable crops in this tracklet")

        rows = self.embed_images([c.image for c in chosen])
        good = [r for r in rows if np.all(np.isfinite(r)) and np.linalg.norm(r) > 1e-6]
        dropped = len(rows) - len(good)
        if dropped:
            log.warning("dropped %d degenerate crop embedding(s)", dropped)
        if not good:
            raise ValueError("no usable crop embeddings in this tracklet")

        return unit_vector(np.mean(np.stack(good), axis=0)), len(good)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_embedder.py -v`
Expected: PASS. The four guard tests always run; the five real-weight tests run once `data/weights/osnet_x1_0_msmt17.pth` exists.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_weights.sh tracer/reid.py tests/test_embedder.py
git commit -m "feat: OSNet embedder with MSMT17 weights and degenerate-embedding guards"
```

---

### Task 11: Tracklet building and merging

**Files:**
- Create: `tracer/detect_track.py`
- Modify: `tracer/store.py` (add `update_tracklet`)
- Test: `tests/test_detect_track.py`

**Interfaces:**
- Consumes: `Crop` (Task 9), `Config` (Task 2), `TrackletRecord` (Task 3).
- Produces: `Detection(track_id: int, xyxy: tuple[int, int, int, int], edge_cut: bool, crop: np.ndarray)`; `RawTracklet(camera_id, track_id, started_at, ended_at, crops: list[Crop], frames: list[tuple[datetime, np.ndarray]])`; `TrackletBuilder(cfg, camera_id, on_close)` with `add(t, detections)` and `flush(t)`; `TrackletMerger(cfg, store)` with `offer(rec: TrackletRecord, embedding_of_prev_weight: int) -> tuple[str, int]` returning `("new" | "merged", tracklet_id)`; `yolo_detections(result, frame, t) -> list[Detection]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detect_track.py
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.config import CameraCfg, Config
from tracer.detect_track import Detection, TrackletBuilder, TrackletMerger
from tracer.fixtures import make_similar_embedding, make_tracklet

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg() -> Config:
    return Config(
        data_dir="/tmp", device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="rtsp")},
        lost_track_s=2.0, min_tracklet_s=1.0, max_tracklet_s=60.0,
        merge_gap_s=3.0, merge_sim=0.8,
    )


def det(track_id: int = 1, h: int = 200) -> Detection:
    return Detection(track_id=track_id, xyxy=(10, 10, 110, 10 + h), edge_cut=False,
                     crop=np.full((h, 100, 3), 128, dtype=np.uint8))


def feed(builder, seconds, track_id=1, step=0.1):
    """Feed detections at `step` intervals for `seconds` from T0."""
    t = T0
    end = T0 + timedelta(seconds=seconds)
    while t <= end:
        builder.add(t, [det(track_id)])
        t += timedelta(seconds=step)
    return t


def test_a_track_closes_after_the_lost_timeout(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    last = feed(b, 3.0)
    b.add(last + timedelta(seconds=2.5), [])        # nothing seen for 2.5s
    assert len(closed) == 1
    assert closed[0].track_id == 1
    assert closed[0].started_at == T0


def test_a_track_does_not_close_before_the_timeout(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    last = feed(b, 3.0)
    b.add(last + timedelta(seconds=1.5), [])        # still within 2.0s
    assert closed == []


def test_tracklets_shorter_than_the_minimum_are_dropped(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    b.add(T0, [det()])
    b.add(T0 + timedelta(seconds=0.4), [det()])
    b.flush(T0 + timedelta(seconds=5))
    assert closed == []


def test_a_long_tracklet_is_force_closed_at_the_cap(cfg):
    # Unbounded tracklets would buffer frames until memory runs out, and a
    # mean embedding over minutes averages across too much pose variation.
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    feed(b, 65.0, step=0.5)
    assert len(closed) >= 1
    assert closed[0].duration_s == pytest.approx(60.0, abs=1.0)


def test_two_track_ids_produce_two_tracklets(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    for i in range(30):
        t = T0 + timedelta(seconds=0.1 * i)
        b.add(t, [det(1), det(2)])
    b.flush(T0 + timedelta(seconds=10))
    assert sorted(c.track_id for c in closed) == [1, 2]


def test_flush_closes_everything_open(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    feed(b, 3.0)
    b.flush(T0 + timedelta(seconds=3.1))
    assert len(closed) == 1


def test_crops_are_collected_on_the_tracklet(cfg):
    closed = []
    b = TrackletBuilder(cfg, "cam1", closed.append)
    feed(b, 3.0)
    b.flush(T0 + timedelta(seconds=4))
    assert len(closed[0].crops) > 10
    assert all(c.t is not None for c in closed[0].crops)


class FakeStore:
    def __init__(self):
        self.inserted, self.updated, self.next_id = [], [], 100

    def insert_tracklet(self, rec, protected=False):
        self.next_id += 1
        self.inserted.append((self.next_id, rec))
        return self.next_id

    def update_tracklet(self, tracklet_id, rec, n_crops):
        self.updated.append((tracklet_id, rec, n_crops))


def test_the_first_tracklet_on_a_camera_is_inserted(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    action, tid = m.offer(make_tracklet(camera_id="cam1"))
    assert action == "new" and tid == 101
    assert len(store.inserted) == 1


def test_a_close_similar_follow_up_is_merged(cfg):
    # The tracker splitting one walk into two tracklets is the documented
    # failure mode; merging is baseline behaviour, not a fallback.
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    second = make_tracklet(camera_id="cam1",
                           started_at=T0 + timedelta(seconds=7),
                           ended_at=T0 + timedelta(seconds=12),
                           embedding=make_similar_embedding(first.embedding, 0.9))
    action, tid = m.offer(second)
    assert action == "merged" and tid == 101
    merged = store.updated[0][1]
    assert merged.started_at == T0
    assert merged.ended_at == T0 + timedelta(seconds=12)
    assert np.isclose(np.linalg.norm(merged.embedding), 1.0, atol=1e-5)


def test_a_distant_follow_up_is_not_merged(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    action, _ = m.offer(make_tracklet(
        camera_id="cam1", started_at=T0 + timedelta(seconds=20),
        ended_at=T0 + timedelta(seconds=25),
        embedding=make_similar_embedding(first.embedding, 0.95)))
    assert action == "new"


def test_a_dissimilar_follow_up_is_not_merged(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    action, _ = m.offer(make_tracklet(
        camera_id="cam1", started_at=T0 + timedelta(seconds=6),
        ended_at=T0 + timedelta(seconds=11),
        embedding=make_similar_embedding(first.embedding, 0.5)))
    assert action == "new"


def test_tracklets_on_different_cameras_are_never_merged(cfg):
    store = FakeStore()
    m = TrackletMerger(cfg, store)
    first = make_tracklet(camera_id="cam1", started_at=T0,
                          ended_at=T0 + timedelta(seconds=5))
    m.offer(first)
    action, _ = m.offer(make_tracklet(
        camera_id="cam2", started_at=T0 + timedelta(seconds=6),
        ended_at=T0 + timedelta(seconds=11),
        embedding=make_similar_embedding(first.embedding, 0.95)))
    assert action == "new"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_detect_track.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.detect_track'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/detect_track.py
"""Turn per-frame detections into tracklets, merging fragments."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np

from tracer.config import Config
from tracer.models import TrackletRecord, unit_vector
from tracer.reid import Crop

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    track_id: int
    xyxy: tuple[int, int, int, int]
    edge_cut: bool
    crop: np.ndarray


@dataclass
class RawTracklet:
    camera_id: str
    track_id: int
    started_at: datetime
    ended_at: datetime
    crops: list[Crop] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


class TrackletBuilder:
    """Accumulates detections per track ID and closes tracklets when a track
    is lost or when it exceeds the duration cap."""

    def __init__(self, cfg: Config, camera_id: str,
                 on_close: Callable[[RawTracklet], None]) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self.on_close = on_close
        self._open: dict[int, RawTracklet] = {}

    def add(self, t: datetime, detections: Sequence[Detection]) -> None:
        for d in detections:
            tr = self._open.get(d.track_id)
            if tr is None:
                tr = RawTracklet(self.camera_id, d.track_id, t, t)
                self._open[d.track_id] = tr
            tr.ended_at = t
            tr.crops.append(Crop(image=d.crop, t=t, edge_cut=d.edge_cut))

        seen = {d.track_id for d in detections}
        for track_id, tr in list(self._open.items()):
            lost = (t - tr.ended_at).total_seconds()
            if track_id not in seen and lost > self.cfg.lost_track_s:
                self._close(track_id)
            elif tr.duration_s >= self.cfg.max_tracklet_s:
                log.info("force-closing tracklet %s at the %.0fs cap",
                         track_id, self.cfg.max_tracklet_s)
                self._close(track_id)

    def flush(self, t: datetime) -> None:
        for track_id in list(self._open):
            self._close(track_id)

    def _close(self, track_id: int) -> None:
        tr = self._open.pop(track_id)
        if tr.duration_s < self.cfg.min_tracklet_s:
            log.debug("dropping %.2fs tracklet as noise", tr.duration_s)
            return
        self.on_close(tr)


class TrackletMerger:
    """Rejoins tracklets the tracker split: same camera, small gap, high
    appearance similarity."""

    def __init__(self, cfg: Config, store) -> None:
        self.cfg = cfg
        self.store = store
        self._last: dict[str, tuple[int, TrackletRecord]] = {}

    def offer(self, rec: TrackletRecord) -> tuple[str, int]:
        prev = self._last.get(rec.camera_id)
        if prev is not None:
            prev_id, prev_rec = prev
            gap = (rec.started_at - prev_rec.ended_at).total_seconds()
            sim = float(rec.embedding @ prev_rec.embedding)
            if 0 <= gap < self.cfg.merge_gap_s and sim >= self.cfg.merge_sim:
                merged = self._merge(prev_rec, rec)
                self.store.update_tracklet(prev_id, merged, merged.n_crops)
                self._last[rec.camera_id] = (prev_id, merged)
                log.info("merged tracklet into %s (gap %.2fs, sim %.3f)",
                         prev_id, gap, sim)
                return "merged", prev_id

        new_id = self.store.insert_tracklet(rec)
        self._last[rec.camera_id] = (new_id, rec)
        return "new", new_id

    @staticmethod
    def _merge(a: TrackletRecord, b: TrackletRecord) -> TrackletRecord:
        total = a.n_crops + b.n_crops
        blended = unit_vector((a.embedding * a.n_crops + b.embedding * b.n_crops) / total)
        return TrackletRecord(
            camera_id=a.camera_id,
            started_at=min(a.started_at, b.started_at),
            ended_at=max(a.ended_at, b.ended_at),
            embedding=blended,
            thumb_path=a.thumb_path,
            clip_path=a.clip_path,
            n_crops=min(total, 10),
        )


def yolo_detections(result, frame: np.ndarray, t: datetime,
                    edge_margin: int = 2) -> list[Detection]:
    """Convert one Ultralytics result into Detections, cropping RGB patches."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return []
    h, w = frame.shape[:2]
    out: list[Detection] = []
    for xyxy, track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        edge_cut = (x1 <= edge_margin or y1 <= edge_margin
                    or x2 >= w - edge_margin or y2 >= h - edge_margin)
        out.append(Detection(track_id=int(track_id), xyxy=(x1, y1, x2, y2),
                             edge_cut=edge_cut,
                             crop=frame[y1:y2, x1:x2, ::-1].copy()))   # BGR->RGB
    return out
```

- [ ] **Step 4: Add `update_tracklet` to the store**

```python
# append to tracer/store.py, inside class Store
    def update_tracklet(self, tracklet_id: int, rec: TrackletRecord, n_crops: int) -> None:
        self.conn.execute(
            """UPDATE tracklet SET started_at = %s, ended_at = %s, embedding = %s,
                                   n_crops = %s
               WHERE id = %s""",
            (rec.started_at, rec.ended_at, rec.embedding, n_crops, tracklet_id),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_detect_track.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 6: Commit**

```bash
git add tracer/detect_track.py tracer/store.py tests/test_detect_track.py
git commit -m "feat: tracklet building with duration cap and same-camera merging"
```

---

### Task 12: Clips and thumbnails

**Files:**
- Create: `tracer/clips.py`
- Test: `tests/test_clips.py`

**Interfaces:**
- Consumes: `Crop` (Task 9).
- Produces: `FrameBuffer(seconds: float)` with `add(t, frame)` and `window(t0, t1) -> list[tuple[datetime, np.ndarray]]`; `write_clip(frames, path, fps) -> Path`; `write_thumbnail(crops: Sequence[Crop], path) -> Path` using the largest crop.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clips.py
import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.clips import FrameBuffer, write_clip, write_thumbnail
from tracer.reid import Crop

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def frame(i: int, h: int = 64, w: int = 128) -> np.ndarray:
    return np.full((h, w, 3), i % 256, dtype=np.uint8)


def test_the_buffer_discards_frames_older_than_its_window():
    buf = FrameBuffer(seconds=2.0)
    for i in range(50):
        buf.add(T0 + timedelta(seconds=0.1 * i), frame(i))
    held = buf.window(T0, T0 + timedelta(seconds=100))
    span = (held[-1][0] - held[0][0]).total_seconds()
    assert span <= 2.0 + 0.1
    assert len(held) <= 22


def test_the_buffer_returns_only_the_requested_window():
    buf = FrameBuffer(seconds=10.0)
    for i in range(50):
        buf.add(T0 + timedelta(seconds=0.1 * i), frame(i))
    got = buf.window(T0 + timedelta(seconds=1.0), T0 + timedelta(seconds=2.0))
    assert all(T0 + timedelta(seconds=1.0) <= t <= T0 + timedelta(seconds=2.0)
               for t, _ in got)
    assert len(got) == pytest.approx(11, abs=1)


def test_an_empty_window_returns_nothing():
    buf = FrameBuffer(seconds=5.0)
    assert buf.window(T0, T0 + timedelta(seconds=1)) == []


def test_write_clip_produces_a_playable_mp4(tmp_path):
    frames = [(T0 + timedelta(seconds=0.1 * i), frame(i)) for i in range(30)]
    out = write_clip(frames, tmp_path / "clip.mp4", fps=10.0)
    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width,height",
         "-count_frames", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout
    assert "width=128" in probe and "height=64" in probe
    assert "nb_read_frames=30" in probe


def test_write_clip_with_no_frames_raises(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        write_clip([], tmp_path / "empty.mp4", fps=10.0)


def test_the_thumbnail_is_the_largest_crop(tmp_path):
    crops = [Crop(frame(1, h=100, w=50), T0),
             Crop(frame(2, h=300, w=150), T0 + timedelta(seconds=1)),
             Crop(frame(3, h=120, w=60), T0 + timedelta(seconds=2))]
    out = write_thumbnail(crops, tmp_path / "thumb.jpg")
    assert out.exists()
    import cv2
    img = cv2.imread(str(out))
    assert img.shape[0] == 300 and img.shape[1] == 150


def test_write_thumbnail_with_no_crops_raises(tmp_path):
    with pytest.raises(ValueError, match="no crops"):
        write_thumbnail([], tmp_path / "thumb.jpg")


def test_parent_directories_are_created(tmp_path):
    out = write_thumbnail([Crop(frame(1, 80, 40), T0)], tmp_path / "a" / "b" / "t.jpg")
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_clips.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.clips'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/clips.py
"""Rolling frame buffer, clip writing and thumbnails."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from tracer.reid import Crop


class FrameBuffer:
    """Keeps the most recent `seconds` of frames so a clip can start before
    the tracklet did. Bounded by time, so memory cannot grow without limit."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._frames: deque[tuple[datetime, np.ndarray]] = deque()

    def add(self, t: datetime, frame: np.ndarray) -> None:
        self._frames.append((t, frame))
        cutoff = t - timedelta(seconds=self.seconds)
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def window(self, t0: datetime, t1: datetime) -> list[tuple[datetime, np.ndarray]]:
        return [(t, f) for t, f in self._frames if t0 <= t <= t1]

    def __len__(self) -> int:
        return len(self._frames)


def write_clip(frames: Sequence[tuple[datetime, np.ndarray]],
               path: str | Path, fps: float) -> Path:
    if not frames:
        raise ValueError("cannot write a clip with no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0][1].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(fps, 1.0), (w, h))
    try:
        for _, frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def write_thumbnail(crops: Sequence[Crop], path: str | Path) -> Path:
    """Save the largest crop, which is usually the clearest view."""
    if not crops:
        raise ValueError("cannot write a thumbnail with no crops")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = max(crops, key=lambda c: c.image.shape[0] * c.image.shape[1])
    cv2.imwrite(str(path), best.image[:, :, ::-1])      # RGB -> BGR for cv2
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_clips.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add tracer/clips.py tests/test_clips.py
git commit -m "feat: time-bounded frame buffer, clip writing and thumbnails"
```

---

### Task 13: Frame sources

**Files:**
- Create: `tracer/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Config` (Task 2).
- Produces: `FileSource(path, start_at: datetime, sample_fps: float)` and `RtspSource(url, sample_fps, capture_factory=None, max_backoff_s=8.0)`, both iterable as `Iterator[tuple[datetime, np.ndarray]]`; `RtspSource.disconnects: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.ingest import FileSource, RtspSource

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def clip(tmp_path):
    """A 5-second 20 fps synthetic clip: 100 frames."""
    import cv2
    path = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    for i in range(100):
        w.write(np.full((48, 64, 3), i % 256, dtype=np.uint8))
    w.release()
    return path


def test_a_naive_start_time_is_rejected(clip):
    # Review Focus #2: a naive start time shifts every timestamp in the file
    # and silently corrupts the timing check.
    with pytest.raises(ValueError, match="timezone-aware"):
        FileSource(clip, start_at=datetime(2026, 10, 13, 14, 0, 0), sample_fps=10)


def test_file_timestamps_derive_from_the_start_time_and_frame_position(clip):
    got = list(FileSource(clip, start_at=T0, sample_fps=20))
    assert got[0][0] == T0
    assert got[1][0] == pytest.approx(T0 + timedelta(seconds=0.05), abs=timedelta(milliseconds=2))
    assert got[-1][0] < T0 + timedelta(seconds=5.1)


def test_sampling_reduces_the_frame_rate(clip):
    at_20 = list(FileSource(clip, start_at=T0, sample_fps=20))
    at_10 = list(FileSource(clip, start_at=T0, sample_fps=10))
    assert len(at_20) == pytest.approx(100, abs=2)
    assert len(at_10) == pytest.approx(50, abs=2)


def test_sampled_timestamps_are_still_wall_clock_correct(clip):
    got = list(FileSource(clip, start_at=T0, sample_fps=10))
    gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(got, got[1:])]
    assert all(abs(g - 0.1) < 0.02 for g in gaps)


def test_frames_are_returned_as_arrays(clip):
    t, frame = next(iter(FileSource(clip, start_at=T0, sample_fps=10)))
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (48, 64, 3)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(FileSource(tmp_path / "nope.mp4", start_at=T0, sample_fps=10))


class FakeCapture:
    """Yields `n` frames, then fails as if the stream dropped."""
    def __init__(self, n: int, fail_forever: bool = False):
        self.n, self.i, self.fail_forever = n, 0, fail_forever
        self.released = False

    def isOpened(self):
        return not self.fail_forever

    def grab(self):
        return self.read()[0]

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((48, 64, 3), self.i % 256, dtype=np.uint8)

    def release(self):
        self.released = True

    def set(self, *_):
        return True

    def get(self, *_):
        return 0.0


def test_rtsp_reconnects_after_the_stream_drops():
    attempts = []

    def factory(url):
        attempts.append(url)
        return FakeCapture(n=5) if len(attempts) <= 2 else FakeCapture(0, fail_forever=True)

    src = RtspSource("rtsp://x/cam1", sample_fps=10, capture_factory=factory,
                     max_backoff_s=0.0, max_reconnects=3)
    frames = list(src)
    assert len(frames) == 10          # 5 from each of the two good captures
    assert src.disconnects >= 1
    assert len(attempts) >= 3


def test_rtsp_timestamps_are_arrival_times_and_increase():
    src = RtspSource("rtsp://x/cam1", sample_fps=10,
                     capture_factory=lambda url: FakeCapture(n=5),
                     max_backoff_s=0.0, max_reconnects=1)
    times = [t for t, _ in src]
    assert times == sorted(times)
    assert all(t.tzinfo is not None for t in times)


def test_rtsp_releases_the_capture_on_exit():
    caps = []

    def factory(url):
        c = FakeCapture(n=3)
        caps.append(c)
        return c

    list(RtspSource("rtsp://x/cam1", sample_fps=10, capture_factory=factory,
                    max_backoff_s=0.0, max_reconnects=1))
    assert all(c.released for c in caps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.ingest'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/ingest.py
"""Frame sources: recorded files and live RTSP streams."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Low-latency RTSP: prefer TCP, keep the decoder queue short.
RTSP_OPTIONS = "rtsp_transport;tcp|max_delay;500000|reorder_queue_size;0"


def _require_aware(t: datetime) -> None:
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise ValueError("start_at must be timezone-aware")


class FileSource:
    """Recorded video. Timestamps come from the file's start time plus the
    frame's position, so a clip can be placed on the real-world clock."""

    def __init__(self, path: str | Path, start_at: datetime, sample_fps: float) -> None:
        _require_aware(start_at)
        self.path = Path(path)
        self.start_at = start_at
        self.sample_fps = float(sample_fps)

    def __iter__(self) -> Iterator[tuple[datetime, np.ndarray]]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {self.path}")
        try:
            native = cap.get(cv2.CAP_PROP_FPS) or self.sample_fps
            stride = max(int(round(native / self.sample_fps)), 1)
            index = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if index % stride == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    yield self.start_at + timedelta(seconds=index / native), frame
                index += 1
        finally:
            cap.release()


class RtspSource:
    """Live stream. Only the newest frame matters: OpenCV will otherwise hand
    over frames seconds stale while the decoder drains its backlog, which
    would inflate every timestamp and corrupt the timing check.

    Timestamps are arrival times. Drops trigger reconnection with backoff.
    """

    def __init__(self, url: str, sample_fps: float,
                 capture_factory: Callable[[str], object] | None = None,
                 max_backoff_s: float = 8.0, max_reconnects: int | None = None) -> None:
        self.url = url
        self.sample_fps = float(sample_fps)
        self.max_backoff_s = max_backoff_s
        self.max_reconnects = max_reconnects
        self.disconnects = 0
        self._factory = capture_factory or self._default_factory

    @staticmethod
    def _default_factory(url: str):
        import os
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_OPTIONS)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def __iter__(self) -> Iterator[tuple[datetime, np.ndarray]]:
        interval = 1.0 / self.sample_fps
        attempts = 0
        backoff = 0.5

        while self.max_reconnects is None or attempts < self.max_reconnects:
            attempts += 1
            cap = self._factory(self.url)
            try:
                if not cap.isOpened():
                    log.warning("cannot open %s (attempt %d)", self.url, attempts)
                else:
                    next_at = 0.0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            self.disconnects += 1
                            log.warning("stream %s dropped", self.url)
                            break
                        now = time.monotonic()
                        if now < next_at:
                            continue            # discard: keeps only fresh frames
                        next_at = now + interval
                        yield datetime.now(timezone.utc), frame
                    backoff = 0.5
            finally:
                cap.release()

            if self.max_backoff_s > 0:
                time.sleep(min(backoff, self.max_backoff_s))
                backoff *= 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_ingest.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add tracer/ingest.py tests/test_ingest.py
git commit -m "feat: file and RTSP frame sources with staleness discard and reconnection"
```

---

### Task 14: Worker pipeline and file ingest CLI

**Files:**
- Create: `tracer/worker.py`, `scripts/ingest_file.py`
- Modify: `tracer/clips.py` (add `ClipRecorder`)
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `FileSource`/`RtspSource` (Task 13), `TrackletBuilder`/`TrackletMerger`/`yolo_detections` (Task 11), `OsnetEmbedder` (Task 10), `FrameBuffer`/`write_thumbnail` (Task 12), `Store` (Task 4).
- Produces: `ClipRecorder(path, fps, preroll)` with `add(frame)`, `close() -> Path`, `frame_count`; `Pipeline(cfg, camera_id, store, embedder, detector)` with `run(source) -> int`; `YoloDetector(cfg)` callable as `(frame, t) -> list[Detection]`.

**Why `ClipRecorder` rather than buffering:** a 60-second tracklet at 720p/10 fps is ~1.6 GB of frames. The recorder streams frames to disk as they arrive, taking only a short pre-roll from the `FrameBuffer`. Post-roll comes free: the tracklet does not close until the track has been lost for `lost_track_s` (2 s), so the clip already ends ~2 s after the person left — a superset of the spec's `end + 1 s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker.py
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.clips import ClipRecorder
from tracer.config import CameraCfg, Config
from tracer.detect_track import Detection
from tracer.worker import Pipeline

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file")},
        lost_track_s=2.0, min_tracklet_s=1.0, max_tracklet_s=60.0,
        merge_gap_s=3.0, merge_sim=0.8,
    )


class FakeEmbedder:
    def embed_tracklet(self, crops):
        from tracer.models import unit_vector
        rng = np.random.default_rng(len(crops))
        return unit_vector(rng.standard_normal(512).astype(np.float32)), min(len(crops), 10)


def source(n: int, fps: float = 10.0):
    """A synthetic source: `n` frames at `fps`, timestamps on the wall clock."""
    for i in range(n):
        yield T0 + timedelta(seconds=i / fps), np.full((120, 80, 3), i % 256, dtype=np.uint8)


def detector_present_for(frames: int):
    """A detector that sees one person for the first `frames` frames."""
    seen = {"n": 0}

    def detect(frame, t):
        seen["n"] += 1
        if seen["n"] > frames:
            return []
        return [Detection(track_id=1, xyxy=(5, 5, 55, 105), edge_cut=False,
                          crop=np.full((100, 50, 3), 128, dtype=np.uint8))]
    return detect


def test_a_clean_pass_writes_one_tracklet_with_clip_and_thumbnail(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    written = p.run(source(60))
    assert written == 1

    rows = store.conn.execute("SELECT id, camera_id, thumb_path, clip_path, n_crops "
                              "FROM tracklet").fetchall()
    assert len(rows) == 1
    tid, cam, thumb, clip, n_crops = rows[0]
    assert cam == "cam1"
    assert (cfg.data_dir / thumb).exists()
    assert (cfg.data_dir / clip).exists()
    assert 1 <= n_crops <= 10


def test_tracklet_times_match_when_the_person_was_visible(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    p.run(source(60))
    started, ended = store.conn.execute(
        "SELECT started_at, ended_at FROM tracklet").fetchone()
    assert started == T0
    assert ended == pytest.approx(T0 + timedelta(seconds=2.9),
                                  abs=timedelta(milliseconds=150))


def test_a_person_visible_too_briefly_writes_nothing(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(4))
    assert p.run(source(60)) == 0
    assert store.conn.execute("SELECT count(*) FROM tracklet").fetchone()[0] == 0


def test_the_source_ending_mid_track_still_writes_the_tracklet(store, cfg):
    # Flush on exhaustion: otherwise the last pass of a recorded file is lost.
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(100))
    assert p.run(source(40)) == 1


def test_the_recorder_writes_a_playable_clip(tmp_path):
    rec = ClipRecorder(tmp_path / "c.mp4", fps=10.0,
                       preroll=[(T0, np.zeros((48, 64, 3), np.uint8))])
    for i in range(20):
        rec.add(np.full((48, 64, 3), i, dtype=np.uint8))
    out = rec.close()
    assert out.exists() and rec.frame_count == 21


def test_the_recorder_tolerates_being_closed_twice(tmp_path):
    rec = ClipRecorder(tmp_path / "c.mp4", fps=10.0, preroll=[])
    rec.add(np.zeros((48, 64, 3), np.uint8))
    rec.close()
    assert rec.close().exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_worker.py -v`
Expected: FAIL with `ImportError: cannot import name 'ClipRecorder'`

- [ ] **Step 3: Add `ClipRecorder` to `tracer/clips.py`**

```python
# append to tracer/clips.py
class ClipRecorder:
    """Streams frames to an mp4 as they arrive, so memory stays flat however
    long the tracklet runs. Pre-roll frames come from a FrameBuffer."""

    def __init__(self, path: str | Path, fps: float,
                 preroll: Sequence[tuple[datetime, np.ndarray]] = ()) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(fps, 1.0)
        self._writer: cv2.VideoWriter | None = None
        self.frame_count = 0
        for _, frame in preroll:
            self.add(frame)

    def add(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self._writer.write(frame)
        self.frame_count += 1

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self.path
```

- [ ] **Step 4: Write the pipeline**

```python
# tracer/worker.py
"""One worker per camera: frames in, tracklets out."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Callable, Iterable, Sequence

import numpy as np

from tracer.clips import ClipRecorder, FrameBuffer, write_thumbnail
from tracer.config import Config
from tracer.detect_track import (Detection, RawTracklet, TrackletBuilder,
                                 TrackletMerger, yolo_detections)
from tracer.models import TrackletRecord
from tracer.store import Store

log = logging.getLogger(__name__)

PREROLL_S = 1.5

Detector = Callable[[np.ndarray, datetime], Sequence[Detection]]


class YoloDetector:
    """YOLO26 + ByteTrack. One instance per camera: Ultralytics keeps tracker
    state inside the model, so sharing it across cameras mixes track IDs."""

    def __init__(self, cfg: Config, weights: str = "yolo26n.pt", conf: float = 0.5) -> None:
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.cfg = cfg
        self.conf = conf
        self.device = os.environ.get("DEVICE", cfg.device)

    def __call__(self, frame: np.ndarray, t: datetime) -> list[Detection]:
        result = self.model.track(
            frame, classes=[0], conf=self.conf, imgsz=self.cfg.imgsz,
            tracker="bytetrack.yaml", persist=True, verbose=False,
            device=0 if self.device in ("cuda", "gpu") else "cpu",
        )[0]
        return yolo_detections(result, frame, t)


class Pipeline:
    def __init__(self, cfg: Config, camera_id: str, store: Store,
                 embedder, detector: Detector) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self.store = store
        self.embedder = embedder
        self.detector = detector
        self.merger = TrackletMerger(cfg, store)
        self.buffer = FrameBuffer(seconds=PREROLL_S)
        self.written = 0
        self._recorders: dict[int, ClipRecorder] = {}
        self._builder = TrackletBuilder(cfg, camera_id, self._on_close)

    def run(self, source: Iterable[tuple[datetime, np.ndarray]]) -> int:
        last_t: datetime | None = None
        for t, frame in source:
            last_t = t
            self.buffer.add(t, frame)
            detections = list(self.detector(frame, t))
            self._open_recorders(detections, t)
            self._builder.add(t, detections)
            for rec in self._recorders.values():
                rec.add(frame)
        if last_t is not None:
            self._builder.flush(last_t)
        return self.written

    def _open_recorders(self, detections: Sequence[Detection], t: datetime) -> None:
        for d in detections:
            if d.track_id not in self._recorders:
                path = self.cfg.data_dir / "clips" / f"{self.camera_id}-{int(t.timestamp()*1000)}-{d.track_id}.mp4"
                self._recorders[d.track_id] = ClipRecorder(
                    path, fps=self.cfg.sample_fps,
                    preroll=self.buffer.window(t, t))

    def _on_close(self, raw: RawTracklet) -> None:
        recorder = self._recorders.pop(raw.track_id, None)
        clip_path = recorder.close() if recorder else None
        try:
            embedding, n_crops = self.embedder.embed_tracklet(raw.crops)
        except ValueError as exc:
            log.warning("skipping tracklet on %s: %s", raw.camera_id, exc)
            return

        rel_clip = str(clip_path.relative_to(self.cfg.data_dir)) if clip_path else ""
        thumb_rel = f"thumbs/{raw.camera_id}-{int(raw.started_at.timestamp()*1000)}-{raw.track_id}.jpg"
        write_thumbnail(raw.crops, self.cfg.data_dir / thumb_rel)

        record = TrackletRecord(
            camera_id=raw.camera_id, started_at=raw.started_at, ended_at=raw.ended_at,
            embedding=embedding, thumb_path=thumb_rel, clip_path=rel_clip,
            n_crops=n_crops,
        )
        action, tracklet_id = self.merger.offer(record)
        if action == "new":
            self.written += 1
        log.info("%s tracklet %s on %s (%.1fs, %d crops)", action, tracklet_id,
                 raw.camera_id, raw.duration_s, n_crops)
```

- [ ] **Step 5: Write the ingest CLI**

```python
# scripts/ingest_file.py
"""Ingest a recorded video file as one camera.

  python -m scripts.ingest_file clip.mp4 --camera cam1 \
      --start 2026-10-13T14:00:00+07:00
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from tracer.deps import get_config, get_store
from tracer.ingest import FileSource
from tracer.reid import OsnetEmbedder
from tracer.worker import Pipeline, YoloDetector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--start", required=True,
                    help="ISO 8601 with an offset, e.g. 2026-10-13T14:00:00+07:00")
    ap.add_argument("--weights", default="data/weights/osnet_x1_0_msmt17.pth")
    ap.add_argument("--protected", action="store_true",
                    help="exempt these tracklets from retention (evaluation footage)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = datetime.fromisoformat(args.start)
    if start.tzinfo is None:
        raise SystemExit("--start must include a UTC offset, e.g. +07:00")

    cfg, store = get_config(), get_store()
    pipeline = Pipeline(cfg, args.camera, store,
                        OsnetEmbedder(args.weights, device=cfg.device),
                        YoloDetector(cfg))
    written = pipeline.run(FileSource(args.video, start, cfg.sample_fps))
    print(f"wrote {written} tracklet(s) for {args.camera}")

    if args.protected:
        store.conn.execute(
            "UPDATE tracklet SET protected = true WHERE camera_id = %s", (args.camera,))
        print(f"marked {args.camera} tracklets as protected")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_worker.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 7: Run a real ingest end to end**

```bash
mkdir -p data/samples
curl -sL -o data/samples/people.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
./scripts/fetch_weights.sh data/weights
docker compose run --rm app python -m scripts.ingest_file data/samples/people.mp4 \
  --camera cam1 --start 2026-10-13T14:00:00+00:00
```

Expected: `wrote N tracklet(s) for cam1` with N ≥ 1, and files under `data/clips/` and `data/thumbs/`.

- [ ] **Step 8: Commit**

```bash
git add tracer/worker.py tracer/clips.py scripts/ingest_file.py tests/test_worker.py
git commit -m "feat: per-camera worker pipeline with streaming clip recorder and file CLI"
```

---

### Task 15: Reference-photo embedding and search CLI

**Files:**
- Modify: `tracer/reid.py` (add `PhotoEmbedder`)
- Create: `scripts/search_cli.py`
- Test: `tests/test_photo_embedder.py`

**Interfaces:**
- Consumes: `OsnetEmbedder` (Task 10), `Config` (Task 2).
- Produces: `PhotoEmbedder(cfg, weights=None, detector=None, embedder=None)` with `embed_image(data: bytes) -> tuple[np.ndarray, list[dict]]`, where each detection dict has `xyxy` and `area`, sorted largest first, and the returned embedding is of the largest detection. Raises `ValueError` when no person is found.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_photo_embedder.py
import numpy as np
import pytest

from tracer.config import CameraCfg, Config
from tracer.models import unit_vector
from tracer.reid import PhotoEmbedder


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.78, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file")})


def jpeg_bytes(h=400, w=300) -> bytes:
    import cv2
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class FakeDetections:
    """Mimics one Ultralytics result with `n` person boxes of varying size."""
    def __init__(self, n):
        self.n = n

    @property
    def boxes(self):
        if self.n == 0:
            return None
        import torch

        class B:
            xyxy = torch.tensor([[10.0, 10.0, 60.0, 10.0 + 100 * (i + 1)]
                                 for i in range(self.n)])
            id = torch.tensor([float(i + 1) for i in range(self.n)])
        return B()


class FakeEmbedder:
    def embed_images(self, images):
        return np.stack([unit_vector(np.full(512, float(i + 1), dtype=np.float32))
                         for i in range(len(images))])


def make(cfg, n_people: int) -> PhotoEmbedder:
    return PhotoEmbedder(cfg, detector=lambda img: [FakeDetections(n_people)],
                         embedder=FakeEmbedder())


def test_one_person_gives_one_detection_and_a_unit_embedding(cfg):
    emb, dets = make(cfg, 1).embed_image(jpeg_bytes())
    assert len(dets) == 1
    assert emb.shape == (512,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_several_people_are_returned_largest_first(cfg):
    # Review Focus #1: the reviewer must be able to pick the right person.
    emb, dets = make(cfg, 3).embed_image(jpeg_bytes())
    areas = [d["area"] for d in dets]
    assert areas == sorted(areas, reverse=True)
    assert len(dets) == 3
    assert all("xyxy" in d for d in dets)


def test_no_person_raises_a_readable_error(cfg):
    # Review Focus #1: the reviewer uploaded a photo with no person in it.
    with pytest.raises(ValueError, match="no person"):
        make(cfg, 0).embed_image(jpeg_bytes())


def test_undecodable_image_data_raises(cfg):
    with pytest.raises(ValueError, match="could not read"):
        make(cfg, 1).embed_image(b"this is not an image")


def test_the_embedding_belongs_to_the_largest_detection(cfg):
    # The third fake box is the tallest; its embedding is the constant-3 vector.
    emb, dets = make(cfg, 3).embed_image(jpeg_bytes())
    expected = unit_vector(np.full(512, 3.0, dtype=np.float32))
    assert np.allclose(emb, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_photo_embedder.py -v`
Expected: FAIL with `ImportError: cannot import name 'PhotoEmbedder'`

- [ ] **Step 3: Write the implementation**

```python
# append to tracer/reid.py
class PhotoEmbedder:
    """Embeds a reference photo through the *same* crop pipeline as ingest.

    A query embedded differently from the gallery is the classic silent
    re-ID bug: similarities stay plausible while ranking quietly degrades.
    """

    def __init__(self, cfg, weights: str | None = None,
                 detector=None, embedder=None) -> None:
        self.cfg = cfg
        if detector is not None:
            self._detect = detector
        else:
            from ultralytics import YOLO
            model = YOLO("yolo26n.pt")
            device = 0 if str(cfg.device) in ("cuda", "gpu") else "cpu"
            self._detect = lambda img: model.predict(
                img, classes=[0], conf=0.5, imgsz=cfg.imgsz,
                device=device, verbose=False)
        self.embedder = embedder or OsnetEmbedder(
            weights or "data/weights/osnet_x1_0_msmt17.pth", device=str(cfg.device))

    def embed_image(self, data: bytes) -> tuple[np.ndarray, list[dict]]:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not read that image file")

        result = self._detect(image)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes.xyxy) == 0:
            raise ValueError("no person detected in the reference photo")

        h, w = image.shape[:2]
        found: list[dict] = []
        crops: list[np.ndarray] = []
        for xyxy in boxes.xyxy.tolist():
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            found.append({"xyxy": [x1, y1, x2, y2], "area": (x2 - x1) * (y2 - y1)})
            crops.append(image[y1:y2, x1:x2, ::-1].copy())       # BGR -> RGB

        if not found:
            raise ValueError("no person detected in the reference photo")

        order = sorted(range(len(found)), key=lambda i: found[i]["area"], reverse=True)
        rows = self.embedder.embed_images([crops[i] for i in order])
        return unit_vector(np.asarray(rows)[0]), [found[i] for i in order]
```

- [ ] **Step 4: Write the search CLI**

```python
# scripts/search_cli.py
"""Search for a person from a reference photo.

  python -m scripts.search_cli ref.jpg --cameras cam1 cam2 --limit 10
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tracer.deps import get_config, get_store
from tracer.reid import PhotoEmbedder
from tracer.search import run_search

REASONS = {
    "no_tracklets_in_scope": "no sightings at all in that window and camera set",
    "none_above_tau": "no sighting passed the similarity threshold",
    "all_rejected_by_timing": "candidates found, but all impossible in time",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photo")
    ap.add_argument("--cameras", nargs="+", default=None)
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    cfg, store = get_config(), get_store()
    cameras = args.cameras or list(cfg.cameras)
    window = (datetime.fromisoformat(args.frm) if args.frm else None,
              datetime.fromisoformat(args.to) if args.to else None)

    embedding, detections = PhotoEmbedder(cfg).embed_image(Path(args.photo).read_bytes())
    print(f"{len(detections)} person(s) detected; using the largest "
          f"({detections[0]['area']} px²)")

    result = run_search(store, cfg, embedding, "osnet", window, cameras, args.limit)
    if not result.matches:
        best = f" (best score {result.best_score:.3f})" if result.best_score else ""
        print(f"no matches: {REASONS.get(result.reason, result.reason)}{best}")
        return

    print(f"{'rank':<5}{'camera':<10}{'started':<22}{'score':<8}tracklet")
    for i, m in enumerate(result.matches, 1):
        print(f"{i:<5}{m.camera_id:<10}"
              f"{m.started_at.isoformat(timespec='seconds'):<22}"
              f"{m.score:<8.3f}{m.tracklet_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm app pytest tests/test_photo_embedder.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Commit**

```bash
git add tracer/reid.py scripts/search_cli.py tests/test_photo_embedder.py
git commit -m "feat: reference-photo embedding with detection choice and search CLI"
```

---

### Task 16: MVP acceptance — cross-camera search on recorded clips

**Files:**
- Create: `tests/test_mvp_acceptance.py`, `scripts/make_two_camera_demo.sh`
- Test: the same file

**Interfaces:**
- Consumes: everything from Tasks 1–15.
- Produces: `tests/test_mvp_acceptance.py::test_mvp_acceptance`, the executable form of the spec's MVP criterion.

This is decision #3's smoke test: one clip ingested twice under two camera IDs with different start times. The same person therefore genuinely appears on both cameras, so a true cross-camera match exists and must rank first. Because the pixels are identical, failure means something is broken rather than merely inaccurate.

- [ ] **Step 1: Write the demo script**

```bash
# scripts/make_two_camera_demo.sh
#!/usr/bin/env bash
# Ingest one clip as two cameras with a plausible walk gap between them, so
# cross-camera search can be exercised before any real two-camera footage.
set -euo pipefail
CLIP="${1:-data/samples/people.mp4}"
BASE="${2:-2026-10-13T14:00:00+00:00}"
OFFSET_S="${3:-600}"

python3 - "$BASE" "$OFFSET_S" <<'PY' > /tmp/cam2_start
import sys, datetime
base = datetime.datetime.fromisoformat(sys.argv[1])
print((base + datetime.timedelta(seconds=int(sys.argv[2]))).isoformat())
PY
CAM2_START="$(cat /tmp/cam2_start)"

echo "cam1 at $BASE"
python -m scripts.ingest_file "$CLIP" --camera cam1 --start "$BASE"
echo "cam2 at $CAM2_START"
python -m scripts.ingest_file "$CLIP" --camera cam2 --start "$CAM2_START"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_mvp_acceptance.py
"""MVP acceptance (spec section 'Acceptance'): a photo of person A returns
A's sightings from both camera spots at the top, in correct time order."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracer.config import CameraCfg, Config, PairCfg
from tracer.ingest import FileSource
from tracer.reid import OsnetEmbedder, PhotoEmbedder
from tracer.search import run_search
from tracer.worker import Pipeline, YoloDetector

CLIP = Path("data/samples/people.mp4")
WEIGHTS = Path("data/weights/osnet_x1_0_msmt17.pth")
T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
CAM2_OFFSET = timedelta(seconds=600)

pytestmark = pytest.mark.skipif(
    not (CLIP.exists() and WEIGHTS.exists()),
    reason="needs data/samples/people.mp4 and OSNet weights; see Task 14 step 7")


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path, device=os.environ.get("DEVICE", "cpu"),
        sample_fps=10, imgsz=640, retention_days=3,
        tau={"osnet": 0.78, "clip": 0.25}, default_min_walk_s=10.0,
        cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file"),
                 "cam2": CameraCfg(id="cam2", name="cam2", kind="file")},
        pairs={frozenset(("cam1", "cam2")): PairCfg(min_walk_s=4.0)},
    )


@pytest.fixture
def ingested(store, cfg):
    """One clip as two cameras, ten minutes apart."""
    embedder = OsnetEmbedder(str(WEIGHTS), device=str(cfg.device))
    for camera, start in (("cam1", T0), ("cam2", T0 + CAM2_OFFSET)):
        Pipeline(cfg, camera, store, embedder, YoloDetector(cfg)).run(
            FileSource(CLIP, start, cfg.sample_fps))
    return store


def reference_photo(store, cfg) -> bytes:
    """Use the thumbnail of the best cam1 tracklet as the reference photo,
    standing in for a separately taken photo of the same person."""
    thumb = store.conn.execute(
        "SELECT thumb_path FROM tracklet WHERE camera_id = 'cam1' "
        "ORDER BY ended_at - started_at DESC LIMIT 1").fetchone()[0]
    img = cv2.imread(str(cfg.data_dir / thumb))
    # Pad the crop so the detector has a full frame to find a person in.
    canvas = np.full((img.shape[0] + 80, img.shape[1] + 80, 3), 60, dtype=np.uint8)
    canvas[40:40 + img.shape[0], 40:40 + img.shape[1]] = img
    ok, buf = cv2.imencode(".jpg", canvas)
    assert ok
    return buf.tobytes()


def test_both_cameras_produced_tracklets(ingested):
    rows = dict(ingested.conn.execute(
        "SELECT camera_id, count(*) FROM tracklet GROUP BY camera_id").fetchall())
    assert rows.get("cam1", 0) >= 1
    assert rows.get("cam2", 0) >= 1


def test_mvp_acceptance(ingested, cfg):
    """A photo returns sightings from both cameras, best first, time-ordered."""
    embedding, detections = PhotoEmbedder(
        cfg, weights=str(WEIGHTS)).embed_image(reference_photo(ingested, cfg))
    assert detections, "the reference photo must contain a detectable person"

    result = run_search(ingested, cfg, embedding, "osnet",
                        (T0 - timedelta(hours=1), T0 + timedelta(hours=2)),
                        ["cam1", "cam2"], limit=10)

    assert result.matches, f"no matches: {result.reason} (best {result.best_score})"
    cameras = {m.camera_id for m in result.matches}
    assert cameras == {"cam1", "cam2"}, f"expected both cameras, got {cameras}"

    scores = [m.score for m in result.matches]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] >= cfg.tau["osnet"]


def test_results_can_be_placed_in_time_order_across_cameras(ingested, cfg):
    embedding, _ = PhotoEmbedder(
        cfg, weights=str(WEIGHTS)).embed_image(reference_photo(ingested, cfg))
    result = run_search(ingested, cfg, embedding, "osnet",
                        (T0 - timedelta(hours=1), T0 + timedelta(hours=2)),
                        ["cam1", "cam2"], limit=10)
    in_time_order = sorted(result.matches, key=lambda m: m.started_at)
    assert in_time_order[0].camera_id == "cam1"
    assert in_time_order[-1].camera_id == "cam2"


def test_the_timing_check_rejects_a_physically_impossible_pair(ingested, store, cfg):
    """With cam2 shifted to overlap cam1, the two sightings become impossible
    and the filter must drop one of them."""
    store.conn.execute(
        "UPDATE tracklet SET started_at = started_at - %s::interval, "
        "ended_at = ended_at - %s::interval WHERE camera_id = 'cam2'",
        ("600 seconds", "600 seconds"))
    embedding, _ = PhotoEmbedder(
        cfg, weights=str(WEIGHTS)).embed_image(reference_photo(store, cfg))
    result = run_search(store, cfg, embedding, "osnet",
                        (T0 - timedelta(hours=1), T0 + timedelta(hours=2)),
                        ["cam1", "cam2"], limit=10)
    for a in result.matches:
        for b in result.matches:
            if a.tracklet_id != b.tracklet_id and a.camera_id != b.camera_id:
                gap = abs((b.started_at - a.ended_at).total_seconds())
                assert gap >= cfg.min_walk(a.camera_id, b.camera_id)
```

- [ ] **Step 3: Run the acceptance test**

Run: `docker compose run --rm app pytest tests/test_mvp_acceptance.py -v`
Expected: PASS, 4 tests. If `test_mvp_acceptance` fails on the camera-set assertion, the τ value is too high for this footage — record the real best score before changing τ, and note it for the Phase 4 tuning.

- [ ] **Step 4: Run the whole suite**

Run: `docker compose run --rm app pytest -v`
Expected: PASS, every test from Tasks 1–16.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/make_two_camera_demo.sh
git add tests/test_mvp_acceptance.py scripts/make_two_camera_demo.sh
git commit -m "test: MVP acceptance — cross-camera search with timing check on recorded clips"
```

**Phase 2 checkpoint (MVP complete):** R1–R6 are implemented and the acceptance criterion is an executable test. The UI serves ranked results with clips.

---

# Phase 3 — Live cameras and timeline

### Task 17: MediaMTX ingest hub, replay harness, Pi camera node

**Files:**
- Create: `mediamtx/pepper.yml`, `pi/mediamtx.yml`, `pi/setup.sh`
- Modify: `docker-compose.yml` (add the `mediamtx` service)
- Test: `tests/test_rtsp_integration.py`

**Interfaces:**
- Consumes: `RtspSource` (Task 13).
- Produces: `rtsp://mediamtx:8554/<cam_id>` inside compose and `rtsp://localhost:8554/<cam_id>` on the host, for every source regardless of the protocol it speaks.

**Why a hub:** IP Webcam is Android-only and iOS RTSP publishing is unreliable, but Larix pushes SRT dependably. MediaMTX accepts RTSP, SRT and RTMP and republishes all of them as RTSP, so workers only ever speak one protocol and a dead phone can be swapped for a replayed file by editing config.

- [ ] **Step 1: Write the MediaMTX config for pepper**

```yaml
# mediamtx/pepper.yml
logLevel: info
rtspAddress: :8554
rtmpAddress: :1935
srtAddress: :8890
hlsDisable: yes
webrtcDisable: yes

paths:
  # Pulled from the Pi camera node.
  cam1:
    source: rtsp://raspberrypi.local:8554/cam1
    sourceOnDemand: no

  # Pushed by the iPhone running Larix:
  #   SRT URL: srt://<pepper-ip>:8890?streamid=publish:cam2
  # Larix publishes SRT reliably where iOS RTSP publishing does not.
  cam2: {}

  # Demo-night fallback and the development harness: a recorded clip
  # re-streamed in real time, so the live path can be exercised with no
  # camera attached.
  replay:
    runOnInit: >
      ffmpeg -re -stream_loop -1 -i /samples/people.mp4
      -c:v libx264 -preset veryfast -tune zerolatency -g 30
      -f rtsp rtsp://localhost:8554/replay
    runOnInitRestart: yes
```

- [ ] **Step 2: Add the service to compose**

```yaml
# add to docker-compose.yml under services:
  mediamtx:
    image: bluenviron/mediamtx:latest
    network_mode: host
    volumes:
      - ./mediamtx/pepper.yml:/mediamtx.yml:ro
      - ./data/samples:/samples:ro
    restart: unless-stopped
```

`network_mode: host` keeps RTSP/SRT port handling simple and lets the Pi and the phone reach it directly. Workers then use `rtsp://host.docker.internal:8554/<cam>` or the host IP; set `source_url` in `config.yaml` accordingly.

- [ ] **Step 3: Write the Pi camera node config and setup**

```yaml
# pi/mediamtx.yml
logLevel: info
rtspAddress: :8554
hlsDisable: yes
webrtcDisable: yes
paths:
  cam1:
    # 720p is plenty: re-ID works on person crops, not full frames.
    # Use '-c:v copy' instead if the webcam already outputs H.264.
    runOnInit: >
      ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30
      -i /dev/video0 -c:v h264_v4l2m2m -b:v 3M
      -f rtsp rtsp://localhost:8554/cam1
    runOnInitRestart: yes
```

```bash
# pi/setup.sh
#!/usr/bin/env bash
# Install MediaMTX and ffmpeg on the Pi 4 and stream the USB webcam at boot.
set -euo pipefail
VERSION="${MEDIAMTX_VERSION:-v1.9.3}"
ARCH=arm64v8

sudo apt-get update
sudo apt-get install -y ffmpeg v4l-utils curl

echo "cameras seen by the Pi:"
v4l2-ctl --list-devices || true

tmp="$(mktemp -d)"
curl -fsSL -o "$tmp/m.tar.gz" \
  "https://github.com/bluenviron/mediamtx/releases/download/${VERSION}/mediamtx_${VERSION}_linux_${ARCH}.tar.gz"
tar -xzf "$tmp/m.tar.gz" -C "$tmp"
sudo install -m 755 "$tmp/mediamtx" /usr/local/bin/mediamtx
sudo install -m 644 "$(dirname "$0")/mediamtx.yml" /usr/local/etc/mediamtx.yml

sudo tee /etc/systemd/system/mediamtx.service >/dev/null <<'UNIT'
[Unit]
Description=MediaMTX camera node
After=network-online.target
[Service]
ExecStart=/usr/local/bin/mediamtx /usr/local/etc/mediamtx.yml
Restart=always
RestartSec=2
[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx
sleep 3
systemctl is-active mediamtx
echo "stream should be at rtsp://$(hostname -I | awk '{print $1}'):8554/cam1"
```

- [ ] **Step 4: Write the integration test**

```python
# tests/test_rtsp_integration.py
"""Exercises the real RTSP path. Skipped unless MediaMTX is reachable."""
from __future__ import annotations

import socket
import time

import pytest

from tracer.ingest import RtspSource

RTSP_HOST = "127.0.0.1"
RTSP_PORT = 8554
REPLAY_URL = f"rtsp://{RTSP_HOST}:{RTSP_PORT}/replay"


def _reachable() -> bool:
    try:
        with socket.create_connection((RTSP_HOST, RTSP_PORT), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason="MediaMTX not running; start it with 'docker compose up -d mediamtx'")


def test_frames_arrive_from_the_replay_stream():
    src = RtspSource(REPLAY_URL, sample_fps=10, max_reconnects=1, max_backoff_s=0.0)
    frames = []
    for t, frame in src:
        frames.append((t, frame))
        if len(frames) >= 20:
            break
    assert len(frames) == 20
    assert frames[0][1].shape[0] > 0


def test_arrival_timestamps_track_real_time():
    """If OpenCV were handing over a stale decoder backlog, 20 frames at
    10 fps would arrive far faster than the two seconds they represent."""
    src = RtspSource(REPLAY_URL, sample_fps=10, max_reconnects=1, max_backoff_s=0.0)
    started = time.monotonic()
    times = []
    for t, _ in src:
        times.append(t)
        if len(times) >= 20:
            break
    elapsed = time.monotonic() - started
    assert elapsed > 1.0, f"20 frames at 10 fps arrived in {elapsed:.2f}s — stale buffer"
    spread = (times[-1] - times[0]).total_seconds()
    assert spread == pytest.approx(elapsed, abs=0.5)
```

- [ ] **Step 5: Start MediaMTX and run the test**

```bash
mkdir -p data/samples
test -s data/samples/people.mp4 || curl -sL -o data/samples/people.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
docker compose up -d mediamtx
sleep 5
docker compose logs --tail=20 mediamtx
docker compose run --rm --network host app pytest tests/test_rtsp_integration.py -v
```

Expected: PASS, 2 tests. If the replay path is missing from the logs, check that `data/samples/people.mp4` exists — `runOnInit` fails silently otherwise.

- [ ] **Step 6: Commit**

```bash
chmod +x pi/setup.sh
git add mediamtx/ pi/ docker-compose.yml tests/test_rtsp_integration.py
git commit -m "feat: MediaMTX ingest hub, file-replay harness and Pi camera node setup"
```

---

### Task 18: Live worker and R8 latency measurement

**Files:**
- Modify: `tracer/worker.py` (add latency metrics and a `__main__` entrypoint)
- Create: `scripts/report_latency.py`
- Modify: `docker-compose.yml` (add `worker-cam1`, `worker-cam2`)
- Test: `tests/test_latency.py`

**Interfaces:**
- Consumes: `Pipeline` (Task 14), `RtspSource` (Task 13).
- Produces: `Pipeline.latencies: list[float]`; a JSONL metrics file at `data/metrics/latency.jsonl` with keys `camera_id`, `tracklet_id`, `last_seen`, `searchable_at`, `latency_s`; `python -m tracer.worker --camera cam1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latency.py
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tracer.config import CameraCfg, Config
from tracer.detect_track import Detection
from tracer.worker import Pipeline

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.78, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="rtsp")},
                  lost_track_s=2.0, min_tracklet_s=1.0, max_tracklet_s=60.0,
                  merge_gap_s=3.0, merge_sim=0.8)


class FakeEmbedder:
    def embed_tracklet(self, crops):
        from tracer.models import unit_vector
        return unit_vector(np.ones(512, dtype=np.float32)), min(len(crops), 10)


def live_source(n: int, fps: float = 10.0):
    """Frames timestamped at *now*, as a live stream produces them."""
    base = datetime.now(timezone.utc)
    for i in range(n):
        yield base + timedelta(seconds=i / fps), np.full((120, 80, 3), i % 256, np.uint8)


def detector_present_for(frames: int):
    seen = {"n": 0}

    def detect(frame, t):
        seen["n"] += 1
        if seen["n"] > frames:
            return []
        return [Detection(track_id=1, xyxy=(5, 5, 55, 105), edge_cut=False,
                          crop=np.full((100, 50, 3), 128, dtype=np.uint8))]
    return detect


def test_latency_is_recorded_for_each_tracklet(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    p.run(live_source(60))
    assert len(p.latencies) == 1
    assert p.latencies[0] >= 0.0


def test_latency_meets_the_ten_second_requirement(store, cfg):
    # R8: a sighting must be searchable within 10s of the person leaving frame.
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    p.run(live_source(60))
    assert max(p.latencies) < 10.0


def test_a_metrics_line_is_written_per_tracklet(store, cfg):
    p = Pipeline(cfg, "cam1", store, FakeEmbedder(), detector_present_for(30))
    p.run(live_source(60))
    lines = (cfg.data_dir / "metrics" / "latency.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["camera_id"] == "cam1"
    assert row["tracklet_id"] > 0
    assert "last_seen" in row and "searchable_at" in row
    assert row["latency_s"] == pytest.approx(p.latencies[0], abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_latency.py -v`
Expected: FAIL with `AttributeError: 'Pipeline' object has no attribute 'latencies'`

- [ ] **Step 3: Add metrics to the pipeline**

```python
# in tracer/worker.py: add to Pipeline.__init__
        self.latencies: list[float] = []
        self._metrics_path = cfg.data_dir / "metrics" / "latency.jsonl"
```

```python
# in tracer/worker.py: append at the end of Pipeline._on_close,
# after `action, tracklet_id = self.merger.offer(record)`
        searchable_at = datetime.now(timezone.utc)
        latency = (searchable_at - raw.ended_at).total_seconds()
        self.latencies.append(latency)
        self._write_metric(raw, tracklet_id, searchable_at, latency)
```

```python
# in tracer/worker.py: add as a Pipeline method
    def _write_metric(self, raw: RawTracklet, tracklet_id: int,
                      searchable_at: datetime, latency: float) -> None:
        """R8 evidence: time from the person leaving frame to being searchable."""
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"camera_id": raw.camera_id, "tracklet_id": tracklet_id,
               "last_seen": raw.ended_at.isoformat(),
               "searchable_at": searchable_at.isoformat(),
               "latency_s": round(latency, 3)}
        with self._metrics_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
```

Add `import json` and `from datetime import datetime, timezone` to the imports at the top of `tracer/worker.py`.

- [ ] **Step 4: Add the worker entrypoint**

```python
# append to tracer/worker.py
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run one camera's ingest worker")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--weights", default="data/weights/osnet_x1_0_msmt17.pth")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    from tracer.deps import get_config, get_store
    from tracer.ingest import RtspSource
    from tracer.reid import OsnetEmbedder

    cfg, store = get_config(), get_store()
    camera = cfg.cameras[args.camera]
    if camera.kind != "rtsp" or not camera.source_url:
        raise SystemExit(f"{args.camera} is not an rtsp camera with a source_url")

    pipeline = Pipeline(cfg, args.camera, store,
                        OsnetEmbedder(args.weights, device=os.environ.get("DEVICE", cfg.device)),
                        YoloDetector(cfg))
    log.info("worker for %s reading %s", args.camera, camera.source_url)
    pipeline.run(RtspSource(camera.source_url, cfg.sample_fps))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the latency report script**

```python
# scripts/report_latency.py
"""Report the R8 latency distribution: median and worst case, not one run."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/metrics/latency.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"no measurements in {path}")

by_camera: dict[str, list[float]] = {}
for r in rows:
    by_camera.setdefault(r["camera_id"], []).append(r["latency_s"])

print(f"{'camera':<10}{'n':<5}{'median':<10}{'p90':<10}{'worst':<10}")
for cam, values in sorted(by_camera.items()):
    values.sort()
    p90 = values[min(int(len(values) * 0.9), len(values) - 1)]
    print(f"{cam:<10}{len(values):<5}{statistics.median(values):<10.2f}{p90:<10.2f}{max(values):<10.2f}")

allv = sorted(v for vs in by_camera.values() for v in vs)
print(f"\nall cameras: n={len(allv)} median={statistics.median(allv):.2f}s "
      f"worst={max(allv):.2f}s  (R8 requires <= 10s)")
print("R8: PASS" if max(allv) <= 10.0 else "R8: FAIL")
```

- [ ] **Step 6: Add worker services to compose**

```yaml
# add to docker-compose.yml under services:
  worker-cam1:
    build:
      context: .
      args:
        TORCH_INDEX: ${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
    <<: *gpu
    environment:
      TRACER_DSN: postgresql://tracer:tracer@db:5432/tracer
      TRACER_CONFIG: /app/config.yaml
      DEVICE: ${DEVICE:-cpu}
      HSA_OVERRIDE_GFX_VERSION: "10.3.0"
    volumes:
      - .:/app
      - ./data:/app/data
      - modelcache:/root/.cache/torch
    depends_on:
      db: {condition: service_healthy}
    command: python -m tracer.worker --camera cam1
    restart: unless-stopped

  worker-cam2:
    extends:
      service: worker-cam1
    command: python -m tracer.worker --camera cam2
```

- [ ] **Step 7: Run the tests**

Run: `docker compose run --rm app pytest tests/test_latency.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 8: Measure latency against the live replay stream**

```bash
docker compose up -d db mediamtx
# Point cam1 at the replay path for this measurement:
#   source_url: rtsp://<host-ip>:8554/replay
docker compose up -d worker-cam1
sleep 120
docker compose run --rm app python scripts/report_latency.py
```

Expected: a table with median well under 10 s and a final `R8: PASS`.

- [ ] **Step 9: Commit**

```bash
git add tracer/worker.py scripts/report_latency.py docker-compose.yml tests/test_latency.py
git commit -m "feat: live RTSP workers with R8 latency instrumentation and reporting"
```

---

### Task 19: Retention

**Files:**
- Create: `tracer/retention.py`
- Modify: `tracer/api.py` (start the hourly task)
- Test: `tests/test_retention.py`

**Interfaces:**
- Consumes: `Store.expired_tracklets`, `Store.delete_tracklets`, `Store.all_media_paths` (Task 4).
- Produces: `sweep(store, cfg, now=None) -> dict` with keys `tracklets`, `files`, `orphans`, `photos`; `retention_loop(store, cfg, interval_s=3600)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retention.py
from datetime import datetime, timedelta, timezone

import pytest

from tracer.config import CameraCfg, Config
from tracer.fixtures import make_tracklet
from tracer.retention import sweep

NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.78, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file")})


def add(store, cfg, *, age_days: float, protected: bool = False) -> int:
    start = NOW - timedelta(days=age_days)
    rec = make_tracklet(camera_id="cam1", started_at=start,
                        ended_at=start + timedelta(seconds=5))
    tid = store.insert_tracklet(rec, protected=protected)
    for rel in (rec.thumb_path, rec.clip_path):
        p = cfg.data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return tid


def test_tracklets_past_the_retention_window_are_deleted(store, cfg):
    old = add(store, cfg, age_days=5)
    fresh = add(store, cfg, age_days=1)
    counts = sweep(store, cfg, now=NOW)
    assert counts["tracklets"] == 1
    remaining = [r[0] for r in store.conn.execute("SELECT id FROM tracklet").fetchall()]
    assert remaining == [fresh]
    assert old not in remaining


def test_protected_tracklets_are_never_deleted(store, cfg):
    # The labelled evaluation set must survive, or the ground-truth CSV
    # ends up pointing at rows that no longer exist.
    kept = add(store, cfg, age_days=30, protected=True)
    counts = sweep(store, cfg, now=NOW)
    assert counts["tracklets"] == 0
    assert store.conn.execute("SELECT id FROM tracklet").fetchone()[0] == kept


def test_clips_and_thumbnails_are_removed_from_disk(store, cfg):
    add(store, cfg, age_days=5)
    paths = list((cfg.data_dir / "clips").glob("*")) + list((cfg.data_dir / "thumbs").glob("*"))
    assert paths
    counts = sweep(store, cfg, now=NOW)
    assert counts["files"] == len(paths)
    assert not any(p.exists() for p in paths)


def test_orphaned_files_are_swept(store, cfg):
    # Left behind by a crashed ingest: on disk, absent from the database.
    orphan = cfg.data_dir / "clips" / "orphan.mp4"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"x")
    counts = sweep(store, cfg, now=NOW)
    assert counts["orphans"] == 1
    assert not orphan.exists()


def test_reference_photos_of_closed_searches_are_deleted(store, cfg):
    photo = cfg.data_dir / "photos" / "ref.jpg"
    photo.parent.mkdir(parents=True, exist_ok=True)
    photo.write_bytes(b"x")
    sid = store.create_search("photos/ref.jpg", None, make_tracklet().embedding,
                              "osnet", (None, None))
    store.conn.execute("UPDATE search SET closed_at = %s WHERE id = %s", (NOW, sid))
    counts = sweep(store, cfg, now=NOW)
    assert counts["photos"] == 1
    assert not photo.exists()


def test_an_open_search_keeps_its_photo(store, cfg):
    photo = cfg.data_dir / "photos" / "open.jpg"
    photo.parent.mkdir(parents=True, exist_ok=True)
    photo.write_bytes(b"x")
    store.create_search("photos/open.jpg", None, make_tracklet().embedding,
                        "osnet", (None, None))
    counts = sweep(store, cfg, now=NOW)
    assert counts["photos"] == 0
    assert photo.exists()


def test_a_sweep_with_nothing_to_do_reports_zeros(store, cfg):
    assert sweep(store, cfg, now=NOW) == {"tracklets": 0, "files": 0,
                                          "orphans": 0, "photos": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_retention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.retention'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/retention.py
"""Delete footage past its retention window, logging what went.

Rows marked `protected` are never deleted: the labelled evaluation set has
to outlive the retention period or the ground-truth CSV points at nothing.
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

    known = store.all_media_paths()
    orphans = 0
    for sub in MEDIA_DIRS:
        for path in (root / sub).glob("*"):
            if path.is_file() and str(path.relative_to(root)) not in known:
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
    """Hourly sweep inside the app process: no extra container, no host cron."""
    while True:
        try:
            sweep(store, cfg)
        except Exception:
            log.exception("retention sweep failed; continuing")
        await asyncio.sleep(interval_s)
```

- [ ] **Step 4: Start the loop with the app**

```python
# append to tracer/api.py
import asyncio
import contextlib


@app.on_event("startup")
async def _start_retention() -> None:
    from tracer.retention import retention_loop
    interval = float(os.environ.get("RETENTION_INTERVAL_S", 3600))
    app.state.retention_task = asyncio.create_task(
        retention_loop(get_store(), get_config(), interval))


@app.on_event("shutdown")
async def _stop_retention() -> None:
    task = getattr(app.state, "retention_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
```

Add `import os` to the top of `tracer/api.py` if it is not already there.

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm app pytest tests/test_retention.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 6: Demonstrate the job working on a short override**

```bash
RETENTION_INTERVAL_S=10 docker compose up -d app
docker compose logs -f app | grep -m1 "retention sweep"
```

Expected: a log line listing the counts. For the demo, temporarily set `retention_days: 0` in `config.yaml` so a just-ingested tracklet is removed and the deletion is visible.

- [ ] **Step 7: Commit**

```bash
git add tracer/retention.py tracer/api.py tests/test_retention.py
git commit -m "feat: hourly retention sweep with protected rows and orphan cleanup"
```

**Phase 3 checkpoint (final acceptance reachable):** two live cameras feed tracklets through MediaMTX, the timeline updates by polling, confirm/reject works, and retention runs.

---

# Phase 4 — Evaluation and stretch feature

### Task 20: Market-1501 download and injection as tracklets

**Files:**
- Create: `scripts/fetch_market1501.sh`, `tracer/market1501.py`, `scripts/inject_market1501.py`
- Test: `tests/test_market1501.py`

**Interfaces:**
- Consumes: `OsnetEmbedder` (Task 10), `Store` (Task 4).
- Produces: `parse_name(filename) -> tuple[int, int]` returning `(person_id, camera_number)`; `list_images(root, split) -> list[Path]`; `group_by_identity_camera(paths) -> dict[tuple[int, int], list[Path]]`; `inject(store, embedder, paths, base_time, spacing_s, protected=True) -> dict[int, list[int]]` mapping person id to inserted tracklet ids.

- [ ] **Step 1: Write the download script**

```bash
# scripts/fetch_market1501.sh
#!/usr/bin/env bash
# Market-1501 (Zheng et al., 2015), research use. The original link is
# unreliable; this is a mirror of the same archive on HuggingFace.
set -euo pipefail
DEST="${1:-data/market1501}"
URL="https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/Market-1501-v15.09.15.zip"

mkdir -p "$DEST"
if [[ -d "$DEST/Market-1501-v15.09.15/query" ]]; then
  echo "already extracted at $DEST"
else
  echo "downloading (~1.8 GB)…"
  curl -fL --progress-bar -o "$DEST/market1501.zip" "$URL"
  unzip -q "$DEST/market1501.zip" -d "$DEST"
  rm -f "$DEST/market1501.zip"
fi

for split in query bounding_box_test bounding_box_train; do
  n=$(find "$DEST/Market-1501-v15.09.15/$split" -name '*.jpg' | wc -l)
  echo "  $split: $n images"
  test "$n" -gt 0
done
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_market1501.py
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracer.market1501 import (group_by_identity_camera, inject, list_images,
                               parse_name)

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def test_filenames_parse_into_person_and_camera():
    assert parse_name("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_name("1500_c6s3_086567_01.jpg") == (1500, 6)


def test_junk_and_distractor_ids_parse_too():
    assert parse_name("-1_c2s3_065187_03.jpg") == (-1, 2)
    assert parse_name("0000_c1s1_000151_01.jpg") == (0, 1)


def test_an_unparseable_name_raises():
    with pytest.raises(ValueError, match="unexpected"):
        parse_name("not-a-market-filename.jpg")


@pytest.fixture
def fake_dataset(tmp_path) -> Path:
    """Three identities over two cameras, four crops each."""
    root = tmp_path / "bounding_box_test"
    root.mkdir()
    for pid in (1, 2, 3):
        for cam in (1, 2):
            for k in range(4):
                img = np.full((128, 64, 3), (pid * 60 + cam * 20 + k) % 256, np.uint8)
                cv2.imwrite(str(root / f"{pid:04d}_c{cam}s1_{k:06d}_01.jpg"), img)
    # Junk images that must be ignored.
    for k in range(2):
        cv2.imwrite(str(root / f"-1_c1s1_{k:06d}_01.jpg"),
                    np.zeros((128, 64, 3), np.uint8))
    return tmp_path


def test_listing_skips_junk_images(fake_dataset):
    paths = list_images(fake_dataset, "bounding_box_test")
    assert len(paths) == 24
    assert all(parse_name(p.name)[0] > 0 for p in paths)


def test_grouping_produces_one_group_per_identity_and_camera(fake_dataset):
    groups = group_by_identity_camera(list_images(fake_dataset, "bounding_box_test"))
    assert len(groups) == 6
    assert all(len(v) == 4 for v in groups.values())
    assert (1, 1) in groups and (3, 2) in groups


class StubEmbedder:
    """One embedding per identity, so injected tracklets of the same person
    are similar and different people are not."""
    def embed_images(self, images):
        from tracer.models import unit_vector
        out = []
        for img in images:
            seed = int(img.mean()) % 97
            rng = np.random.default_rng(seed)
            out.append(unit_vector(rng.standard_normal(512).astype(np.float32)))
        return np.stack(out)

    def embed_tracklet(self, crops):
        from tracer.models import unit_vector
        rows = self.embed_images([c.image for c in crops])
        return unit_vector(rows.mean(axis=0)), min(len(crops), 10)


def test_injection_writes_one_tracklet_per_identity_camera_group(store, fake_dataset):
    paths = list_images(fake_dataset, "bounding_box_test")
    mapping = inject(store, StubEmbedder(), paths, base_time=T0, spacing_s=60.0)
    assert set(mapping) == {1, 2, 3}
    assert all(len(v) == 2 for v in mapping.values())
    assert store.conn.execute("SELECT count(*) FROM tracklet").fetchone()[0] == 6


def test_injected_tracklets_use_market_camera_ids(store, fake_dataset):
    inject(store, StubEmbedder(), list_images(fake_dataset, "bounding_box_test"),
           base_time=T0, spacing_s=60.0)
    cams = {r[0] for r in store.conn.execute(
        "SELECT DISTINCT camera_id FROM tracklet").fetchall()}
    assert cams == {"m1", "m2"}


def test_injected_tracklets_are_spaced_so_timing_never_rejects_them(store, fake_dataset):
    inject(store, StubEmbedder(), list_images(fake_dataset, "bounding_box_test"),
           base_time=T0, spacing_s=60.0)
    rows = store.conn.execute(
        "SELECT started_at, ended_at FROM tracklet ORDER BY started_at").fetchall()
    gaps = [(b[0] - a[1]).total_seconds() for a, b in zip(rows, rows[1:])]
    assert all(g >= 10.0 for g in gaps), f"smallest gap {min(gaps)}s"


def test_injected_tracklets_are_protected_from_retention(store, fake_dataset):
    inject(store, StubEmbedder(), list_images(fake_dataset, "bounding_box_test"),
           base_time=T0, spacing_s=60.0, protected=True)
    assert store.conn.execute(
        "SELECT bool_and(protected) FROM tracklet").fetchone()[0] is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_market1501.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.market1501'`

- [ ] **Step 4: Write the implementation**

```python
# tracer/market1501.py
"""Market-1501 as a source of labelled, genuinely cross-camera gallery data.

The dataset is crops rather than video, so it cannot flow through detection
and tracking — but crops are exactly what the embedder consumes. Injecting
them as tracklets gives real identity labels through the real search path,
which single-subject own footage cannot provide.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

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
    """All non-junk images in a split. Person id -1 is junk and 0 is a
    distractor the standard protocol excludes from queries."""
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
        crops.append(Crop(image=img[:, :, ::-1].copy(),          # BGR -> RGB
                          t=t + timedelta(milliseconds=100 * i)))
    return crops


def inject(store, embedder, paths: list[Path], base_time: datetime,
           spacing_s: float = 60.0, protected: bool = True,
           max_identities: int | None = None) -> dict[int, list[int]]:
    """Write one tracklet per (identity, camera) group.

    Groups are spaced `spacing_s` apart so the impossible-timing filter never
    rejects a true match purely because of synthetic timestamps.
    """
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
            thumb_path=str(group[0]),
            clip_path="",
            n_crops=n_crops,
        )
        mapping.setdefault(pid, []).append(store.insert_tracklet(rec, protected=protected))

    log.info("injected %d identities as %d tracklets",
             len(mapping), sum(len(v) for v in mapping.values()))
    return mapping
```

- [ ] **Step 5: Write the injection CLI**

```python
# scripts/inject_market1501.py
"""Inject Market-1501 identities as tracklets, for a labelled gallery.

  python -m scripts.inject_market1501 --root data/market1501 --identities 200
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from tracer.deps import get_config, get_store
from tracer.market1501 import inject, list_images
from tracer.reid import OsnetEmbedder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/market1501")
    ap.add_argument("--split", default="bounding_box_test")
    ap.add_argument("--identities", type=int, default=200)
    ap.add_argument("--weights", default="data/weights/osnet_x1_0_msmt17.pth")
    ap.add_argument("--base-time", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--out", default="data/eval/market_injection.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg, store = get_config(), get_store()
    paths = list_images(args.root, args.split)
    print(f"{len(paths)} images in {args.split}")

    mapping = inject(store, OsnetEmbedder(args.weights, device=str(cfg.device)), paths,
                     base_time=datetime.fromisoformat(args.base_time),
                     max_identities=args.identities)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2))
    print(f"injected {len(mapping)} identities, "
          f"{sum(len(v) for v in mapping.values())} tracklets -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and a real injection**

Run: `docker compose run --rm app pytest tests/test_market1501.py -v`
Expected: PASS, 9 tests.

```bash
./scripts/fetch_market1501.sh data/market1501
docker compose run --rm app python -m scripts.inject_market1501 --identities 200
```

Expected: roughly 200 identities and several hundred tracklets written.

- [ ] **Step 7: Commit**

```bash
chmod +x scripts/fetch_market1501.sh
git add scripts/fetch_market1501.sh scripts/inject_market1501.py tracer/market1501.py tests/test_market1501.py
git commit -m "feat: Market-1501 parsing and injection as a labelled cross-camera gallery"
```

---

### Task 21: Benchmark, threshold sweep, and the OSNet-AIN A/B

**Files:**
- Create: `tracer/evaluation.py`, `scripts/eval_market1501.py`, `scripts/tune_tau.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `OsnetEmbedder` (Task 10), `market1501` (Task 20), `run_search` (Task 6).
- Produces: `cmc_and_map(query_ids, query_cams, gallery_ids, gallery_cams, sims) -> dict` with keys `rank1`, `rank5`, `mAP`; `sweep_tau(store, cfg, queries, taus) -> list[dict]` with keys `tau`, `top1`, `top5`, `false_positives`.

**Reporting trap to respect:** the well-known ~94% rank-1 OSNet figures are for a model *trained on* Market-1501. These weights are trained on MSMT17, so this is a cross-domain evaluation and the correct comparison is the cross-domain table in the OSNet-AIN paper. Verify those figures from the paper when writing the report; do not quote them from memory.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation.py
import numpy as np
import pytest

from tracer.evaluation import cmc_and_map


def test_a_perfect_ranking_scores_one():
    # One query (id 1, cam 1); gallery has the same id on another camera first.
    sims = np.array([[0.9, 0.1, 0.2]])
    got = cmc_and_map(query_ids=[1], query_cams=[1],
                      gallery_ids=[1, 2, 3], gallery_cams=[2, 2, 2], sims=sims)
    assert got["rank1"] == 1.0
    assert got["rank5"] == 1.0
    assert got["mAP"] == 1.0


def test_a_wrong_top_match_fails_rank1_but_can_pass_rank5():
    sims = np.array([[0.1, 0.9, 0.2]])
    got = cmc_and_map([1], [1], [1, 2, 3], [2, 2, 2], sims)
    assert got["rank1"] == 0.0
    assert got["rank5"] == 1.0
    assert 0.0 < got["mAP"] < 1.0


def test_same_camera_gallery_items_are_excluded_from_scoring():
    # The standard protocol ignores gallery images of the same person on the
    # same camera; counting them would inflate every number reported.
    sims = np.array([[0.99, 0.5, 0.1]])
    got = cmc_and_map([1], [1], [1, 1, 2], [1, 2, 2], sims)
    assert got["rank1"] == 1.0      # the 0.99 same-camera hit is dropped,
    assert got["mAP"] == 1.0        # leaving the 0.5 cross-camera hit first


def test_a_query_with_no_valid_gallery_match_is_skipped():
    sims = np.array([[0.9, 0.8]])
    got = cmc_and_map([1], [1], [1, 1], [1, 1], sims)
    assert got["n_queries"] == 0
    assert got["rank1"] == 0.0


def test_multiple_correct_matches_contribute_to_map():
    sims = np.array([[0.9, 0.8, 0.1]])
    got = cmc_and_map([1], [1], [1, 1, 2], [2, 3, 2], sims)
    assert got["rank1"] == 1.0
    assert got["mAP"] == pytest.approx(1.0)


def test_averaging_over_several_queries():
    sims = np.array([[0.9, 0.1], [0.1, 0.9]])
    got = cmc_and_map([1, 2], [1, 1], [1, 2], [2, 2], sims)
    assert got["rank1"] == 1.0
    assert got["n_queries"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.evaluation'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/evaluation.py
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
    excluded, because matching a person to another shot from the same camera
    is not re-identification.
    """
    query_ids = np.asarray(query_ids)
    query_cams = np.asarray(query_cams)
    gallery_ids = np.asarray(gallery_ids)
    gallery_cams = np.asarray(gallery_cams)

    ranks, aps = [], []
    for i in range(len(query_ids)):
        order = np.argsort(-sims[i])
        keep = ~((gallery_ids[order] == query_ids[i]) &
                 (gallery_cams[order] == query_cams[i]))
        hits = (gallery_ids[order] == query_ids[i])[keep]
        if not hits.any():
            continue                       # no valid cross-camera match exists

        first = int(np.argmax(hits))
        ranks.append(first)

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
    matches get through — the "wrong matches passing tau" column.

    `queries` entries are {"embedding": np.ndarray, "person": int,
    "truth": set[int]} where `truth` holds the tracklet ids of that person.
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
```

- [ ] **Step 4: Write the benchmark script**

```python
# scripts/eval_market1501.py
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
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the threshold tuning script**

```python
# scripts/tune_tau.py
"""Tier 2: sweep tau against the injected Market-1501 gallery.

  python -m scripts.tune_tau --identities 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tracer.deps import get_config, get_store
from tracer.evaluation import sweep_tau
from tracer.market1501 import list_images, parse_name
from tracer.reid import OsnetEmbedder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/market1501")
    ap.add_argument("--mapping", default="data/eval/market_injection.json")
    ap.add_argument("--weights", default="data/weights/osnet_x1_0_msmt17.pth")
    ap.add_argument("--identities", type=int, default=50)
    ap.add_argument("--taus", default="0.50,0.60,0.65,0.70,0.74,0.78,0.82,0.86,0.90")
    ap.add_argument("--out", default="data/eval/tau_sweep.json")
    args = ap.parse_args()

    cfg, store = get_config(), get_store()
    mapping = {int(k): v for k, v in json.loads(Path(args.mapping).read_text()).items()}
    embedder = OsnetEmbedder(args.weights, device=str(cfg.device))

    queries = []
    for p in list_images(args.root, "query"):
        pid, _ = parse_name(p.name)
        if pid not in mapping or len(queries) >= args.identities:
            continue
        if any(q["person"] == pid for q in queries):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        emb = embedder.embed_images([img[:, :, ::-1].copy()])[0]
        queries.append({"embedding": np.asarray(emb, dtype=np.float32),
                        "person": pid, "truth": set(mapping[pid])})

    print(f"{len(queries)} query identities against "
          f"{sum(len(v) for v in mapping.values())} gallery tracklets")

    rows = sweep_tau(store, cfg, queries, [float(t) for t in args.taus.split(",")])
    best = max(rows, key=lambda r: (r["top1"], -r["false_positives"]))
    print(f"\n{'tau':<8}{'top1':<10}{'top5':<10}{'false positives':<16}")
    for r in rows:
        mark = "  <-- best top-1" if r is best else ""
        print(f"{r['tau']:<8.2f}{r['top1']:<10.3f}{r['top5']:<10.3f}"
              f"{r['false_positives']:<16}{mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "best_tau": best["tau"],
                               "n_queries": len(queries)}, indent=2))
    print(f"\nwrote {out}. Put best_tau into config.yaml if it beats 0.78.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests, then both benchmark rows**

Run: `docker compose run --rm app pytest tests/test_evaluation.py -v`
Expected: PASS, 6 tests.

```bash
DEVICE=cuda docker compose run --rm app python -m scripts.eval_market1501 \
  --weights data/weights/osnet_x1_0_msmt17.pth --device cuda
DEVICE=cuda docker compose run --rm app python -m scripts.eval_market1501 \
  --weights data/weights/osnet_ain_x1_0_msmt17.pth --model-name osnet_ain_x1_0 --device cuda
docker compose run --rm app python -m scripts.tune_tau --identities 50
```

Expected: two result rows in `data/eval/market1501_results.json`, and a τ sweep table. OSNet-AIN is expected to win cross-domain; record whichever actually does.

- [ ] **Step 7: Commit**

```bash
git add tracer/evaluation.py scripts/eval_market1501.py scripts/tune_tau.py tests/test_evaluation.py
git commit -m "feat: Market-1501 benchmark, tau sweep and OSNet-AIN comparison"
```

---

### Task 22: R13 — text search with CLIP

**Files:**
- Create: `tracer/textsearch.py`, `scripts/backfill_clip.py`
- Modify: `requirements.txt` (add `open_clip_torch`)
- Test: `tests/test_textsearch.py`

**Interfaces:**
- Consumes: `Store.set_clip_embedding` (Task 4), `run_search` with `space="clip"` (Task 6).
- Produces: `ClipEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")` with `encode_text(text: str) -> np.ndarray` and `encode_images(images: list[np.ndarray]) -> np.ndarray`, both 512-d L2-normalised; `embed_text(text: str, cfg) -> np.ndarray`.

ViT-B-32's embedding width is 512, which matches the `clip_emb VECTOR(512)` column — no schema change.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_textsearch.py
from pathlib import Path

import numpy as np
import pytest

from tracer.textsearch import ClipEncoder

CLIP_AVAILABLE = True
try:
    import open_clip  # noqa: F401
except ImportError:
    CLIP_AVAILABLE = False


def test_the_encoder_rejects_an_empty_query():
    enc = ClipEncoder.__new__(ClipEncoder)
    with pytest.raises(ValueError, match="empty"):
        ClipEncoder.validate_text(enc, "   ")


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="open_clip not installed")
class TestWithRealClip:
    @pytest.fixture(scope="class")
    def enc(self) -> ClipEncoder:
        return ClipEncoder(device="cpu")

    def test_text_embeddings_are_512d_unit_vectors(self, enc):
        v = enc.encode_text("a person in a red shirt")
        assert v.shape == (512,)
        assert v.dtype == np.float32
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)

    def test_image_embeddings_share_the_shape(self, enc):
        imgs = [np.full((128, 64, 3), 100, np.uint8), np.full((128, 64, 3), 200, np.uint8)]
        out = enc.encode_images(imgs)
        assert out.shape == (2, 512)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_colour_text_prefers_the_matching_colour_patch(self, enc):
        """A weak but real signal: a red patch should score higher against
        'red' than a blue patch does."""
        red = np.zeros((128, 64, 3), np.uint8); red[:, :, 0] = 220
        blue = np.zeros((128, 64, 3), np.uint8); blue[:, :, 2] = 220
        t = enc.encode_text("a red shirt")
        imgs = enc.encode_images([red, blue])
        assert float(imgs[0] @ t) > float(imgs[1] @ t)

    def test_text_image_similarities_sit_near_the_clip_threshold(self, enc):
        """CLIP text-image cosines live around 0.2-0.35, which is why tau is
        per-space: the osnet value of 0.78 would reject everything."""
        t = enc.encode_text("a person walking")
        img = np.full((128, 64, 3), 120, np.uint8)
        sim = float(enc.encode_images([img])[0] @ t)
        assert -0.1 < sim < 0.6
```

- [ ] **Step 2: Add the dependency and run the test**

Append `open_clip_torch==2.30.0` to `requirements.txt`, rebuild, then:

Run: `docker compose build app && docker compose run --rm app pytest tests/test_textsearch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracer.textsearch'`

- [ ] **Step 3: Write the implementation**

```python
# tracer/textsearch.py
"""R13: find a person from a description instead of a photo.

CLIP vectors are not comparable with OSNet vectors, so they live in their own
column and carry their own threshold. `search.space` records which space a
query belongs to; mixing them would return confident nonsense.
"""
from __future__ import annotations

import functools
import logging

import numpy as np

from tracer.models import unit_vector

log = logging.getLogger(__name__)

PROMPT = "a photo of {}"          # CLIP works better with a prompt template


class ClipEncoder:
    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", device: str = "cpu") -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if device in ("cuda", "gpu") else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @staticmethod
    def validate_text(_self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("text query must not be empty")
        return text.strip()

    def encode_text(self, text: str) -> np.ndarray:
        text = ClipEncoder.validate_text(self, text)
        tokens = self.tokenizer([PROMPT.format(text)]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
        return unit_vector(feats[0].cpu().numpy())

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        batch = self.torch.stack([
            self.preprocess(Image.fromarray(img)) for img in images]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_image(batch)
        arr = feats.cpu().numpy().astype(np.float32)
        return np.stack([unit_vector(row) for row in arr])


@functools.lru_cache(maxsize=1)
def _encoder(device: str) -> ClipEncoder:
    return ClipEncoder(device=device)


def embed_text(text: str, cfg) -> np.ndarray:
    """Entry point used by the API when a text query arrives."""
    return _encoder(str(cfg.device)).encode_text(text)
```

- [ ] **Step 4: Write the backfill script**

```python
# scripts/backfill_clip.py
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
    ap.add_argument("--all", action="store_true", help="re-embed rows that already have one")
    args = ap.parse_args()

    cfg, store = get_config(), get_store()
    where = "" if args.all else "WHERE clip_emb IS NULL"
    rows = store.conn.execute(
        f"SELECT id, thumb_path FROM tracklet {where} ORDER BY id").fetchall()
    print(f"{len(rows)} tracklet(s) to embed")
    if not rows:
        return

    enc = ClipEncoder(device=str(cfg.device))
    done = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        ids, images = [], []
        for tid, thumb in chunk:
            path = Path(thumb)
            if not path.is_absolute():
                path = Path(cfg.data_dir) / thumb
            img = cv2.imread(str(path))
            if img is None:
                print(f"  skipping {tid}: cannot read {path}")
                continue
            ids.append(tid)
            images.append(img[:, :, ::-1].copy())
        if not images:
            continue
        for tid, vec in zip(ids, enc.encode_images(images)):
            store.set_clip_embedding(tid, vec)
            done += 1
        print(f"  {done}/{len(rows)}", end="\r")
    print(f"\nembedded {done} tracklet(s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add an end-to-end text-search test**

```python
# tests/test_text_search_end_to_end.py
import numpy as np
import pytest

from tracer.config import CameraCfg, Config
from tracer.fixtures import make_similar_embedding, make_tracklet
from tracer.search import run_search


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.78, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file")})


def test_a_text_query_ranks_the_matching_tracklet_first(store, cfg):
    """The CLIP space uses its own threshold, so a 0.31 similarity — far
    below tau.osnet — is a legitimate text match."""
    query = make_tracklet().embedding
    good = store.insert_tracklet(make_tracklet(camera_id="cam1"))
    poor = store.insert_tracklet(make_tracklet(camera_id="cam1"))
    store.set_clip_embedding(good, make_similar_embedding(query, 0.31))
    store.set_clip_embedding(poor, make_similar_embedding(query, 0.12))

    res = run_search(store, cfg, query, "clip", (None, None), ["cam1"], limit=10)
    assert [m.tracklet_id for m in res.matches] == [good]


def test_osnet_rows_are_untouched_by_a_text_search(store, cfg):
    query = make_tracklet().embedding
    store.insert_tracklet(make_tracklet(camera_id="cam1",
                                        embedding=make_similar_embedding(query, 0.99)))
    res = run_search(store, cfg, query, "clip", (None, None), ["cam1"], limit=10)
    assert res.matches == []
    assert res.reason == "no_tracklets_in_scope"
```

- [ ] **Step 6: Run everything and backfill**

```bash
docker compose run --rm app pytest tests/test_textsearch.py tests/test_text_search_end_to_end.py -v
docker compose run --rm app python -m scripts.backfill_clip
docker compose run --rm app python -c "
from tracer.deps import get_config, get_store
from tracer.textsearch import embed_text
from tracer.search import run_search
cfg, store = get_config(), get_store()
res = run_search(store, cfg, embed_text('a person in a dark jacket', cfg), 'clip',
                 (None, None), list(cfg.cameras), limit=5)
print(res.reason, [(m.tracklet_id, round(m.score, 3)) for m in res.matches])
"
```

Expected: tests pass; the backfill reports a count; the text query returns ranked tracklets with scores in the 0.2–0.4 band. If every score is under `tau.clip`, lower it in `config.yaml` and record the calibrated value.

- [ ] **Step 7: Commit**

```bash
git add tracer/textsearch.py scripts/backfill_clip.py requirements.txt tests/test_textsearch.py tests/test_text_search_end_to_end.py
git commit -m "feat: R13 text search via CLIP with its own embedding space and threshold"
```

---

### Task 23: Own-footage evaluation and the generated results table

**Files:**
- Create: `scripts/label_helper.py`, `scripts/eval_own_footage.py`, `scripts/eval_table.py`
- Test: `tests/test_eval_own_footage.py`

**Interfaces:**
- Consumes: `run_search` (Task 6), `Store` (Task 4).
- Produces: `score_queries(store, cfg, queries, distractor_count) -> dict` with keys `top1`, `top5`, `n_queries`, `distractors`, `false_positives`; `data/eval/ground_truth.csv` with columns `tracklet_id,person`; `docs/evaluation.md` generated from the JSON result files.

**Why a distractor count is reported with every number:** with one subject and no distractors, top-1 is vacuously 100%. "Correct sighting ranked above N real distractors" is the honest claim, and N belongs in the table.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_own_footage.py
import numpy as np
import pytest

from tracer.config import CameraCfg, Config
from tracer.fixtures import make_similar_embedding, make_tracklet
from scripts.eval_own_footage import score_queries


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.78, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file"),
                           "cam2": CameraCfg(id="cam2", name="cam2", kind="file")})


def test_a_correct_top_match_scores_one(store, cfg):
    q = make_tracklet().embedding
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.95)))
    got = score_queries(store, cfg, [{"embedding": q, "truth": {tid}}], distractor_count=0)
    assert got["top1"] == 1.0
    assert got["top5"] == 1.0
    assert got["n_queries"] == 1


def test_a_distractor_outranking_the_truth_fails_top1(store, cfg):
    q = make_tracklet().embedding
    truth = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.85)))
    store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.97)))
    got = score_queries(store, cfg, [{"embedding": q, "truth": {truth}}],
                        distractor_count=1)
    assert got["top1"] == 0.0
    assert got["top5"] == 1.0
    assert got["false_positives"] >= 1


def test_the_distractor_count_is_reported(store, cfg):
    q = make_tracklet().embedding
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.95)))
    got = score_queries(store, cfg, [{"embedding": q, "truth": {tid}}],
                        distractor_count=750)
    assert got["distractors"] == 750


def test_no_match_above_threshold_scores_zero(store, cfg):
    q = make_tracklet().embedding
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.40)))
    got = score_queries(store, cfg, [{"embedding": q, "truth": {tid}}], distractor_count=0)
    assert got["top1"] == 0.0
    assert got["top5"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_eval_own_footage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval_own_footage'`

- [ ] **Step 3: Write the labelling helper**

```python
# scripts/label_helper.py
"""Write a ground-truth CSV template listing every tracklet to label.

  python -m scripts.label_helper --cameras cam1 cam2 > data/eval/ground_truth.csv

Then fill the `person` column by looking at each thumbnail. Leave the row
out, or write `?`, for tracklets that are not the subject.
"""
from __future__ import annotations

import argparse

from tracer.deps import get_store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cameras", nargs="+", required=True)
    args = ap.parse_args()

    store = get_store()
    rows = store.conn.execute(
        """SELECT id, camera_id, started_at, ended_at, n_crops, thumb_path
           FROM tracklet WHERE camera_id = ANY(%s) ORDER BY started_at""",
        (list(args.cameras),)).fetchall()

    print("tracklet_id,person,camera,started_at,duration_s,n_crops,thumb")
    for tid, cam, start, end, n_crops, thumb in rows:
        print(f"{tid},?,{cam},{start.isoformat()},"
              f"{(end - start).total_seconds():.1f},{n_crops},{thumb}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the own-footage evaluation**

```python
# scripts/eval_own_footage.py
"""Tier 3: own-footage accuracy against a padded distractor gallery.

  python -m scripts.eval_own_footage --photos data/eval/photos \
      --truth data/eval/ground_truth.csv --cameras cam1 cam2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tracer.search import run_search


def score_queries(store, cfg, queries: list[dict], distractor_count: int,
                  cameras: list[str] | None = None) -> dict:
    """top-1 / top-5 against a gallery that includes `distractor_count`
    tracklets belonging to other people."""
    cameras = cameras or sorted({r[0] for r in store.conn.execute(
        "SELECT DISTINCT camera_id FROM tracklet").fetchall()})
    top1 = top5 = false_pos = 0
    for q in queries:
        res = run_search(store, cfg, q["embedding"], "osnet", (None, None),
                         cameras, limit=5)
        ids = [m.tracklet_id for m in res.matches]
        if ids and ids[0] in q["truth"]:
            top1 += 1
        if any(i in q["truth"] for i in ids):
            top5 += 1
        false_pos += sum(1 for i in ids if i not in q["truth"])
    n = max(len(queries), 1)
    return {"top1": top1 / n, "top5": top5 / n, "n_queries": len(queries),
            "distractors": distractor_count, "false_positives": false_pos,
            "tau": cfg.tau["osnet"]}


def load_truth(path: Path) -> dict[str, set[int]]:
    """person -> set of tracklet ids, from the hand-filled CSV."""
    truth: dict[str, set[int]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            person = (row.get("person") or "").strip()
            if not person or person == "?":
                continue
            truth.setdefault(person, set()).add(int(row["tracklet_id"]))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos", required=True,
                    help="directory of reference photos named <person>*.jpg")
    ap.add_argument("--truth", default="data/eval/ground_truth.csv")
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--label", default="own footage",
                    help="row label, e.g. 'own footage, jacket change'")
    ap.add_argument("--weights", default="data/weights/osnet_x1_0_msmt17.pth")
    ap.add_argument("--out", default="data/eval/own_footage_results.json")
    args = ap.parse_args()

    from tracer.deps import get_config, get_store
    from tracer.reid import PhotoEmbedder

    cfg, store = get_config(), get_store()
    truth = load_truth(Path(args.truth))
    if not truth:
        raise SystemExit(f"{args.truth} has no labelled rows; fill the person column")

    labelled = {t for ids in truth.values() for t in ids}
    total = store.conn.execute("SELECT count(*) FROM tracklet").fetchone()[0]
    distractors = total - len(labelled)

    embedder = PhotoEmbedder(cfg, weights=args.weights)
    queries = []
    for photo in sorted(Path(args.photos).glob("*.jpg")):
        person = photo.stem.split("_")[0]
        if person not in truth:
            print(f"  skipping {photo.name}: no ground truth for '{person}'")
            continue
        emb, _ = embedder.embed_image(photo.read_bytes())
        queries.append({"embedding": emb, "truth": truth[person], "person": person})

    if not queries:
        raise SystemExit("no usable reference photos")

    result = score_queries(store, cfg, queries, distractors, args.cameras)
    result.update(label=args.label, people=sorted(truth),
                  note=("single subject; gallery padded with Market-1501 identities "
                        "and people tracked from public clips"))
    print(json.dumps(result, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else []
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the table generator**

```python
# scripts/eval_table.py
"""Generate docs/evaluation.md from the result JSON files.

The table in the report is generated, never hand-typed, so a late change to
tau cannot leave stale numbers behind.

  python -m scripts.eval_table
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

EVAL = Path("data/eval")
OUT = Path("docs/evaluation.md")


def load(name: str) -> list[dict]:
    p = EVAL / name
    return json.loads(p.read_text()) if p.exists() else []


def pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def main() -> None:
    lines = ["# Evaluation results", "",
             "Generated by `scripts/eval_table.py`. Do not edit by hand.", ""]

    lines += ["## Tier 1 — Market-1501, embedder alone", "",
              "Cross-domain: weights trained on MSMT17, evaluated on Market-1501.",
              "Compare against the cross-domain table in the OSNet-AIN paper, not",
              "against same-domain figures.", "",
              "| Weights | Model | Rank-1 | Rank-5 | mAP | Queries |",
              "|---|---|---|---|---|---|"]
    for r in load("market1501_results.json"):
        lines.append(f"| {r['weights']} | {r['model']} | {pct(r['rank1'])} | "
                     f"{pct(r['rank5'])} | {pct(r['mAP'])} | {r['n_queries']} |")
    if not load("market1501_results.json"):
        lines.append("| _not yet run_ | | | | | |")

    sweep = load("tau_sweep.json")
    lines += ["", "## Tier 2 — Market-1501 through the real search path", ""]
    if sweep:
        rows = sweep["rows"] if isinstance(sweep, dict) else sweep
        best = sweep.get("best_tau") if isinstance(sweep, dict) else None
        lines += [f"{sweep.get('n_queries', '?')} query identities. "
                  f"Best top-1 at tau = {best}.", "",
                  "| tau | Top-1 | Top-5 | Wrong matches passing tau |",
                  "|---|---|---|---|"]
        for r in rows:
            mark = " **(chosen)**" if best is not None and r["tau"] == best else ""
            lines.append(f"| {r['tau']:.2f}{mark} | {pct(r['top1'])} | "
                         f"{pct(r['top5'])} | {r['false_positives']} |")
    else:
        lines.append("_not yet run_")

    lines += ["", "## Tier 3 — own footage", "",
              "Single subject. Every row states the number of distractor",
              "tracklets in the gallery; top-1 without distractors is vacuous.", "",
              "| Run | tau | Top-1 | Top-5 | Queries | Distractors | False positives |",
              "|---|---|---|---|---|---|---|"]
    own = load("own_footage_results.json")
    for r in own:
        lines.append(f"| {r['label']} | {r['tau']:.2f} | {pct(r['top1'])} | "
                     f"{pct(r['top5'])} | {r['n_queries']} | {r['distractors']} | "
                     f"{r['false_positives']} |")
    if not own:
        lines.append("| _not yet run_ | | | | | | |")

    lines += ["", "## R8 — live latency", ""]
    metrics = Path("data/metrics/latency.jsonl")
    if metrics.exists():
        values = [json.loads(l)["latency_s"] for l in metrics.read_text().splitlines() if l.strip()]
        if values:
            lines += [f"{len(values)} measurements. Median "
                      f"{statistics.median(values):.2f}s, worst {max(values):.2f}s. "
                      f"Requirement is 10s: "
                      f"**{'PASS' if max(values) <= 10 else 'FAIL'}**."]
    else:
        lines.append("_no measurements yet; run a live walk-through_")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and generate the table**

```bash
docker compose run --rm app pytest tests/test_eval_own_footage.py -v
touch scripts/__init__.py
docker compose run --rm app python -m scripts.eval_table
cat docs/evaluation.md
```

Expected: 4 tests pass; `docs/evaluation.md` exists with "_not yet run_" placeholders for tiers not yet measured, which fill in as each script runs.

- [ ] **Step 7: Commit**

```bash
git add scripts/label_helper.py scripts/eval_own_footage.py scripts/eval_table.py scripts/__init__.py tests/test_eval_own_footage.py docs/evaluation.md
git commit -m "feat: own-footage evaluation with distractor counts and a generated results table"
```

---

### Task 24: README, demo runbook, and the final acceptance check

**Files:**
- Create: `README.md`, `docs/demo-runbook.md`
- Test: `tests/test_final_acceptance.py`

**Interfaces:**
- Consumes: everything.
- Produces: a documented setup path and `tests/test_final_acceptance.py::test_two_cameras_reach_the_timeline`, the executable form of the spec's final acceptance criterion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_final_acceptance.py
"""Final acceptance (spec section 'Acceptance'): a volunteer walks past two
cameras; both sightings appear on the timeline within 10s and rank first.

The walk itself is physical, so this test asserts the machine-checkable
parts against whatever the live workers most recently wrote.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tracer.deps import get_config, get_store

METRICS = Path("data/metrics/latency.jsonl")

pytestmark = pytest.mark.skipif(
    not METRICS.exists(),
    reason="run a live walk-through first: docker compose up -d worker-cam1 worker-cam2")


def recent_latencies(within_minutes: int = 30) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=within_minutes)
    rows = [json.loads(l) for l in METRICS.read_text().splitlines() if l.strip()]
    return [r for r in rows
            if datetime.fromisoformat(r["searchable_at"]) >= cutoff]


def test_both_cameras_produced_live_sightings():
    rows = recent_latencies()
    cameras = {r["camera_id"] for r in rows}
    assert len(cameras) >= 2, f"only {cameras} produced sightings"


def test_every_live_sighting_was_searchable_within_ten_seconds():
    rows = recent_latencies()
    assert rows, "no recent sightings"
    worst = max(r["latency_s"] for r in rows)
    assert worst <= 10.0, f"worst latency {worst:.2f}s exceeds the 10s requirement"


def test_two_cameras_reach_the_timeline():
    """The newest search must show both cameras as lanes."""
    store = get_store()
    row = store.conn.execute(
        "SELECT id FROM search ORDER BY created_at DESC LIMIT 1").fetchone()
    assert row is not None, "no search has been run; use the UI first"
    lanes = store.timeline(row[0])
    assert len(lanes) >= 2, f"timeline has lanes {list(lanes)}; expected two cameras"
    for cam, sightings in lanes.items():
        times = [m.started_at for m in sightings]
        assert times == sorted(times), f"{cam} lane is not in time order"


def test_the_timing_check_held_on_the_live_route():
    store, cfg = get_store(), get_config()
    row = store.conn.execute(
        "SELECT id FROM search ORDER BY created_at DESC LIMIT 1").fetchone()
    lanes = store.timeline(row[0])
    flat = [m for ms in lanes.values() for m in ms]
    for a in flat:
        for b in flat:
            if a.camera_id != b.camera_id and a.started_at <= b.started_at:
                gap = (b.started_at - a.ended_at).total_seconds()
                assert gap >= cfg.min_walk(a.camera_id, b.camera_id), (
                    f"{a.camera_id}->{b.camera_id} gap {gap:.1f}s is impossible")
```

- [ ] **Step 2: Run it to confirm it skips cleanly**

Run: `docker compose run --rm app pytest tests/test_final_acceptance.py -v`
Expected: SKIPPED with the message about running a live walk-through. It turns into a real check after Step 5.

- [ ] **Step 3: Write the README**

```markdown
# Missing Person Tracing

From one photo of a missing person, find where and when they appeared across
several cameras, ranked and time-ordered, with clips — so a reviewer checks
minutes of video instead of hours.

People are matched by whole-body appearance (clothing, build, colour) using
OSNet re-identification, with a physical-plausibility check that drops
sightings too close together in time to be the same person walking. Every
match is a suggestion with a score; a human confirms or rejects each one.

- Product spec: `Missing Person Tracing — Product Spec.md`
- Technical spec: `Missing Person Tracing — Technical Spec.md`
- Design: `docs/design.md`
- Results: `docs/evaluation.md` (generated)
- Demo steps: `docs/demo-runbook.md`

## What it is not

It does not identify unknown people and keeps no database of everyone seen.
A search always starts from a photo or a description. Footage auto-deletes
after three days. All processing is local; no video leaves the machine.

## Setup

Requires Docker, an AMD GPU with ROCm for live use (CPU works for recorded
files), and roughly 30 GB of disk for the ROCm image plus datasets.

```bash
# 1. Model weights (OSNet, MSMT17-trained)
./scripts/fetch_weights.sh data/weights

# 2. Build. CPU:
docker compose build
#    ROCm (~22 GB image):
TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4 docker compose build

# 3. Start the database and web UI on http://localhost:8000
docker compose up -d db app

# 4. Tests
docker compose run --rm app pytest -v
```

## Recorded footage

```bash
python -m scripts.ingest_file clip.mp4 --camera cam1 --start 2026-10-13T14:00:00+07:00
python -m scripts.search_cli reference.jpg --cameras cam1 cam2
```

Pass `--protected` when ingesting evaluation footage so the retention job
never deletes it.

## Live cameras

Edit `config.yaml` with your cameras and the measured walk time between
them, then:

```bash
docker compose up -d mediamtx worker-cam1 worker-cam2
docker compose run --rm app python scripts/report_latency.py
```

- **cam1** — USB webcam on a Raspberry Pi 4. Run `pi/setup.sh` on the Pi.
- **cam2** — iPhone running Larix, pushing SRT to
  `srt://<pepper-ip>:8890?streamid=publish:cam2`. MediaMTX republishes it as
  RTSP, so the worker only ever speaks one protocol.
- **replay** — a recorded clip re-streamed in real time, for development and
  as a demo-night fallback.

## Evaluation

```bash
./scripts/fetch_market1501.sh data/market1501
python -m scripts.eval_market1501 --weights data/weights/osnet_x1_0_msmt17.pth
python -m scripts.inject_market1501 --identities 200
python -m scripts.tune_tau --identities 50
python -m scripts.eval_table          # regenerates docs/evaluation.md
```

Accuracy figures are measured with a single subject against a distractor
gallery, and every number states its distractor count. See
`docs/evaluation.md` for why, and what that limits the claims to.

## Configuration worth knowing

| Key | Default | Note |
|---|---|---|
| `tau.osnet` | 0.78 | Measured. 0.5 admits essentially everything |
| `tau.clip` | 0.25 | CLIP cosines are much lower than OSNet's |
| `sample_fps` | 10 | Drop to 5 if GPU headroom runs short |
| `retention_days` | 3 | `protected` rows are never deleted |
| `tracklet.max_tracklet_s` | 60 | Caps memory and keeps embeddings meaningful |
| `pairs[].min_walk_s` | 4 | Set from your measured walk between spots |
| `pairs[].overlapping` | false | Set true for cameras that see the same space |
```

- [ ] **Step 4: Write the demo runbook**

```markdown
# Demo runbook

Rehearse this once before recording. Total runtime about six minutes.

## Beforehand

- [ ] `./scripts/fetch_weights.sh data/weights`
- [ ] ROCm image built; `docker compose run --rm app python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] Pi powered, Ethernet in, `rtsp://raspberrypi.local:8554/cam1` opens in VLC
- [ ] iPhone on the same WiFi, Larix configured for `srt://<pepper-ip>:8890?streamid=publish:cam2`
- [ ] Walk between the two spots and time it; put that number in `config.yaml` as `min_walk_s`
- [ ] `docker compose up -d` and confirm both workers log "worker for camX reading…"
- [ ] A reference photo of the subject, taken separately from the demo clips
- [ ] Fallback ready: if the phone drops, point `cam2` at the `replay` path

## Running it

1. **Recorded search (MVP, R1–R6).** Ingest two clips, search by photo, show ranked
   results with camera, time, score, thumbnail and a playable clip.
2. **Timing check (R6).** Show a candidate dropped because the gap between cameras
   was shorter than the walk takes.
3. **Live walk-through (R7, R8).** Walk past cam1, then cam2. Wait for the timeline
   to refresh — polling every 3 s — then run `scripts/report_latency.py` and read out
   the median and worst case against the 10 s requirement.
4. **Timeline and review (R9, R10).** One lane per camera, sightings in time order,
   last known location highlighted. Reject a match and show it leave the timeline.
5. **Text search (R13).** Search "dark jacket" with no photo at all.
6. **Retention (R11).** Set `retention_days: 0`, wait for the sweep, show the log line
   naming what was deleted — then set it back to 3.
7. **Numbers.** Open `docs/evaluation.md`: the cross-domain benchmark, the τ sweep,
   and the own-footage rows with their distractor counts.

## Saying the limitations out loud

- One subject, so top-1 is measured against a stated distractor count rather than
  against several mutually confusable volunteers.
- MSMT17-trained weights evaluated on Market-1501 is cross-domain; the right
  comparison is the cross-domain literature, not the ~94% same-domain figures.
- Accuracy drops with changed clothing, lighting and camera angle. The jacket-change
  and dim-light rows are reported separately rather than averaged away.
```

- [ ] **Step 5: Do the live walk-through and run the final check**

```bash
docker compose up -d
# Walk past cam1, then cam2. Then run a search in the UI at localhost:8000.
docker compose run --rm app python scripts/report_latency.py
docker compose run --rm app pytest tests/test_final_acceptance.py -v
docker compose run --rm app python -m scripts.eval_table
```

Expected: `R8: PASS`, four acceptance tests passing, and `docs/evaluation.md` showing the live latency section filled in.

- [ ] **Step 6: Run the whole suite one last time**

Run: `docker compose run --rm app pytest -v`
Expected: PASS. Tests needing datasets or a live stream skip with a clear reason; nothing fails.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/demo-runbook.md tests/test_final_acceptance.py docs/evaluation.md
git commit -m "docs: README, demo runbook and executable final acceptance check"
```

**Phase 4 checkpoint (final release):** R1–R11 plus R13 are implemented, the evaluation table is generated from real runs, and both acceptance criteria exist as tests.

---

## Appendix: parallelism

Tasks 1–4 come first and block everything. After that:

| Lane | Tasks | Needs |
|---|---|---|
| Search path | 5, 6, 7, 8 | Fixtures only — no models, no GPU |
| Vision | 9, 10, 11, 12, 13 | GPU image, OSNet weights |
| Hardware | 17 (Pi and phone parts) | Physical access; no application code |
| Downloads | — | Market-1501, weights, ROCm image: unattended |

Tasks 14–16 integrate the two lanes, so they wait for both. Tasks 18–19 follow 17;
20–23 follow 16; task 24 is last.

Builder-only work, on the critical path for every reported number and blocked by
nothing in this plan: film at two spots in several outfits and lighting conditions,
measure the walk time between them, and hand-label `data/eval/ground_truth.csv`.
