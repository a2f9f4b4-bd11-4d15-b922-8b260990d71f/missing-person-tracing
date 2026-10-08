# Missing Person Tracing — Design

Oct 13, 2026 · design for the MVP and final releases described in
`Missing Person Tracing — Product Spec.md` and `Missing Person Tracing — Technical Spec.md`

## 1. Intent

From a single reference photo, find where and when that person appeared across two or
more cameras, and present the sightings as a ranked, time-ordered timeline with clips, so
a reviewer checks minutes of video instead of hours.

Whole-body appearance is the primary matching signal. A human confirms every match. The
system never identifies unknown people and never browses "everyone seen today" — a search
always starts from a photo (or, with R13, a description).

**Target.** Both releases carry equal weight: the MVP's recorded-footage search (R1–R6)
and the final's live cameras plus timeline (R7–R11), plus one stretch feature. Confirmed
with the builder on 13/10.

**Constraints that shaped this design.**

- Five days remain (13/10 → 18/10), evenings only on weekdays. The original plan allotted
  eleven days and no code existed on 13/10.
- All AI runs on pepper. The Raspberry Pi 4 only captures and streams.
- One subject is available for filming. No volunteers.
- Python 3.14 is the only host interpreter, and PyTorch requires ≤3.13. Containers are
  therefore mandatory, not a packaging preference.

**Cut order if the schedule slips**, carried over from the product spec: the stretch
feature first, then own-footage τ tuning. The live demo and the timeline are never cut.

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | ROCm with a first-class CPU fallback (`DEVICE=cpu`) | Measured 2.6× speedup; two live cameras are infeasible on CPU |
| 2 | `torchreid` for model code, MSMT17 weights from HuggingFace | Author's own repo mirrors the model zoo; removes Google Drive from the dependency chain |
| 3 | Same-clip smoke test **and** Market-1501 injection | Public clips have one person at a time and no shared identity, so they cannot validate ranking |
| 4 | Two camera spots, short non-overlapping walk | Keeps R6 a real feature; overlapping fields of view would make it a no-op |
| 5 | Single subject against a distractor gallery, limitation disclosed | With one identity and no distractors, top-1 is vacuously 100% |
| 6 | R13 text search via CLIP as the stretch feature | Visibly works at CCTV distances, and doubles as the CLIP-ReID fallback |
| 7 | Retention 3 days, with a `protected` exemption | Matches the "first hours matter" premise; protects the evaluation set from deletion |

**Cameras.** `cam1` is the USB webcam on the Pi 4 (both in hand). `cam2` is an iPhone 11
running Larix. The MacBook webcam is a spare, usable as `cam3` or as a cam2 substitute.

## 3. Probe evidence

All figures measured on pepper on 13/10 in throwaway containers. They are the basis for
the decisions above and for the corrections in section 9.

**Detector and tracker.** `ultralytics` 8.4.176 (published 12/10). `yolo26n.pt` loads and
detects `person`; the spec's named detector stands. ByteTrack requires `lap>=0.5.12`,
which Ultralytics auto-installs at runtime — the image must pin it instead.

**Throughput** (`yolo26n` + ByteTrack, 150-frame runs):

| Configuration | fps |
|---|---|
| CPU, 768×432 | 10.9 |
| ROCm GPU, 768×432 | 28.5 |
| ROCm GPU, 720p, one worker | 14.1 |
| ROCm GPU, 720p, two concurrent workers | 11.4 + 11.5 = **22.9 aggregate** |

Two processes produce more aggregate throughput than one, so a single worker does not
saturate the GPU and per-frame overhead dominates. Two cameras at 10 fps need 20 fps
aggregate against 22.9 measured — roughly 15% headroom before embedding and clip writing
take their share. Sampling rate and `imgsz` are therefore config knobs, with the 5 fps
fallback held in reserve.

**ROCm.** `torch` 2.9.1+rocm6.4, HIP 6.4.43484. `torch.cuda.is_available()` is `True`
with `/dev/kfd`, `/dev/dri`, `--group-add video` and `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
A 4000³ matmul ×20 runs in 0.21 s. Image sizes: 2.93 GB CPU, **22.2 GB ROCm** — budget
disk accordingly.

**Re-ID.** `torchreid` on PyPI is 0.2.5, last published 2022-10-16, and unusable. The
GitHub package installs only with `build-essential`, sixteen preinstalled dependencies,
and `--no-build-isolation --no-deps`; its `setup.py` imports the package to read a version
string, so every runtime dependency must exist at build time. Calling `build_model` with
`pretrained=True` fetches **ImageNet** weights, which are inadequate for re-ID. The
re-ID weights come
from `kaiyangzhou/osnet` on HuggingFace:
`osnet_x1_0_msmt17_combineall_256x128_...pth` (17.3 MB, download verified), alongside
`osnet_ain_x1_0_msmt17_...pth`. `FeatureExtractor` returns 512-d embeddings;
`osnet_x1_0` is 2,193,616 params / 978,878,352 flops.

`boxmot` 25.0.0 is **not** a weight source: `create_reid_encoder(spec: ReIDEncoderSpec)`
expects a caller-supplied `artifact`, and its factory and specs modules contain no OSNet
references. It was evaluated and rejected.

**Detection quality on sample footage** (151 frames): identical results at conf 0.5 and
0.25 — 50 detections, 101 empty frames, max 1 person per frame. Box heights min 143,
median 194, max 263 px; none below the 64 px crop floor.

**Re-ID signal** (OSNet + MSMT17, spec crop rules, tracks split in half to synthesise
same-person tracklet pairs):

```
same-person  (7 pairs):  min 0.791   mean 0.842
diff-person (21 pairs):  max 0.873   mean 0.720
τ = 0.5 would keep 7/7 same and admit 21/21 different
```

Same-person similarity exceeds different-person similarity, so the core premise holds.
Three caveats constrain how far these numbers go: the clip carries no identity labels, so
"different-person" pairs are assumed rather than known, and since the clip shows roughly
one person at a time while yielding 7 track IDs, the 0.873 outlier is plausibly a
same-person pair under a switched track ID; the footage was upscaled from 768×432, so
crops are soft and these figures are a floor; and 7 track IDs for fewer people confirms
tracker fragmentation is real.

**Datasets.** `aveocr/Market-1501-v15.09.15.zip` on HuggingFace mirrors the original
archive (research terms apply, cite Zheng et al.). Sample clips: `people-detection.mp4`
and `store-aisle-detection.mp4` from intel-iot-devkit.

## 4. Architecture

```
 SOURCES                                 PEPPER
 Pi4 + USB webcam  ──RTSP──┐
 iPhone 11 (Larix) ──SRT───┼──▶ MediaMTX ──▶ ingest worker (one process per camera)
 file replay       ──RTSP──┘                   sample → YOLO26 + ByteTrack
 video files       ───────────────────────▶     → merge → close tracklet → OSNet embed
                                                          │
                                       ═══════════════════▼═══════════════════
                                        Postgres 16 + pgvector  ·  data/ on disk
                                       ═══════════════════╤═══════════════════
                                                          ▼
                                        FastAPI: search (candidates → τ → timing)
                                                 + JSON API + Jinja2/HTMX UI
                                                 + hourly retention task
```

**The tracklet record is the only interface between the vision half and the database
half.** Everything upstream of it is computer vision; everything downstream is database
and web work. This boundary is what allows the two work lanes in section 10 to proceed
simultaneously.

**MediaMTX runs on pepper**, not only on the Pi as originally specified. Sources speak
whatever they speak — RTSP from the Pi, SRT from the iPhone, RTSP from the file-replay
harness — and every worker opens a uniform `rtsp://localhost:8554/<camera_id>`. Adding the
MacBook or swapping a failed phone for a replayed file becomes a config change. This also
resolves a factual problem with the original spec: IP Webcam is Android-only, and iOS
RTSP *publishing* is unreliable, whereas Larix pushes SRT dependably.

**One OS process per camera.** Ultralytics holds ByteTrack state inside the model instance
under `persist=True`, so two cameras sharing one model object would corrupt each other's
track IDs. Separate processes also contain failures: a stuttering phone stream cannot take
down `cam1` mid-demo. The cost is one set of model weights per process, which the
two-worker measurement shows is affordable.

**Containers.** `db` (pgvector), `app` (API, UI, retention task), `mediamtx`, and one
`worker` per camera from `config.yaml`. Workers receive `/dev/kfd` and `/dev/dri`;
`DEVICE=cpu` skips the GPU entirely.

## 5. Data model

Four tables, as originally specified, with five changes. Clips, thumbnails and reference
photos live on disk under `data/`, referenced by relative path.

| Change | Reason |
|---|---|
| `tracklet.clip_emb VECTOR(512)` nullable | R13 — CLIP space, populated by a second pass |
| `tracklet.protected BOOLEAN NOT NULL DEFAULT false` | Retention must not delete the evaluation set |
| `search.space TEXT NOT NULL DEFAULT 'osnet'` | A text query lives in CLIP space; see section 6 |
| `match.status` CHECK (`pending\|confirmed\|rejected`) | Enforced by the database, not by convention |
| `camera.kind TEXT` (`file\|rtsp`) | Workers dispatch on source type |

**The tracklet contract.** Lane 1 produces these; Lane 2 consumes them.

```python
@dataclass(frozen=True)
class TrackletRecord:
    camera_id: str
    started_at: datetime      # tz-aware, UTC
    ended_at: datetime        # >= started_at; duration >= 1.0s
    embedding: np.ndarray     # float32, shape (512,), ||v||₂ == 1 ± 1e-5
    thumb_path: str           # relative to data/
    clip_path: str
    n_crops: int              # 1..10
```

A `fixtures.make_tracklet()` generator emitting valid random records lets search, API, UI
and retention be built and tested before YOLO or OSNet exist.

**Query exactly; keep the HNSW index for the growth story.** pgvector applies a WHERE
filter *after* the index scan, so `ORDER BY embedding <=> q LIMIT 50` under a selective
time-window filter can return far fewer than 50 true candidates — degrading recall in
precisely the narrow-window case a reviewer cares about. At a few thousand tracklets an
exact scan is correct and comfortably inside the 2 s target. The index stays defined, and
the report explains the choice.

## 6. Search and scoring

**Query path.** Reference photo → YOLO detects people → crop → OSNet embed **through the
same crop pipeline as ingest**, since a query embedded differently from the gallery is the
classic silent re-ID bug. No person detected returns 422 with a plain message. Multiple
people returns all detections ranked by box area with the largest pre-selected, which is
what the product spec's user flow requires.

**Candidates.** Exact cosine scan filtered by time window and camera, top 50.

**Threshold.** Per embedding space, because the spaces have different scales:
`tau: {osnet: 0.78, clip: 0.25}`. A single τ would make text search return either
everything or nothing. The CLIP value is calibrated in tier 2 of the evaluation.

**Timing check (R6).** Process candidates greedily in descending score order; keep one if
it is temporally compatible with every already-kept sighting, otherwise drop it. This is
deterministic, unlike the original wording, which was order-dependent. Gaps are measured
**end-to-start**, not centre-to-centre:

```python
def compatible(a, b, cfg) -> bool:
    if a.camera_id == b.camera_id:
        return True                      # a person can reappear on one camera
    first, second = sorted((a, b), key=lambda t: t.started_at)
    gap = (second.started_at - first.ended_at).total_seconds()
    return gap >= cfg.min_walk(a.camera_id, b.camera_id)   # 0 for overlapping FOV
```

Overlapping intervals give a negative gap and are incompatible unless the camera pair is
configured as overlapping. Walk times and overlap flags live in `config.yaml`, per pair.

**Ranking.** Surviving sightings by score, top 10. The timeline shows the same set in time
order.

**Empty results carry a reason.** "Nothing above threshold, best was 0.71" and "nothing in
that time window" are different answers for a reviewer under time pressure.

**Reviewer feedback is display-only**, as originally specified: confirmed sightings form
the route, rejected ones leave the timeline, nothing retrains. Query expansion from
confirmed matches is noted as future work and deliberately not built.

## 7. Live ingest, timeline, retention

**R8 latency budget** — from a person leaving frame to the sighting being searchable:

| Stage | Cost |
|---|---|
| Track-lost timeout before the tracklet closes | 2.0 s |
| Buffering `end + 1 s` of frames for the clip | 1.0 s |
| Crop selection + OSNet embedding, 10 crops on GPU | ~0.3 s |
| Thumbnail write + database insert | <0.1 s |
| **Total** | **~3.4 s**, against a 10 s requirement |

**RTSP buffering misreports time.** OpenCV will hand over frames one to three seconds
stale while the decoder drains its backlog, inflating every timestamp and corrupting the
timing check. Workers run a grab-loop that discards the backlog and keeps only the newest
frame, with low-latency FFmpeg capture options. Live timestamps are arrival times at read.

**Tracklets are capped at 60 s.** Writing a clip from `start − 1 s` to `end + 1 s` means
buffering the tracklet's whole life; a person loitering for ten minutes at 720p/10 fps is
roughly 6,000 frames and several GB per camera. The cap also improves re-ID, since a mean
embedding over ten minutes averages across too much pose variation to be meaningful. On
reaching the cap the tracklet closes and a new one begins.

**Streams drop.** Workers reconnect with exponential backoff and flush open tracklets on
disconnect rather than leaving them open indefinitely.

**Same-camera tracklet merging is baseline behaviour**, not a fallback. Merge consecutive
same-camera tracklets whose gap is under 3 s and whose similarity is high, before writing
the record. The probe produced 7 track IDs on a clip showing roughly one person at a time,
so fragmentation is confirmed rather than hypothetical.

**Timeline uses HTMX polling, not SSE.** SSE was a stretch item and R13 took that slot. A
`hx-trigger="every 3s"` poll on the timeline fragment meets the 10 s requirement with no
event-stream lifecycle and no reconnection logic. One lane per camera on a shared time
axis scaled to the search window, sightings as positioned blocks, last known location
highlighted, plain CSS grid, no charting library.

**Retention** is an hourly asyncio task inside the app container — no extra container, no
host cron. It deletes tracklets older than `retention_days` (default 3) unless
`protected`, removes their clips and thumbnails, cascades `match` rows, sweeps orphaned
files left by crashed ingest runs, deletes reference photos of closed searches, and logs
counts rather than deleting silently. The demo runs with a minutes-long override.

## 8. Evaluation

**Tier 1 — Market-1501, model alone.** Standard protocol; rank-1, rank-5, mAP. The
published ~94% rank-1 OSNet figures are for a model **trained on Market-1501**; these
weights are trained on MSMT17, making this a *cross-domain* evaluation where published
numbers are substantially lower. The report compares against the cross-domain table in the
OSNet-AIN paper, with the exact figures verified from the paper at write-up time. Both
`osnet_x1_0` and `osnet_ain_x1_0` are run — AIN is built for unseen-domain generalisation,
so it is expected to win, and one config flag decides it.

**Tier 2 — Market-1501 through the real search path.** Crops injected as tracklets with
assigned cameras and timestamps, searched via the API. This measures the system — crop
selection, mean-of-crops embedding, τ, the timing check — rather than the model alone, and
it is where τ is tuned by sweeping across a range to fill the "wrong matches passing τ"
column. Timestamp assignments include both compatible and physically impossible pairs so
the timing check is exercised in both directions.

**Tier 3 — own footage.** One subject in several outfits and lighting conditions, searched
against a gallery padded with Market-1501 identities and with people tracked from the
public clips. Every number reports its distractor count. The clothing-change and dim-light
clips are **separate rows, not averaged in**, since they are the known weakness. Ground
truth is a hand-labelled CSV of `tracklet_id, person`.

**Live latency** is reported as a distribution — median and worst case over N
walk-throughs, instrumented as `last_seen → tracklet_closed` and `last_seen → db_insert`.
One lucky run is not evidence.

**The table is generated by one script**, with fixed seeds for distractor sampling, so a
late change to τ cannot leave stale numbers in the report.

## 9. Corrections to the original specs

| Original | Corrected | Why |
|---|---|---|
| τ starts at 0.5 | τ starts at **0.78** | Measured: 0.5 admits 21/21 different-person pairs — it filters nothing |
| Top-1 ≥ 80% on own clips, 3–4 people | Top-1 against a **stated distractor count**, one subject | No volunteers; one identity makes top-1 vacuous |
| MediaMTX on the Pi only | Also on **pepper**, as protocol front door | Workers then speak one protocol; sources vary freely |
| "Larix on iOS" for RTSP | Larix pushes **SRT**, MediaMTX republishes RTSP | iOS RTSP publishing is unreliable; IP Webcam is Android-only |
| Tracklet merging as a fallback | **Baseline** behaviour | Fragmentation measured, not hypothetical |
| HNSW vector search | **Exact** scan; index retained | pgvector post-filters, under-returning on narrow windows |
| SSE for live updates (stretch) | HTMX polling | R13 took the stretch slot; polling meets the requirement |
| One stretch: R12 **or** R13, decided 17/10 | **R13** decided now | Avoids a schema migration; R12 likely measures nothing at these distances |
| Retention days open | **3 days**, with `protected` | Matches the premise; protects evaluation data |
| `torchreid` via pip, MSMT17 weights | GitHub + 16 deps + weights from HuggingFace | PyPI package is from 2022; `pretrained=True` gives ImageNet weights |

## 10. Work lanes

The tracklet contract in section 5 is fixed first. Then two lanes proceed independently
and meet at the database.

**Lane 1, model-dependent:** `ingest.py` (file and RTSP sources), `detect_track.py`
(YOLO26 + ByteTrack → merge → tracklets → clips), `reid.py` (crop selection, embeddings).
Live ingest is an additional *source type* here, not new code elsewhere.

**Lane 2, no models required:** `store.py`, `search.py` (candidates, τ, timing check —
pure logic, unit-testable immediately), `api.py`, `web/`, `retention.py`. Built against
`fixtures.make_tracklet()` until integration.

**Hardware lane, independent of both:** Pi MediaMTX configuration, iPhone/Larix setup,
the file-replay harness.

**Builder-only, on the critical path for every reported number:** filming at two spots in
several outfits and lighting conditions, measuring the walk time between spots for
`min_walk`, and hand-labelling the ground-truth CSV.

**Unattended downloads, blocking nothing:** Market-1501 (~1.8 GB), model weights,
container images.

Serial chain: contract → Lane 1 integration → real embeddings in the database → τ tuning →
own-footage numbers.

## 11. Out of scope

Identifying unknown people; any database of everyone seen; city-scale deployment or real
CCTV integration; mobile apps, accounts, multi-tenancy; running models on the Pi;
production security hardening; retraining from reviewer feedback; R12 face matching.
