# Missing Person Tracing

From one photo of a missing person, find where and when they appeared across
several cameras — ranked, time-ordered, with clips — so a reviewer checks
minutes of video instead of hours.

People are matched by whole-body appearance (clothing, build, colour) using
OSNet re-identification, with a physical-plausibility check that drops
sightings too close together in time to be the same person walking. Every
match is a suggestion with a score; a human confirms or rejects each one.

- Product spec: `Missing Person Tracing — Product Spec.md`
- Technical spec: `Missing Person Tracing — Technical Spec.md`
- Design: `docs/design.md`
- Báo cáo dự án: `docs/report-vi.txt`
- Results: `docs/evaluation.md` (generated — do not hand-edit)
- Demo steps: `docs/demo-runbook.md`
- Validate it yourself: `docs/validation.md`

## What it is not

It does not identify unknown people and keeps no database of everyone seen. A
search always starts from a photo or a description. Footage auto-deletes after
three days. All processing is local; no video leaves the machine.

## Setup

Requires Docker, roughly 30 GB of disk, and an AMD GPU with ROCm for live use
(CPU is fine for recorded files).

```bash
# 1. Model weights (OSNet, MSMT17-trained)
./scripts/fetch_weights.sh data/weights

# 2. Build. CPU:
docker compose build
#    ROCm (~23 GB image, built as a separate tag):
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4 \
  -t m2-app:rocm .

# 3. Database and web UI on http://localhost:8000
docker compose up -d db app

# 4. Tests (~3 min; they use their own `tracer_test` database)
docker compose run --rm app pytest -q
```

## Recorded footage

```bash
python -m scripts.ingest_file clip.mp4 --camera cam1 --start 2026-10-13T14:00:00+07:00
python -m scripts.search_cli reference.jpg --cameras cam1 cam2
```

Pass `--protected` when ingesting evaluation footage so the retention job
never deletes it.

## Live cameras

Put your cameras and the **measured** walk time between them in `config.yaml`,
then:

```bash
docker compose up -d mediamtx worker-cam1 worker-cam2
docker compose run --rm app python scripts/report_latency.py
```

- **cam1** — USB webcam on a Raspberry Pi 4. Run `pi/setup.sh` on the Pi.
- **cam2** — iPhone running Larix, pushing SRT to
  `srt://<pepper-ip>:8890?streamid=publish:cam2`. MediaMTX republishes it as
  RTSP, so the worker only ever speaks one protocol. (IP Webcam is
  Android-only and iOS RTSP *publishing* is unreliable, hence SRT.)
- **replay** — a recorded clip re-streamed in real time, for development and
  as a demo-night fallback.

MediaMTX runs on the host network, so containers reach it at
`host.docker.internal:8554` via a host-gateway alias, not at `mediamtx:8554`.

## Evaluation

```bash
./scripts/fetch_market1501.sh data/market1501
python -m scripts.eval_market1501 --weights data/weights/osnet_ain_x1_0_msmt17.pth
python -m scripts.inject_market1501 --identities 200     # distractor gallery
python -m scripts.tune_tau                               # tunes on the TRAIN split
python -m scripts.eval_table                             # regenerates docs/evaluation.md
```

**Measured** (see `docs/evaluation.md` for the full tables):

| | rank-1 | rank-5 | mAP |
|---|---|---|---|
| `osnet_ain_x1_0`, Market-1501 cross-domain | 70.3% | 84.4% | 43.7% |
| `osnet_x1_0`, same protocol | 66.7% | 81.0% | 40.5% |

AIN wins, so it is the configured default. R8 live latency: median 2.6s,
worst 5.6s, against a 10s requirement.

**The product spec's ≥80% top-1 target is not met** — the best end-to-end
top-1 is 69.3% on Market-1501 identities. The report says so rather than
quoting a flattering number.

### Why tuning uses the train split

τ and the model choice are selected on Market-1501's `bounding_box_train`
identities, which are disjoint from the `query`/`bounding_box_test` identities
the benchmark reports. Tuning on the reported split leaks it: doing so
produced a top-1 of 100% that collapsed to 69% once the leak was removed.
`scripts/eval_table.py` marks the tier-2 table **LEAKED** if the tuning split
ever overlaps the reported one.

## Configuration worth knowing

| Key | Default | Note |
|---|---|---|
| `weights` | `osnet_ain_x1_0_msmt17.pth` | Measured better cross-domain |
| `tz` | `+07:00` | The web form's date pickers send no offset; naive input is read in this zone |
| `tau.osnet` | 0.74 | Tuned on the disjoint train split. 0.5 filters nothing |
| `tau.clip` | 0.25 | CLIP cosines are much lower, and vary per prompt |
| `sample_fps` | 10 | Drop to 5 if GPU headroom runs short |
| `retention_days` | 3 | `protected` rows are never deleted |
| `tracklet.max_tracklet_s` | 60 | Caps memory and keeps embeddings meaningful |
| `pairs[].min_walk_s` | 4 | Set from your measured walk between spots |
| `pairs[].overlapping` | false | True for cameras seeing the same space |

## Known issues

- **Files under `data/` are root-owned**, because containers write them as
  root. Delete them through the container:
  `docker compose run --rm app rm -rf data/clips/...`. A `user:` directive on
  the services would fix it, but the torch cache lives under `/root`.
- **`timeout N docker compose run ...` leaves the container running** — the
  timeout kills the CLI, not the container. Use
  `docker compose up -d worker-camN` and `docker compose stop`, or workers keep
  ingesting unnoticed.
- **`tau.clip` is prompt-sensitive.** Top text matches land 0.25–0.35 while
  gallery medians sit 0.17–0.23, so a fixed threshold is thin. A relative rule
  (top-k with a floor) would be sturdier.
- **One subject only**, so own-footage top-1 is measured against a stated
  distractor count rather than against mutually confusable volunteers.
