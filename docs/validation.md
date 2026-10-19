# Validating this locally

Every step below shows you something, rather than asking you to trust a test
run. Start the stack first:

```bash
docker compose up -d db app mediamtx
```

## 1. The test suite (~3 min)

```bash
docker compose run --rm app pytest -q
```

Expect **194 passed, 4 skipped**. The 4 skips are the live-walk-through
assertions; they turn into real checks once you do step 6.

## 2. Your own acceptance criteria, as executable tests

```bash
docker compose run --rm app pytest tests/test_mvp_acceptance.py -v
```

Four tests, all passing. These are the product spec's MVP criterion in code:
both cameras produce tracklets, a photo returns matches from both ranked
best-first above τ, results order correctly in time, and the timing check
rejects physically impossible pairs.

## 3. The web UI

Open <http://localhost:8000>. You should see the four cameras listed with a
form. Then:

- Upload `data/reference.jpg` (prepared in step 4), tick cam1 and cam2, search
- You should get **two cards**, cam1 and cam2, each with a thumbnail, a time,
  a score near 0.90, and Confirm/Reject buttons
- Click a thumbnail — the clip opens and plays
- Click **Confirm** on one, **Reject** on the other; the rejected card
  disappears
- Follow "Open timeline" — one lane per camera, blocks positioned by time,
  with the last known location named underneath

## 4. Search from the command line

```bash
docker compose run --rm app python -m scripts.search_cli data/reference.jpg \
  --cameras cam1 cam2
```

Expect the same person on both cameras, ten minutes apart, around 0.90.

Honest caveat: `data/reference.jpg` is built from one tracklet's own
thumbnail, so it is a *smoke test* — it proves the pipeline end to end, not
that re-ID generalises. Only your own footage, with photos taken separately
from the clips, can show that.

## 5. Re-create the recorded demo from scratch

```bash
docker compose run --rm app python -c "from tracer.deps import get_store; get_store().reset()"
docker compose run --rm app python -m scripts.ingest_file data/samples/people.mp4 \
  --camera cam1 --start 2026-10-15T09:00:00+07:00
docker compose run --rm app python -m scripts.ingest_file data/samples/people.mp4 \
  --camera cam2 --start 2026-10-15T09:10:00+07:00
```

7 tracklets per camera, with clips under `data/clips/` and thumbnails under
`data/thumbs/`. Open any clip: the person should be visible *before* the
detection starts — that is the 1.5 s pre-roll.

## 6. Live ingest, with no camera attached

```bash
docker compose run --rm app sh -c 'rm -f data/metrics/latency.jsonl'
docker compose run -d --name val1 app python -m tracer.worker --camera replay
docker compose run -d --name val2 app python -m tracer.worker --camera replay2
sleep 150
docker logs val1 | tail -5                     # "new tracklet N on replay (...)"
docker compose run --rm app python scripts/report_latency.py
docker stop val1 val2 && docker rm val1 val2    # IMPORTANT: see note below
```

Expect `R8: PASS`, median around 2.2 s against the 10 s requirement, with rows
for both cameras. Two workers run concurrently on CPU without trouble.

> `timeout N docker compose run ...` kills the CLI but **not** the container,
> which then keeps ingesting unnoticed. Always stop workers by name.

## 7. Text search without a photo

```bash
docker compose run --rm app python -m scripts.inject_market1501 --identities 200
docker compose run --rm app python -m scripts.backfill_clip
docker compose run --rm app python -c "
from tracer.deps import get_config, get_store
from tracer.textsearch import embed_text
from tracer.search import run_search
cfg, store = get_config(), get_store()
cams = sorted({r[0] for r in store.conn.execute('SELECT DISTINCT camera_id FROM tracklet')})
res = run_search(store, cfg, embed_text('a person in a red shirt', cfg), 'clip',
                 (None, None), cams, limit=5)
print([(m.tracklet_id, round(m.score, 3)) for m in res.matches])"
```

Five matches with scores around 0.25–0.35. This also gives you a 901-tracklet
distractor gallery, which makes the UI's results realistic rather than
trivially correct.

## 8. Try to break it

```bash
# A photo with no person in it -> 422 with a readable message
docker compose run --rm app python -c "
import cv2, numpy as np
cv2.imwrite('data/empty.jpg', np.full((400,300,3), 200, np.uint8))"
curl -s -F photo=@data/empty.jpg -F from=2026-10-15T09:00 -F to=2026-10-15T10:00 \
  -F cameras=cam1 http://localhost:8000/ui/searches | head -3

# Reversed window -> 422
curl -s -o /dev/null -w '%{http_code}\n' -F photo=@data/reference.jpg \
  -F from=2026-10-15T10:00 -F to=2026-10-15T09:00 -F cameras=cam1 \
  http://localhost:8000/ui/searches

# Path traversal -> 404
curl -s -o /dev/null -w '%{http_code}\n' --path-as-is \
  http://localhost:8000/media/../etc/passwd

# Unknown camera -> 422 naming it
curl -s -F photo=@data/reference.jpg -F cameras=cam9 \
  http://localhost:8000/ui/searches | head -2
```

## 9. Read the numbers

```bash
docker compose run --rm app python -m scripts.eval_table
cat docs/evaluation.md
```

Check two things in particular: tier 2 says **"identities disjoint from the
reported split"** (if it ever says *LEAKED*, the tuning and reporting splits
have collided), and the top-1 figure is stated against the spec's 80% target —
currently **not met** at 69.3%.

## 10. Retention actually deletes

```bash
docker compose run --rm app python -c "
import asyncio, logging
from datetime import datetime, timedelta, timezone
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from tracer.deps import get_config, get_store
from tracer.retention import retention_loop
from tracer.fixtures import make_tracklet
store, cfg = get_store(), get_config()
old = datetime.now(timezone.utc) - timedelta(days=10)
tid = store.insert_tracklet(make_tracklet(camera_id='cam1', started_at=old,
                                          ended_at=old + timedelta(seconds=5)))
async def main():
    t = asyncio.create_task(retention_loop(store, cfg, interval_s=3))
    await asyncio.sleep(4); t.cancel()
asyncio.run(main())
print('stale tracklet gone:', store.conn.execute(
    'SELECT count(*) FROM tracklet WHERE id = %s', (tid,)).fetchone()[0] == 0)"
```

Expect a `retention sweep (...)` log line naming what it removed, then
`stale tracklet gone: True`. Nothing from the last three days is touched, and
`protected` rows never are.

## What this cannot tell you

- **Whether re-ID works on your cameras.** Everything above runs on public
  clips and Market-1501. Only filming two spots answers that.
- **Whether the Pi node works.** `pi/setup.sh`, its systemd unit and the
  `h264_v4l2m2m` encoder have never run.
- **Whether the iPhone path works.** cam2 has never carried a real frame; all
  live testing used the replay streams.

## Cleaning up

```bash
docker compose down                  # stop everything
docker compose run --rm app rm -rf data/clips data/thumbs data/metrics
```

Files under `data/` are root-owned because containers write them, so delete
them through the container rather than from the host.
