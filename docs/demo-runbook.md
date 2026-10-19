# Demo runbook

Rehearse once before recording. Runtime about six minutes.

## Beforehand

- [ ] `./scripts/fetch_weights.sh data/weights`
- [ ] ROCm image built; `docker run --rm --device /dev/kfd --device /dev/dri
      --group-add video -e HSA_OVERRIDE_GFX_VERSION=10.3.0 m2-app:rocm
      python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] Pi powered, Ethernet in, `rtsp://raspberrypi.local:8554/cam1` opens in VLC
- [ ] iPhone on the same WiFi, Larix pointed at
      `srt://<pepper-ip>:8890?streamid=publish:cam2`
- [ ] **Walk between the two spots and time it.** Put the *fastest* time in
      `config.yaml` as `min_walk_s` — it is a lower bound on possible transit,
      so an average would reject true matches whenever you walked briskly
- [ ] `docker compose up -d` and confirm both workers log
      `worker for camN reading rtsp://...`
- [ ] A reference photo of the subject, taken separately from the demo clips
- [ ] Fallback ready: if the phone drops, repoint `cam2` at the `replay` path

## Running it

1. **Recorded search (R1–R6).** Ingest two clips, search by photo, show ranked
   results with camera, time, score, thumbnail and a playable clip.
2. **Timing check (R6).** Show a candidate dropped because the gap between
   cameras was shorter than the walk takes. `scripts/make_two_camera_demo.sh`
   sets this up from one clip with a controllable offset.
3. **Live walk-through (R7, R8).** Walk past cam1, then cam2. Wait for the
   timeline to refresh — it polls every 3 s — then run
   `scripts/report_latency.py` and read out median and worst case against the
   10 s requirement.
4. **Timeline and review (R9, R10).** One lane per camera, sightings in time
   order, last known location highlighted. Reject a match and show it leave
   the timeline.
5. **Text search (R13).** Search "dark jacket" with no photo at all.
6. **Retention (R11).** Set `retention_days: 0` and `RETENTION_INTERVAL_S=5`,
   wait for the sweep, show the log line naming what was deleted — then set
   both back. Note the loop sleeps before its first sweep, so nothing is
   deleted on startup.
7. **Numbers.** Open `docs/evaluation.md`: the cross-domain benchmark, the
   AIN-vs-plain comparison, the τ sweep on the disjoint split, and the
   own-footage rows.

## Confirming the acceptance criteria

```bash
docker compose run --rm app pytest tests/test_mvp_acceptance.py -v      # R1-R6
docker compose run --rm app pytest tests/test_final_acceptance.py -v    # R7-R10
```

The final acceptance test reads the live metrics written in the last 30
minutes, so run it **right after** the walk-through. It skips with an
explanatory message if there are no recent live sightings.

## Saying the limitations out loud

- **One subject**, so top-1 is measured against a stated distractor count
  rather than against several mutually confusable volunteers.
- **The ≥80% top-1 target is not met**: 69.3% on Market-1501 identities.
- τ and the model were chosen on Market-1501's **train** identities, disjoint
  from the reported test split. An earlier version tuned on the test split and
  reported 100% — a leak, not a result. Worth mentioning as a methods point.
- MSMT17-trained weights on Market-1501 is **cross-domain**; the right
  comparison is the cross-domain literature, not the ~94% same-domain figures.
- Accuracy drops with changed clothing, lighting and camera angle. The
  jacket-change and dim-light rows are reported separately, not averaged away.
