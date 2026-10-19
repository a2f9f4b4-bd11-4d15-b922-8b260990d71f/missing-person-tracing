from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncio
import contextlib
import json
import os

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from tracer.config import Config
from tracer.deps import get_config, get_embedder, get_store
from tracer.models import Match
from tracer.search import run_search
from tracer.store import VALID_STATUS, Store

@asynccontextmanager
async def _lifespan(app: FastAPI):
    from tracer.retention import retention_loop
    interval = float(os.environ.get("RETENTION_INTERVAL_S", 3600))
    task = asyncio.create_task(retention_loop(get_store(), get_config(), interval))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Missing Person Tracing", lifespan=_lifespan)
_WEB = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")
TEMPLATES = Jinja2Templates(directory=str(_WEB / "templates"))


class StatusPatch(BaseModel):
    status: str


def _match_json(m: Match, tz: timezone = timezone.utc) -> dict:
    return {"tracklet_id": m.tracklet_id, "camera_id": m.camera_id,
            "started_at": m.started_at.astimezone(tz).isoformat(),
            "ended_at": m.ended_at.astimezone(tz).isoformat(),
            "score": round(m.score, 4), "thumb_url": m.thumb_url,
            "clip_url": m.clip_url, "status": m.status}


def _parse_window(value, cfg: Config) -> datetime | None:
    """A browser's datetime-local field carries no offset. Reading it as UTC
    shifts a reviewer's window by their whole offset, so naive input is
    interpreted in config's `tz`.
    """
    if not value:
        return None
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=cfg.tzinfo)


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
    scope = s["cameras"] or list(cfg.cameras)
    result = run_search(store, cfg, s["embedding"], s["space"], s["window"],
                        scope, limit=limit)
    store.record_matches(search_id, [(m.tracklet_id, m.score) for m in result.matches])
    return {"matches": [_match_json(m, cfg.tzinfo)
                        for m in store.get_matches(search_id, limit)],
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
def get_timeline(search_id: int, cfg: Config = Depends(get_config),
                 store: Store = Depends(get_store)) -> dict:
    _load_search(store, search_id)
    lanes = store.timeline(search_id)
    return {"lanes": {cam: [_match_json(m, cfg.tzinfo) for m in ms]
                      for cam, ms in lanes.items()}}


def _lane_geometry(lanes: dict[str, list[Match]],
                   tz: timezone = timezone.utc) -> tuple[dict, str | None]:
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
            d = _match_json(m, tz)
            d.update(left_pct=round(left, 2), width_pct=round(min(width, 100 - left), 2))
            out[cam].append(d)
    newest = max(everything, key=lambda m: m.ended_at)
    return out, (f"{newest.camera_id} at "
                 f"{newest.ended_at.astimezone(tz).isoformat(timespec='seconds')}")


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
    form_data = await request.form()
    resp = await create_search(
        photo=form_data.get("photo") or None,
        text=form_data.get("text") or None,
        frm=_parse_window(form_data.get("from"), cfg),
        to=_parse_window(form_data.get("to"), cfg),
        cameras=form_data.getlist("cameras"),
        cfg=cfg, store=store, embedder=embedder,
    )
    search_id = json.loads(resp.body)["search_id"]
    return ui_results(request, search_id, 10, cfg, store)


@app.patch("/ui/searches/{search_id}/matches/{tracklet_id}", response_class=HTMLResponse)
def ui_patch_match(request: Request, search_id: int, tracklet_id: int, status: str,
                   cfg: Config = Depends(get_config),
                   store: Store = Depends(get_store)):
    patch_match(search_id, tracklet_id, StatusPatch(status=status), store)
    if status == "rejected":
        return HTMLResponse("")
    m = next(m for m in store.get_matches(search_id, 50) if m.tracklet_id == tracklet_id)
    return TEMPLATES.TemplateResponse(
        request, "_card.html", {"search_id": search_id, "m": _match_json(m, cfg.tzinfo)})


@app.get("/ui/searches/{search_id}/timeline", response_class=HTMLResponse)
def ui_timeline(request: Request, search_id: int, fragment: int = 0,
                cfg: Config = Depends(get_config),
                store: Store = Depends(get_store)):
    lanes, last_seen = _lane_geometry(store.timeline(search_id), cfg.tzinfo)
    ctx = {"search_id": search_id, "lanes": lanes, "last_seen": last_seen}
    name = "_lanes.html" if fragment else "timeline.html"
    return TEMPLATES.TemplateResponse(request, name, ctx)


@app.get("/media/{path:path}")
def media(path: str, cfg: Config = Depends(get_config)) -> FileResponse:
    """Serve a file from data/, refusing anything that escapes it."""
    root = Path(cfg.data_dir).resolve()
    try:
        target = (root / path).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        raise HTTPException(404, "not found")
    if not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)
