import pytest

from tracer.config import CameraCfg, Config
from tracer.fixtures import make_similar_embedding, make_tracklet
from tracer.search import run_search


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.74, "clip": 0.25},
                  default_min_walk_s=10.0,
                  cameras={"cam1": CameraCfg(id="cam1", name="cam1", kind="file")})


def test_a_text_query_ranks_the_matching_tracklet_first(store, cfg):
    """The CLIP space uses its own threshold, so a 0.31 similarity — far below
    tau.osnet — is a legitimate text match."""
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
