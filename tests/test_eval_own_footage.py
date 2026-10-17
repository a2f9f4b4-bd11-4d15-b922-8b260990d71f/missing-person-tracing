import pytest

from tracer.config import CameraCfg, Config
from tracer.fixtures import make_similar_embedding, make_tracklet
from scripts.eval_own_footage import load_truth, score_queries


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path, device="cpu", sample_fps=10, imgsz=640,
                  retention_days=3, tau={"osnet": 0.74, "clip": 0.25},
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


def test_a_single_subject_without_distractors_is_flagged_as_vacuous(store, cfg):
    """With one identity and nothing to confuse it, top-1 is 100% by
    construction. The number must not be reportable without that warning."""
    q = make_tracklet().embedding
    tid = store.insert_tracklet(make_tracklet(
        camera_id="cam1", embedding=make_similar_embedding(q, 0.95)))
    got = score_queries(store, cfg, [{"embedding": q, "truth": {tid}}], distractor_count=0)
    assert got["vacuous"] is True

    got = score_queries(store, cfg, [{"embedding": q, "truth": {tid}}],
                        distractor_count=900)
    assert got["vacuous"] is False


def test_truth_csv_reads_labelled_rows_and_ignores_unlabelled(tmp_path):
    csv = tmp_path / "gt.csv"
    csv.write_text(
        "tracklet_id,person,camera,started_at,duration_s,n_crops,thumb\n"
        "1,me,cam1,2026-10-13T14:00:00+00:00,4.0,10,thumbs/1.jpg\n"
        "2,me,cam2,2026-10-13T14:01:00+00:00,3.0,10,thumbs/2.jpg\n"
        "3,?,cam1,2026-10-13T14:02:00+00:00,2.0,10,thumbs/3.jpg\n"
        "4,,cam1,2026-10-13T14:03:00+00:00,2.0,10,thumbs/4.jpg\n"
        "5,friend,cam2,2026-10-13T14:04:00+00:00,5.0,10,thumbs/5.jpg\n")
    truth = load_truth(csv)
    assert truth == {"me": {1, 2}, "friend": {5}}
