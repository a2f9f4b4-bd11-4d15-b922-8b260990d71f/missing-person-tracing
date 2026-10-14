from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from tracer.reid import Crop, OsnetEmbedder

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
WEIGHTS = Path("data/weights/osnet_x1_0_msmt17.pth")


class FakeExtractor:
    """Returns rows we control, so the degenerate-embedding guards can be
    tested without depending on model behaviour."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, images):
        import torch
        return torch.tensor(np.asarray(self.rows[:len(images)], dtype=np.float32))


def crop(i: int, h: int = 200) -> Crop:
    return Crop(image=np.full((h, h // 2, 3), (i * 37) % 256, dtype=np.uint8),
                t=T0 + timedelta(milliseconds=100 * i))


def test_zero_embeddings_are_skipped_not_normalised():
    rows = [np.zeros(512), np.ones(512), np.ones(512) * 2]
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    emb.extractor = FakeExtractor(rows)
    vec, used = emb.embed_tracklet([crop(0), crop(1), crop(2)])
    assert used == 2
    assert np.all(np.isfinite(vec))
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_nan_embeddings_are_skipped():
    bad = np.ones(512)
    bad[3] = np.nan
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
        assert used == 10
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_model_name_is_inferred_from_the_weights_filename():
    """Loading AIN weights into the plain osnet_x1_0 architecture silently
    discards the mismatched layers, leaving a half-random model that still
    produces 512-d vectors — and therefore plausible but meaningless scores."""
    from tracer.reid import infer_model_name
    assert infer_model_name("data/weights/osnet_x1_0_msmt17.pth") == "osnet_x1_0"
    assert infer_model_name("data/weights/osnet_ain_x1_0_msmt17.pth") == "osnet_ain_x1_0"
    assert infer_model_name("/abs/path/OSNET_AIN_X1_0.PTH") == "osnet_ain_x1_0"
