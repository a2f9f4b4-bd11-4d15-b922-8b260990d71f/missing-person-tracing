import numpy as np
import pytest

from tracer.textsearch import ClipEncoder

try:
    import open_clip  # noqa: F401
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


def test_the_encoder_rejects_an_empty_query():
    with pytest.raises(ValueError, match="empty"):
        ClipEncoder.validate_text("   ")


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
        red = np.zeros((128, 64, 3), np.uint8)
        red[:, :, 0] = 220
        blue = np.zeros((128, 64, 3), np.uint8)
        blue[:, :, 2] = 220
        t = enc.encode_text("a red shirt")
        imgs = enc.encode_images([red, blue])
        assert float(imgs[0] @ t) > float(imgs[1] @ t)

    def test_text_image_similarities_sit_near_the_clip_threshold(self, enc):
        """CLIP text-image cosines live around 0.2-0.35, which is why tau is
        per-space: the osnet value would reject everything."""
        t = enc.encode_text("a person walking")
        img = np.full((128, 64, 3), 120, np.uint8)
        sim = float(enc.encode_images([img])[0] @ t)
        assert -0.1 < sim < 0.6
