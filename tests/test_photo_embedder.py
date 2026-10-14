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
        n = self.n

        class B:
            xyxy = torch.tensor([[10.0, 10.0, 60.0, 10.0 + 100 * (i + 1)]
                                 for i in range(n)])
            id = torch.tensor([float(i + 1) for i in range(n)])
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
    emb, dets = make(cfg, 3).embed_image(jpeg_bytes())
    areas = [d["area"] for d in dets]
    assert areas == sorted(areas, reverse=True)
    assert len(dets) == 3
    assert all("xyxy" in d for d in dets)


def test_no_person_raises_a_readable_error(cfg):
    with pytest.raises(ValueError, match="no person"):
        make(cfg, 0).embed_image(jpeg_bytes())


def test_undecodable_image_data_raises(cfg):
    with pytest.raises(ValueError, match="could not read"):
        make(cfg, 1).embed_image(b"this is not an image")


def test_the_embedding_belongs_to_the_largest_detection(cfg):
    emb, dets = make(cfg, 3).embed_image(jpeg_bytes())
    expected = unit_vector(np.full(512, 3.0, dtype=np.float32))
    assert np.allclose(emb, expected, atol=1e-5)
