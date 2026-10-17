"""R13: find a person from a description instead of a photo.

CLIP vectors are not comparable with OSNet vectors, so they live in their own
column and carry their own threshold. `search.space` records which space a
query belongs to; mixing them would return confident nonsense.
"""
from __future__ import annotations

import functools
import logging

import numpy as np

from tracer.models import unit_vector

log = logging.getLogger(__name__)

PROMPT = "a photo of {}"


class ClipEncoder:
    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", device: str = "cpu") -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if device in ("cuda", "gpu") else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @staticmethod
    def validate_text(text: str) -> str:
        if not text or not text.strip():
            raise ValueError("text query must not be empty")
        return text.strip()

    def encode_text(self, text: str) -> np.ndarray:
        text = self.validate_text(text)
        tokens = self.tokenizer([PROMPT.format(text)]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
        return unit_vector(feats[0].cpu().numpy())

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        batch = self.torch.stack([
            self.preprocess(Image.fromarray(img)) for img in images]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_image(batch)
        arr = feats.cpu().numpy().astype(np.float32)
        return np.stack([unit_vector(row) for row in arr])


@functools.lru_cache(maxsize=1)
def _encoder(device: str) -> ClipEncoder:
    return ClipEncoder(device=device)


def embed_text(text: str, cfg) -> np.ndarray:
    """Entry point used by the API when a text query arrives."""
    return _encoder(str(cfg.device)).encode_text(text)
