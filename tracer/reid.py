from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from tracer.models import unit_vector

log = logging.getLogger(__name__)

MIN_CROP_HEIGHT = 64
MAX_CROPS = 10
OSNET_INPUT = (256, 128)


def infer_model_name(weights_path: str) -> str:
    """Pick the architecture the checkpoint belongs to.

    torchreid discards mismatched layers with only a log line, so loading an
    AIN checkpoint into osnet_x1_0 yields a half-random model that still emits
    512-d vectors. Inferring from the filename makes that mistake impossible.
    """
    return "osnet_ain_x1_0" if "ain" in Path(weights_path).name.lower() else "osnet_x1_0"


@dataclass(frozen=True)
class Crop:
    """One person crop taken from one frame, in RGB."""
    image: np.ndarray
    t: datetime
    edge_cut: bool = False

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def select_crops(crops: Sequence[Crop], n: int = MAX_CROPS,
                 min_h: int = MIN_CROP_HEIGHT) -> list[Crop]:
    """Up to `n` usable crops, spread evenly across the tracklet's lifetime.

    Crops shorter than `min_h` are too small to embed reliably, and crops
    touching a frame edge show only part of the person.
    """
    usable = [c for c in sorted(crops, key=lambda c: c.t)
              if c.height >= min_h and not c.edge_cut]
    if len(usable) <= n:
        return usable
    idx = np.linspace(0, len(usable) - 1, n).round().astype(int)
    return [usable[i] for i in dict.fromkeys(idx.tolist())]


class OsnetEmbedder:
    """OSNet appearance embeddings, 512-d and L2-normalised.

    Weights must be re-ID trained (MSMT17); torchreid's pretrained=True loads
    ImageNet weights, which do not work for re-identification.
    """

    def __init__(self, weights_path: str, device: str = "cpu",
                 model_name: str | None = None, extractor=None) -> None:
        if extractor is not None:
            self.extractor = extractor
            return
        from torchreid.utils import FeatureExtractor
        self.model_name = model_name or infer_model_name(weights_path)
        self.extractor = FeatureExtractor(
            model_name=self.model_name, model_path=weights_path, device=device)

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h < 2 or w < 2:
            raise ValueError(f"crop too small to resize: {image.shape}")
        return cv2.resize(image, (OSNET_INPUT[1], OSNET_INPUT[0]),
                          interpolation=cv2.INTER_LINEAR)

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        prepared = [self._prepare(im) for im in images]
        raw = self.extractor(prepared)
        out = np.asarray(raw.cpu() if hasattr(raw, "cpu") else raw, dtype=np.float32)
        out = np.atleast_2d(out)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (out / norms).astype(np.float32)

    def embed_tracklet(self, crops: Sequence[Crop]) -> tuple[np.ndarray, int]:
        """Mean of up to 10 evenly spread crop embeddings, L2-normalised."""
        chosen = select_crops(crops)
        if not chosen:
            raise ValueError("no usable crops in this tracklet")

        rows = self.embed_images([c.image for c in chosen])
        good = [r for r in rows if np.all(np.isfinite(r)) and np.linalg.norm(r) > 1e-6]
        dropped = len(rows) - len(good)
        if dropped:
            log.warning("dropped %d degenerate crop embedding(s)", dropped)
        if not good:
            raise ValueError("no usable crop embeddings in this tracklet")

        return unit_vector(np.mean(np.stack(good), axis=0)), len(good)


class PhotoEmbedder:
    """Embeds a reference photo through the same crop pipeline as ingest.

    A query embedded differently from the gallery is the classic silent re-ID
    bug: similarities stay plausible while ranking quietly degrades.
    """

    def __init__(self, cfg, weights: str | None = None,
                 detector=None, embedder=None) -> None:
        self.cfg = cfg
        if detector is not None:
            self._detect = detector
        else:
            from ultralytics import YOLO
            model = YOLO("yolo26n.pt")
            device = 0 if str(cfg.device) in ("cuda", "gpu") else "cpu"
            self._detect = lambda img: model.predict(
                img, classes=[0], conf=0.5, imgsz=cfg.imgsz,
                device=device, verbose=False)
        self.embedder = embedder or OsnetEmbedder(
            weights or getattr(cfg, "weights", None)
            or "data/weights/osnet_ain_x1_0_msmt17.pth", device=str(cfg.device))

    def embed_image(self, data: bytes) -> tuple[np.ndarray, list[dict]]:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not read that image file")

        result = self._detect(image)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes.xyxy) == 0:
            raise ValueError("no person detected in the reference photo")

        h, w = image.shape[:2]
        found: list[dict] = []
        crops: list[np.ndarray] = []
        for xyxy in boxes.xyxy.tolist():
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            found.append({"xyxy": [x1, y1, x2, y2], "area": (x2 - x1) * (y2 - y1)})
            crops.append(image[y1:y2, x1:x2, ::-1].copy())

        if not found:
            raise ValueError("no person detected in the reference photo")

        order = sorted(range(len(found)), key=lambda i: found[i]["area"], reverse=True)
        rows = self.embedder.embed_images([crops[i] for i in order])
        return unit_vector(np.asarray(rows)[0]), [found[i] for i in order]
