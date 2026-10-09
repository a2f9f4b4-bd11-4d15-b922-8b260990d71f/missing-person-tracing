from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# AIN measures better cross-domain: rank-1 70.35% vs 66.75% on Market-1501.
DEFAULT_WEIGHTS = "data/weights/osnet_ain_x1_0_msmt17.pth"


@dataclass(frozen=True)
class CameraCfg:
    id: str
    name: str
    kind: str
    source_url: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("file", "rtsp"):
            raise ValueError(f"camera {self.id}: kind must be 'file' or 'rtsp', got {self.kind!r}")


@dataclass(frozen=True)
class PairCfg:
    min_walk_s: float = 0.0
    overlapping: bool = False


@dataclass(frozen=True)
class Config:
    data_dir: Path
    device: str
    sample_fps: float
    imgsz: int
    retention_days: int
    tau: dict[str, float]
    default_min_walk_s: float
    cameras: dict[str, CameraCfg]
    pairs: dict[frozenset[str], PairCfg] = field(default_factory=dict)
    lost_track_s: float = 2.0
    min_tracklet_s: float = 1.0
    max_tracklet_s: float = 60.0
    merge_gap_s: float = 3.0
    merge_sim: float = 0.8
    weights: str = DEFAULT_WEIGHTS
    tz: str = "UTC"

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, Path):
            object.__setattr__(self, "data_dir", Path(self.data_dir))

    def min_walk(self, a: str, b: str) -> float:
        """Minimum seconds a person needs to get from camera `a` to camera `b`."""
        if a == b:
            return 0.0
        pair = self.pairs.get(frozenset((a, b)))
        if pair is None:
            return self.default_min_walk_s
        if pair.overlapping:
            return 0.0
        return pair.min_walk_s

    def overlapping(self, a: str, b: str) -> bool:
        """Whether two cameras see shared space, so simultaneous sightings of
        one person are legitimate. A zero min_walk cannot express this: the
        gap between overlapping sightings is negative, not zero."""
        pair = self.pairs.get(frozenset((a, b)))
        return bool(pair and pair.overlapping)


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    cameras = {
        cid: CameraCfg(
            id=cid,
            name=c["name"],
            kind=c["kind"],
            source_url=c.get("source_url"),
            location=c.get("location"),
        )
        for cid, c in (raw.get("cameras") or {}).items()
    }

    pairs: dict[frozenset[str], PairCfg] = {}
    for p in raw.get("pairs") or []:
        a, b = p["cameras"]
        pairs[frozenset((a, b))] = PairCfg(
            min_walk_s=float(p.get("min_walk_s", 0.0)),
            overlapping=bool(p.get("overlapping", False)),
        )

    t = raw.get("tracklet") or {}
    return Config(
        data_dir=Path(raw["data_dir"]),
        device=raw.get("device", "cpu"),
        sample_fps=float(raw.get("sample_fps", 10)),
        imgsz=int(raw.get("imgsz", 640)),
        retention_days=int(raw.get("retention_days", 3)),
        tau=dict(raw["tau"]),
        default_min_walk_s=float(raw.get("default_min_walk_s", 10.0)),
        cameras=cameras,
        pairs=pairs,
        lost_track_s=float(t.get("lost_track_s", 2.0)),
        min_tracklet_s=float(t.get("min_tracklet_s", 1.0)),
        max_tracklet_s=float(t.get("max_tracklet_s", 60.0)),
        merge_gap_s=float(t.get("merge_gap_s", 3.0)),
        merge_sim=float(t.get("merge_sim", 0.8)),
        weights=str(raw.get("weights", DEFAULT_WEIGHTS)),
        tz=str(raw.get("tz", "UTC")),
    )
