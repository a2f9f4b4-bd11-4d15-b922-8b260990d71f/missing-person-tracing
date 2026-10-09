import pytest

from tracer.config import load_config, Config

YAML = """
data_dir: /app/data
device: cpu
sample_fps: 10
imgsz: 640
retention_days: 3
tau:
  osnet: 0.78
  clip: 0.25
tracklet:
  lost_track_s: 2.0
  min_tracklet_s: 1.0
  max_tracklet_s: 60.0
  merge_gap_s: 3.0
  merge_sim: 0.8
default_min_walk_s: 10.0
cameras:
  cam1:
    name: Corridor
    kind: rtsp
    source_url: rtsp://localhost:8554/cam1
    location: corridor
  cam2:
    name: Doorway
    kind: rtsp
    source_url: rtsp://localhost:8554/cam2
    location: doorway
pairs:
  - cameras: [cam1, cam2]
    min_walk_s: 4.0
  - cameras: [cam1, cam3]
    overlapping: true
"""


@pytest.fixture
def cfg(tmp_path) -> Config:
    p = tmp_path / "config.yaml"
    p.write_text(YAML)
    return load_config(str(p))


def test_data_dir_is_always_a_path(cfg):
    from pathlib import Path
    assert isinstance(cfg.data_dir, Path)
    assert (cfg.data_dir / "clips").name == "clips"


def test_loads_cameras_and_scalars(cfg):
    assert set(cfg.cameras) == {"cam1", "cam2"}
    assert cfg.cameras["cam1"].kind == "rtsp"
    assert cfg.tau["osnet"] == 0.78
    assert cfg.retention_days == 3
    assert cfg.max_tracklet_s == 60.0


def test_min_walk_is_zero_for_the_same_camera(cfg):
    # A person can leave and re-enter one camera with no delay at all.
    assert cfg.min_walk("cam1", "cam1") == 0.0


def test_min_walk_uses_the_configured_pair_value(cfg):
    assert cfg.min_walk("cam1", "cam2") == 4.0


def test_min_walk_is_symmetric(cfg):
    assert cfg.min_walk("cam2", "cam1") == cfg.min_walk("cam1", "cam2")


def test_overlapping_pairs_have_no_minimum(cfg):
    # Overlapping fields of view see the same person simultaneously, which is
    # legitimate, so the timing check must not reject it.
    assert cfg.min_walk("cam1", "cam3") == 0.0


def test_unconfigured_pairs_fall_back_to_the_default(cfg):
    assert cfg.min_walk("cam2", "cam9") == 10.0


def test_overlapping_is_reported_separately_from_min_walk(cfg):
    assert cfg.overlapping("cam1", "cam3") is True
    assert cfg.overlapping("cam3", "cam1") is True
    assert cfg.overlapping("cam1", "cam2") is False
    assert cfg.overlapping("cam2", "cam9") is False


def test_tau_of_an_unknown_space_is_an_error(cfg):
    with pytest.raises(KeyError):
        cfg.tau["nonexistent"]


def test_weights_default_to_the_ain_checkpoint(cfg):
    assert cfg.weights.endswith("osnet_ain_x1_0_msmt17.pth")


def test_weights_can_be_overridden(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML + "\nweights: /custom/osnet_x1_0_msmt17.pth\n")
    assert load_config(str(p)).weights == "/custom/osnet_x1_0_msmt17.pth"
