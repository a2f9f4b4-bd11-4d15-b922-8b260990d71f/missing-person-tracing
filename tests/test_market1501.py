from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracer.market1501 import (group_by_identity_camera, inject, list_images,
                               parse_name)

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def test_filenames_parse_into_person_and_camera():
    assert parse_name("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_name("1500_c6s3_086567_01.jpg") == (1500, 6)


def test_junk_and_distractor_ids_parse_too():
    assert parse_name("-1_c2s3_065187_03.jpg") == (-1, 2)
    assert parse_name("0000_c1s1_000151_01.jpg") == (0, 1)


def test_an_unparseable_name_raises():
    with pytest.raises(ValueError, match="unexpected"):
        parse_name("not-a-market-filename.jpg")


@pytest.fixture
def fake_dataset(tmp_path) -> Path:
    """Three identities over two cameras, four crops each."""
    root = tmp_path / "bounding_box_test"
    root.mkdir()
    for pid in (1, 2, 3):
        for cam in (1, 2):
            for k in range(4):
                img = np.full((128, 64, 3), (pid * 60 + cam * 20 + k) % 256, np.uint8)
                cv2.imwrite(str(root / f"{pid:04d}_c{cam}s1_{k:06d}_01.jpg"), img)
    for k in range(2):
        cv2.imwrite(str(root / f"-1_c1s1_{k:06d}_01.jpg"),
                    np.zeros((128, 64, 3), np.uint8))
    return tmp_path


def test_listing_skips_junk_images(fake_dataset):
    paths = list_images(fake_dataset, "bounding_box_test")
    assert len(paths) == 24
    assert all(parse_name(p.name)[0] > 0 for p in paths)


def test_grouping_produces_one_group_per_identity_and_camera(fake_dataset):
    groups = group_by_identity_camera(list_images(fake_dataset, "bounding_box_test"))
    assert len(groups) == 6
    assert all(len(v) == 4 for v in groups.values())
    assert (1, 1) in groups and (3, 2) in groups


class StubEmbedder:
    def embed_images(self, images):
        from tracer.models import unit_vector
        out = []
        for img in images:
            rng = np.random.default_rng(int(np.asarray(img).mean()) % 97)
            out.append(unit_vector(rng.standard_normal(512).astype(np.float32)))
        return np.stack(out)

    def embed_tracklet(self, crops):
        from tracer.models import unit_vector
        rows = self.embed_images([c.image for c in crops])
        return unit_vector(rows.mean(axis=0)), min(len(crops), 10)


def test_injection_writes_one_tracklet_per_identity_camera_group(store, fake_dataset):
    paths = list_images(fake_dataset, "bounding_box_test")
    mapping = inject(store, StubEmbedder(), paths, base_time=T0, spacing_s=60.0)
    assert set(mapping) == {1, 2, 3}
    assert all(len(v) == 2 for v in mapping.values())
    assert store.conn.execute("SELECT count(*) FROM tracklet").fetchone()[0] == 6


def test_injected_tracklets_use_market_camera_ids(store, fake_dataset):
    inject(store, StubEmbedder(), list_images(fake_dataset, "bounding_box_test"),
           base_time=T0, spacing_s=60.0)
    cams = {r[0] for r in store.conn.execute(
        "SELECT DISTINCT camera_id FROM tracklet").fetchall()}
    assert cams == {"m1", "m2"}


def test_injected_tracklets_are_spaced_so_timing_never_rejects_them(store, fake_dataset):
    inject(store, StubEmbedder(), list_images(fake_dataset, "bounding_box_test"),
           base_time=T0, spacing_s=60.0)
    rows = store.conn.execute(
        "SELECT started_at, ended_at FROM tracklet ORDER BY started_at").fetchall()
    gaps = [(b[0] - a[1]).total_seconds() for a, b in zip(rows, rows[1:])]
    assert all(g >= 10.0 for g in gaps), f"smallest gap {min(gaps)}s"


def test_injected_tracklets_are_protected_from_retention(store, fake_dataset):
    inject(store, StubEmbedder(), list_images(fake_dataset, "bounding_box_test"),
           base_time=T0, spacing_s=60.0, protected=True)
    assert store.conn.execute(
        "SELECT bool_and(protected) FROM tracklet").fetchone()[0] is True


def test_injected_thumb_paths_resolve_under_the_data_directory(store, fake_dataset, tmp_path):
    """Paths are stored relative to data_dir. Storing "data/market1501/..."
    verbatim yields "<data_dir>/data/market1501/..." — a doubled prefix that
    breaks every thumbnail in the UI and skips every row in the CLIP backfill.
    """
    paths = list_images(fake_dataset, "bounding_box_test")
    inject(store, StubEmbedder(), paths, base_time=T0, data_dir=fake_dataset)

    stored = [r[0] for r in store.conn.execute("SELECT thumb_path FROM tracklet").fetchall()]
    assert stored, "nothing injected"
    for rel in stored:
        assert not rel.startswith("/"), f"{rel} should be relative"
        assert (fake_dataset / rel).exists(), f"{fake_dataset / rel} missing"
