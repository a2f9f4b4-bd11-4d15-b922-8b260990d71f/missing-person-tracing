import os

import numpy as np
import pytest


def test_python_is_312():
    import sys
    assert sys.version_info[:2] == (3, 12)


def test_torch_imports_and_reports_device():
    import torch
    assert torch.__version__.startswith("2.")
    # DEVICE=cuda is only expected when the compose file passed the GPU through
    if os.environ.get("DEVICE") == "cuda":
        assert torch.cuda.is_available()


def test_ultralytics_and_lap_pinned():
    import ultralytics, lap  # noqa: F401
    assert ultralytics.__version__.startswith("8.4")


def test_torchreid_builds_osnet_with_512d_features():
    import torch
    import torchreid
    m = torchreid.models.build_model("osnet_x1_0", num_classes=1000, pretrained=False)
    m.eval()
    with torch.no_grad():
        feats = m(torch.randn(2, 3, 256, 128))
    assert feats.shape == (2, 512)


def test_database_is_reachable_with_pgvector():
    import psycopg
    from pgvector.psycopg import register_vector
    with psycopg.connect(os.environ["TRACER_DSN"]) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        got = conn.execute("SELECT '[1,0,0]'::vector").fetchone()[0]
        assert np.allclose(got, [1.0, 0.0, 0.0])
