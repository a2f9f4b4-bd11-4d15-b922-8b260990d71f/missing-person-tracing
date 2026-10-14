from datetime import datetime, timedelta, timezone

import numpy as np

from tracer.reid import Crop, select_crops

T0 = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)


def crop(h: int, i: int = 0, edge_cut: bool = False) -> Crop:
    return Crop(image=np.full((h, h // 2, 3), i % 256, dtype=np.uint8),
                t=T0 + timedelta(milliseconds=100 * i), edge_cut=edge_cut)


def test_crops_shorter_than_the_minimum_height_are_dropped():
    got = select_crops([crop(63, 0), crop(64, 1), crop(200, 2)])
    assert [c.image.shape[0] for c in got] == [64, 200]


def test_edge_cut_crops_are_dropped():
    got = select_crops([crop(100, 0, edge_cut=True), crop(100, 1)])
    assert len(got) == 1
    assert got[0].t == T0 + timedelta(milliseconds=100)


def test_at_most_n_crops_are_returned():
    got = select_crops([crop(100, i) for i in range(40)], n=10)
    assert len(got) == 10


def test_crops_are_spread_evenly_over_time_not_taken_from_the_start():
    crops = [crop(100, i) for i in range(40)]
    got = select_crops(crops, n=5)
    assert got[0].t == crops[0].t
    assert got[-1].t == crops[-1].t
    gaps = [(b.t - a.t).total_seconds() for a, b in zip(got, got[1:])]
    assert max(gaps) - min(gaps) < 0.25


def test_results_stay_in_time_order():
    got = select_crops([crop(100, i) for i in range(20)], n=7)
    assert [c.t for c in got] == sorted(c.t for c in got)


def test_fewer_crops_than_requested_returns_all_usable_ones():
    got = select_crops([crop(100, 0), crop(100, 1)], n=10)
    assert len(got) == 2


def test_no_usable_crops_returns_empty():
    assert select_crops([crop(10, 0), crop(100, 1, edge_cut=True)]) == []


def test_empty_input_returns_empty():
    assert select_crops([]) == []
