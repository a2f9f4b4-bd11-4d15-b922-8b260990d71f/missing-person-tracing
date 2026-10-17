import numpy as np
import pytest

from tracer.evaluation import cmc_and_map


def test_a_perfect_ranking_scores_one():
    sims = np.array([[0.9, 0.1, 0.2]])
    got = cmc_and_map(query_ids=[1], query_cams=[1],
                      gallery_ids=[1, 2, 3], gallery_cams=[2, 2, 2], sims=sims)
    assert got["rank1"] == 1.0
    assert got["rank5"] == 1.0
    assert got["mAP"] == 1.0


def test_a_wrong_top_match_fails_rank1_but_can_pass_rank5():
    sims = np.array([[0.1, 0.9, 0.2]])
    got = cmc_and_map([1], [1], [1, 2, 3], [2, 2, 2], sims)
    assert got["rank1"] == 0.0
    assert got["rank5"] == 1.0
    assert 0.0 < got["mAP"] < 1.0


def test_same_camera_gallery_items_are_excluded_from_scoring():
    """The standard protocol ignores gallery images of the same person on the
    same camera; counting them would inflate every number reported."""
    sims = np.array([[0.99, 0.5, 0.1]])
    got = cmc_and_map([1], [1], [1, 1, 2], [1, 2, 2], sims)
    assert got["rank1"] == 1.0
    assert got["mAP"] == 1.0


def test_a_query_with_no_valid_gallery_match_is_skipped():
    sims = np.array([[0.9, 0.8]])
    got = cmc_and_map([1], [1], [1, 1], [1, 1], sims)
    assert got["n_queries"] == 0
    assert got["rank1"] == 0.0


def test_multiple_correct_matches_contribute_to_map():
    sims = np.array([[0.9, 0.8, 0.1]])
    got = cmc_and_map([1], [1], [1, 1, 2], [2, 3, 2], sims)
    assert got["rank1"] == 1.0
    assert got["mAP"] == pytest.approx(1.0)


def test_averaging_over_several_queries():
    sims = np.array([[0.9, 0.1], [0.1, 0.9]])
    got = cmc_and_map([1, 2], [1, 1], [1, 2], [2, 2], sims)
    assert got["rank1"] == 1.0
    assert got["n_queries"] == 2
