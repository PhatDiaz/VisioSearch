import numpy as np
import pytest

from src.retrieval import ground_truth_ranks, recall_at_k, search, unit_normalize


def test_normalization():
    vecs = unit_normalize(np.array([[3, 4], [0, -2]], dtype=float))
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1)
    assert np.allclose(vecs[0], [0.6, 0.8])


def test_search_topk_and_filter():
    gallery = unit_normalize(np.array([[1, 0], [0, 1], [-1, 0]], dtype=float))
    assert [hit.image_index for hit in search(np.array([1,0]), gallery, top_k=2)] == [0, 1]
    assert [hit.image_index for hit in search(np.array([1,0]), gallery, top_k=2, allowed_indices=[1,2])] == [1,2]


def test_rank_ties_stable():
    scores = np.array([[.9,.9,.1], [.9,.9,.1], [.4,.5,.6]])
    assert ground_truth_ranks(scores, np.array([0,1,0])).tolist() == [1,2,3]


def test_recall():
    assert recall_at_k(np.array([1,3,7,11])) == {'R@1':25.0,'R@5':50.0,'R@10':75.0}


def test_validation_errors():
    with pytest.raises(ValueError):
        unit_normalize(np.zeros(3))
    with pytest.raises(ValueError):
        search(np.array([1,0]), np.ones((3,3)))
