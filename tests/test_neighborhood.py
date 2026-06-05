import numpy as np

from rlmmo.core.neighborhood import knn_indices, knn_matrix
from rlmmo.core.niching import pairwise_distances


def test_knn_indices_include_and_exclude_self():
    pop = np.array([[0.0], [1.0], [2.0], [10.0]])
    dmat = pairwise_distances(pop)
    with_self = knn_indices(dmat, 1, 3, include_self=True)
    without_self = knn_indices(dmat, 1, 2, include_self=False)
    assert with_self[0] == 1
    assert 1 not in without_self
    assert without_self.shape == (2,)


def test_knn_matrix_shape_and_values():
    pop = np.arange(5, dtype=float).reshape(-1, 1)
    mat = knn_matrix(pop, 3, include_self=True)
    assert mat.shape == (5, 3)
    assert np.all(mat[:, 0] == np.arange(5))
