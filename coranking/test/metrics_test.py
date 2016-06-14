from coranking.coranking import coranking_matrix
from coranking.metrics import trustworthiness, continuity, LCMC
from nose.tools import *
import numpy as np
from sklearn import manifold, datasets

def test_trustworthiness():
    high_data, low_data = make_datasets()

    Q = coranking_matrix(high_data, low_data)
    t = trustworthiness(Q.astype(np.int64), min_k=5, max_k=6)

    assert_almost_equal(t, 0.895, places=3)


def test_trustworthiness_array():
    high_data, low_data = make_datasets()

    Q = coranking_matrix(high_data, low_data)
    result = trustworthiness(Q)

    assert_equal(result.shape, (297, ))


def test_continuity():
    high_data, low_data = make_datasets()

    Q = coranking_matrix(high_data, low_data)
    c = continuity(Q, 5, 6)

    assert_almost_equal(c, 0.982, places=3)

    c2 = trustworthiness(Q, 5, 6)
    assert_true(c, c2)


def test_continuity_array():
    high_data, low_data = make_datasets()

    Q = coranking_matrix(high_data, low_data)
    result = continuity(Q)

    assert_equal(result.shape, (297, ))


def test_LCMC():
    high_data, low_data = make_datasets()

    Q = coranking_matrix(high_data, low_data)
    l = LCMC(Q, 5, 6)

    assert_almost_equal(l, 0.377, places=3)


def test_LCMC_array():
    high_data, low_data = make_datasets()

    Q = coranking_matrix(high_data, low_data)
    result = LCMC(Q)

    assert_equal(result.shape, (297, ))

def make_datasets():
    high_data, color \
        = datasets.samples_generator.make_swiss_roll(n_samples=300,
                                                     random_state=1)

    isomap = manifold.Isomap(n_neighbors=12, n_components=2)
    low_data = isomap.fit_transform(high_data)

    return high_data, low_data
