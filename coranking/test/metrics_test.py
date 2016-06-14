from coranking.coranking import coranking_matrix
from coranking.metrics import trustworthiness
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
    print result
    assert False


def make_datasets():
    high_data, color \
        = datasets.samples_generator.make_swiss_roll(n_samples=300,
                                                     random_state=1)

    isomap = manifold.Isomap(n_neighbors=12, n_components=2)
    low_data = isomap.fit_transform(high_data)

    return high_data, low_data
