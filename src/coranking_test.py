import unittest
import nose.tools
from sklearn import manifold, datasets

from mia.coranking import trustworthiness, continuity, LCMC, coranking_matrix
from mia.utils import *


class CorankingTest(unittest.TestCase):

    def setUp(self):
        self._high_data, color \
            = datasets.samples_generator.make_swiss_roll(n_samples=300,
                                                         random_state=1)

        isomap = manifold.Isomap(n_neighbors=12, n_components=2)
        self._low_data = isomap.fit_transform(self._high_data)

    def test_coranking_matrix(self):
        Q = coranking_matrix(self._high_data, self._low_data)

        n, _ = self._high_data.shape
        nose.tools.assert_equal(Q.shape, (n-1, n-1))

    def test_trustworthiness(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = trustworthiness(Q, 5)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.895582191)

    def test_continuity(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        c = continuity(Q, 5)
        nose.tools.assert_true(isinstance(c, float))
        nose.tools.assert_almost_equal(c, 0.982385844)

        c2 = trustworthiness(Q, 5)
        nose.tools.assert_true(c, c2)

    def test_LCMC(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        l = LCMC(Q, 5)
        nose.tools.assert_true(isinstance(l, float))
        nose.tools.assert_almost_equal(l, 0.474110925)
