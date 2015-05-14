import unittest
import nose.tools
import numpy as np
from sklearn import manifold, datasets
from ..coranking import trustworthiness, continuity, LCMC, coranking_matrix


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

    def test_basic_coranking_matrix_no_change(self):
        high = np.ones((10, 10))
        low = np.zeros((10, 10))
        Q = coranking_matrix(high, low)

        n, _ = high.shape
        nose.tools.assert_equal(Q.shape, (n-1, n-1))
        nose.tools.assert_equal(Q.sum(), 90)
        assert_diagonal_equal(Q, np.eye(n-1)*10)

    def test_basic_coranking_matrix_changed(self):
        high = np.arange(25).reshape(5, 5)
        low = high.copy()
        low[[0, 2], :] = high[[2, 0], :]  # swapping a row changes neighbours

        Q = coranking_matrix(high, low)

        n, _ = high.shape
        nose.tools.assert_equal(Q.shape, (n-1, n-1))
        nose.tools.assert_equal(Q.sum(), 20)
        exp_result = np.array([[4,  0,  0,  1],
                               [0,  2,  2,  1],
                               [0,  2,  3,  0],
                               [1,  1,  0,  3]])

        assert_diagonal_equal(Q, exp_result)

    def test_trustworthiness(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = trustworthiness(Q, 5)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.895582191)

    def test_trustworthiness_if_k_one(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = trustworthiness(Q, 1)
        nose.tools.assert_almost_equal(t, 0.923747203579)

    def test_trustworthiness_if_k_is_n(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = trustworthiness(Q, Q.shape[0])
        nose.tools.assert_almost_equal(t, 1.0)

    @nose.tools.raises(ValueError)
    def test_trustworthiness_throws_if_k_negative(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        trustworthiness(Q, -1)

    @nose.tools.raises(ValueError)
    def test_trustworthiness_throws_if_k_zero(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        trustworthiness(Q, 0)

    @nose.tools.raises(ValueError)
    def test_trustworthiness_throws_if_k_large(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        trustworthiness(Q, Q.shape[0]+1)

    def test_basic_coranking_matrix_no_change_trust(self):
        high = np.ones((10, 10))
        low = np.zeros((10, 10))
        Q = coranking_matrix(high, low)

        t = trustworthiness(Q, 1)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 1.0)

        t = trustworthiness(Q, 4)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 1.0)

    def test_basic_coranking_matrix_changed_trust(self):
        high = np.arange(25).reshape(5, 5)
        low = high.copy()
        low[[0, 2], :] = high[[2, 0], :]  # swapping a row changes neighbours

        Q = coranking_matrix(high, low)

        t = trustworthiness(Q, 1)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.866666666)

        t = trustworthiness(Q, 4)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 1.0)

    def test_continuity(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        c = continuity(Q, 5)

        nose.tools.assert_true(isinstance(c, float))
        nose.tools.assert_almost_equal(c, 0.982385844)

        c2 = trustworthiness(Q, 5)
        nose.tools.assert_true(c, c2)

    def test_continuity_if_k_one(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = continuity(Q, 1)
        nose.tools.assert_almost_equal(t, 0.990749440)

    def test_continuity_if_k_is_n(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = continuity(Q, Q.shape[0])
        nose.tools.assert_almost_equal(t, 1.0)

    @nose.tools.raises(ValueError)
    def test_continuity_throws_if_k_negative(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        continuity(Q, -1)

    @nose.tools.raises(ValueError)
    def test_continuity_throws_if_k_zero(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        continuity(Q, 0)

    @nose.tools.raises(ValueError)
    def test_continuity_throws_if_k_large(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        continuity(Q, Q.shape[0]+1)

    def test_basic_coranking_matrix_no_change_continuity(self):
        high = np.ones((10, 10))
        low = np.zeros((10, 10))
        Q = coranking_matrix(high, low)

        t = continuity(Q, 1)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 1.0)

        t = continuity(Q, 4)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 1.0)

    def test_basic_coranking_matrix_changed_continuity(self):
        high = np.arange(25).reshape(5, 5)
        low = high.copy()
        low[[0, 2], :] = high[[2, 0], :]  # swapping a row changes neighbours

        Q = coranking_matrix(high, low)

        t = continuity(Q, 1)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.866666666)

        t = continuity(Q, 4)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 1.0)

    def test_LCMC(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        l = LCMC(Q, 5)
        nose.tools.assert_true(isinstance(l, float))
        nose.tools.assert_almost_equal(l, 0.377201409)

    def test_LCMC_if_k_one(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = continuity(Q, 1)
        nose.tools.assert_almost_equal(t, 0.990749440)

    def test_LCMC_if_k_is_n(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        t = LCMC(Q, Q.shape[0])
        nose.tools.assert_almost_equal(t, -1.12230926e-05)

    @nose.tools.raises(ValueError)
    def test_LCMC_throws_if_k_negative(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        LCMC(Q, -1)

    @nose.tools.raises(ValueError)
    def test_LCMC_throws_if_k_zero(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        LCMC(Q, 0)

    @nose.tools.raises(ValueError)
    def test_LCMC_throws_if_k_large(self):
        Q = coranking_matrix(self._high_data, self._low_data)
        LCMC(Q, Q.shape[0]+1)

    def test_basic_coranking_matrix_no_change_LCMC(self):
        high = np.ones((10, 10))
        low = np.zeros((10, 10))
        Q = coranking_matrix(high, low)

        t = LCMC(Q, 1)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.98611111)

        t = LCMC(Q, 4)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.61111111)

    def test_basic_coranking_matrix_changed_LCMC(self):
        high = np.arange(25).reshape(5, 5)
        low = high.copy()
        low[[0, 2], :] = high[[2, 0], :]  # swapping a row changes neighbours

        Q = coranking_matrix(high, low)

        t = LCMC(Q, 1)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.666666666)

        t = LCMC(Q, 4)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, -0.083333333)


def assert_diagonal_equal(m, exp_result):
    """Check the elements on the diagonal of a matrix are equal """
    di = np.diag_indices(m.shape[0])
    np.testing.assert_array_equal(m[di], exp_result[di])
