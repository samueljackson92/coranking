""" Module implements a coranking matrix for checking the quality of a
lower dimensional mapping produce by manifold learning algorithms.

Reference:

Lee, John Aldo, and Michel Verleysen. "Rank-based quality assessment of
nonlinear dimensionality reduction." ESANN. 2008.
"""

import itertools
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def coranking_matrix(high_data, low_data):
    """Generate a co-ranking matrix from two data frames of high and low
    dimensional data.

    :param high_data: DataFrame containing the higher dimensional data.
    :param low_data: DataFrame containing the lower dimensional data.
    :returns: the co-ranking matrix of the two data sets.
    """
    n, m = high_data.shape
    high_distance = pairwise_distances(high_data)
    low_distance = pairwise_distances(low_data)

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)

    Q = Q[1:, 1:]  # remove rankings which correspond to themselves
    return Q


def trustworthiness(Q, K):
    """ Trustworthiness measures the number of hard extrusions present in a
    mapping. This can be thought of as a measure of the number of false
    positives.

    :param Q: the co-ranking matrix to calculate trustworthiness from
    :param K: the number of neighbours to use.
    """
    n, _ = Q.shape
    n += 1

    # Indicies for the lower left section of Q. Quantifying hard intrusions.
    it = itertools.product(range(K, n-1), range(K))
    summation = sum([(k - K) * Q[k, l] for k, l in it])

    g = _tc_normalisation_weight(K, n)
    w = 2.0 / g
    return 1 - w * summation


def continuity(Q, K):
    """ Continuity measures the number of hard extrusions present in a
    mapping. I can be thought of as a measure of the number of false negatives.

    :param Q: the co-ranking matrix to calculate continuity from
    :param K: the number of neighbours to use.
    """

    n, _ = Q.shape
    n += 1

    # Indicies for the upper right section of Q. Quantifying hard extrustions.
    it = itertools.product(range(K), range(K, n-1))
    summation = sum([(l - K) * Q[k, l] for k, l in it])

    g = _tc_normalisation_weight(K, n)
    w = 2.0 / g
    return 1 - w * summation


def _tc_normalisation_weight(K, n):
    """ Compute the normalisation weight for the trustworthiness and continuity
    measures.

    :param K: size of the neighbourhood.
    :param n: total size of matrix.
    """
    if K < (n/2):
        return n*K*(2*n - 3*K - 1)
    elif K >= (n/2):
        return n*(n - K)*(n - K - 1)


def LCMC(Q, K):
    """ The local continuity meta-criteria measures the number of mild
    intrusions and extrusions. This can be thought of as a measure of the
    number of true postives.

    :param Q: the co-ranking matrix to calculate continuity from
    :param K: the number of neighbours to use.
    """

    n = Q.shape[0] + 1

    # Indicies for the upper right section of Q. Quantifying true positives.
    it = itertools.product(range(K), range(K))
    summation = sum([Q[k, l] for k, l in it])

    return (K / (1. - n)) + (1. / (n*k)) * summation
