import numpy as np
from scipy.spatial import distance

__version__ = "0.1.1"


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


def pairwise_distances(X):
    return distance.squareform(distance.pdist(X))
