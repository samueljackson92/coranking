import os

# Must mock the C module for read the docs as they have
# no support for compiling C code
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    from unittest.mock import MagicMock
    _metrics_cy = MagicMock()
else:
    import coranking._metrics_cy as metrics_cy

import numpy as np


def trustworthiness(Q, min_k=1, max_k=None):
    """Compute the trustwortiness metric over a range of K values.

    :param Q: coranking matrix
    :param min_k: the lowest K value to compute. Default 1.
    :param max_k: the highest K value to compute. If None the range of values
        will be computer from min_k to n-1

    :returns: array of size min_k - max_k with the corresponding
        trustworthiness values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [metrics_cy.trustworthiness(Q, x) for x in range(min_k, max_k)]
    return np.array(result)


def continuity(Q, min_k=1, max_k=None):
    """Compute the continuity metric over a range of K values.

    :param Q: coranking matrix
    :param min_k: the lowest K value to compute. Default 1.
    :param max_k: the highest K value to compute. If None the range of values
        will be computer from min_k to n-1

    :returns: array of size min_k - max_k with the corresponding continuity
        values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [metrics_cy.continuity(Q, x) for x in range(min_k, max_k)]
    return np.array(result)


def LCMC(Q, min_k=1, max_k=None):
    """Compute the local continuity meta-criteria (LCMC) metric over a range of
    K values.

    :param Q: coranking matrix
    :param min_k: the lowest K value to compute. Default 1.
    :param max_k: the highest K value to compute. If None the range of values
        will be computer from min_k to n-1

    :returns: array of size min_k - max_k with the corresponding LCMC values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [metrics_cy.LCMC(Q, x) for x in range(min_k, max_k)]
    return np.array(result)


def _check_square_matrix(M):
    if M.shape[0] != M.shape[1]:
        msg = "Expected square matrix, but matrix had dimensions (%d, %d)" % M.shape
        raise RuntimeError(msg)
