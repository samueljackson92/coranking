import _metrics_cy as metrics_cy
import numpy as np


def trustworthiness(Q, min_k=1, max_k=None):
    """Compute the trustwortiness metric over a range of K values.

    Args:
        Q (array_like): coranking matrix
        min_k (Optional[int]): the lowest K value to compute. Default 1.
        max_k (Optional[int]): the highest K value to compute. If None the
            range of values will be computer from min_k to n-1

    Returns:
        array of size min_k - max_k with the corresponding trustworthiness
        values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [metrics_cy.trustworthiness(Q, x) for x in range(min_k, max_k)]
    return np.array(result)


def continuity(Q, min_k=1, max_k=None):
    """Compute the continuity metric over a range of K values.

    Args:
        Q (array_like): coranking matrix
        min_k (Optional[int]): the lowest K value to compute. Default 1.
        max_k (Optional[int]): the highest K value to compute. If None the
            range of values will be computer from min_k to n-1

    Returns:
        array of size min_k - max_k with the corresponding continuity values.
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

    Args:
        Q (array_like): coranking matrix
        min_k (Optional[int]): the lowest K value to compute. Default 1.
        max_k (Optional[int]): the highest K value to compute. If None the
            range of values will be computer from min_k to n-1

    Returns:
        array of size min_k - max_k with the corresponding LCMC values.
    """
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [metrics_cy.LCMC(Q, x) for x in range(min_k, max_k)]
    return np.array(result)


def check_square_matrix(M):
    if M.shape[0] != M.shape[1]:
        msg = "Expected square matrix, but matrix had dimensions (%d, %d)" % M.shape
        raise RuntimeError(msg)
