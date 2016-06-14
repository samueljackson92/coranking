import _metrics_cy as metrics_cy
import numpy as np


def trustworthiness(Q, min_k=1, max_k=None):
    if not isinstance(Q, np.int64):
        Q = Q.astype(np.int64)

    if max_k is None:
        max_k = Q.shape[0]-1

    result = [metrics_cy.trustworthiness(Q, x) for x in range(min_k, max_k)]
    return np.array(result)


def check_square_matrix(M):
    if M.shape[0] != M.shape[1]:
        msg = "Expected square matrix, but matrix had dimensions (%d, %d)" % M.shape
        raise RuntimeError(msg)
