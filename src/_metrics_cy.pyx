import numpy as np
cimport numpy as np

def trustworthiness(np.ndarray[np.int64_t, ndim=2] Q, Py_ssize_t K):
    cdef Py_ssize_t i, j
    cdef double summation = 0.0

    cdef double norm_weight = _tc_normalisation_weight(K, Q.shape[0]);
    cdef double w = 2.0 / norm_weight;
    
    for k in range(K, Q.shape[0]):
        for l in range(K):
            summation += w * (k - K) * Q[k, l]

    return 1.0 - summation;

def _tc_normalisation_weight(K, n):
    """ Compute the normalisation weight for the trustworthiness and continuity
    measures.

    :param K: size of the neighbourhood.
    :param n: total size of matrix.
    """
    if K < (n/2):
        return n*K*(2*n - 3*K - 1)
    elif K >= (n/2):
        return n*(n - K)*(n - K)
