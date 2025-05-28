"""Fast implementations for calculating distances between all points in a vector."""

from numba import njit
import numpy as np


@njit(boundscheck=False, fastmath=True)
def pnorm(pts, p: int):
    n = len(pts)
    dists = np.empty((n, n), dtype="float32")
    np.fill_diagonal(dists, 0)

    for x in range(n):
        for y in range(x, n):
            if p % 2 == 0:
                d = (pts[x] - pts[y]) ** p
            else:
                d = np.abs(pts[x] - pts[y]) ** p
            d = np.sum(d) ** (1 / p)
            dists[x, y] = d
            dists[y, x] = d

    return dists


@njit(boundscheck=False, fastmath=True)
def euclid(pts):
    n = len(pts)
    dists = np.empty((n, n), dtype="float32")
    np.fill_diagonal(dists, 0)
    for x in range(n):
        for y in range(x, n):
            res = np.sqrt(np.sum((pts[x] - pts[y]) ** 2))
            dists[x, y] = res
            dists[y, x] = res

    return dists


@njit(boundscheck=False, fastmath=True)
def euclidean_distance_matrix(x, y):
    """
    Compute Euclidean distance matrix between two sets of vectors.

    Parameters:
    -----------
    x : array-like, shape (m, k)
        First set of vectors
    y : array-like, shape (n, k)
        Second set of vectors

    Returns:
    --------
    distances : ndarray, shape (m, n)
        Distance matrix where distances[i, j] is the distance between x[i] and y[j]
    """
    m, k = x.shape
    n = y.shape[0]
    distances = np.empty((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            dist_sq = 0.0
            for d in range(k):
                diff = x[i, d] - y[j, d]
                dist_sq += diff * diff
            distances[i, j] = dist_sq

    return np.sqrt(distances)
