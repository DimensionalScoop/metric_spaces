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
