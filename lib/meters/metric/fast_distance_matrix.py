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


@njit(boundscheck=False, fastmath=True)
def euclidean_range_query_hits(points, queries, r):
    """for each query with radius `r`, count the number of points that are within the query.
    returns the sum of all hits."""
    m, k = points.shape
    n = queries.shape[0]
    r_sq = r**2
    hits = 0

    for i in range(m):
        for j in range(n):
            dist_sq = 0.0
            for d in range(k):
                diff = points[i, d] - queries[j, d]
                dist_sq += diff * diff
                if dist_sq > r_sq:
                    break
            hits += dist_sq < r_sq

    return hits


@njit(boundscheck=False, fastmath=True, cache=True)
def euclidean_range_query_hits_c(points, queries, r):
    """for each query with radius `r`, count the number of points that are within the query.
    returns the sum of all hits."""
    m, k = points.shape
    n = queries.shape[0]
    r_sq = r**2
    hits = 0

    for i in range(m):
        for j in range(n):
            dist_sq = 0.0
            for d in range(k):
                diff = points[i, d] - queries[j, d]
                dist_sq += diff * diff
                if dist_sq > r_sq:
                    break
            hits += dist_sq < r_sq

    return hits


@njit(boundscheck=False, fastmath=True, cache=True)
def euclidean_range_query_hits_v4(points, queries, r):
    """Optimized version 4: Block processing for better cache usage"""
    m, k = points.shape
    n = queries.shape[0]
    r_sq = r * r
    hits = 0

    # Process in blocks for better cache locality
    block_size = min(1024, m)

    for j in range(n):
        query = queries[j]
        for block_start in range(0, m, block_size):
            block_end = min(block_start + block_size, m)

            for i in range(block_start, block_end):
                dist_sq = 0.0
                for d in range(k):
                    diff = points[i, d] - query[d]
                    dist_sq += diff * diff
                    if dist_sq > r_sq:
                        break
                if dist_sq <= r_sq:
                    hits += 1
    return hits


@njit(boundscheck=False, fastmath=True)
def euclidean_range_query_hits_batched(points, queries, r):
    """for each query with radius `r`, count the number of points that are within the query.
    returns the sum of all hits."""
    n_batches, m, k = points.shape
    n = queries.shape[1]
    r_sq = r**2

    hits = np.zeros(n_batches, dtype=np.int32)

    for b in range(n_batches):
        for i in range(m):
            for j in range(n):
                dist_sq = 0.0
                for d in range(k):
                    diff = points[b, i, d] - queries[b, j, d]
                    dist_sq += diff * diff
                    if dist_sq > r_sq:
                        break
                hits[b] += dist_sq < r_sq

    return hits
