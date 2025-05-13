import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from joblib import delayed, Parallel
import time

from generate import point_generator
from metric.metric import PNorm, Euclid


@njit(boundscheck=False, fastmath=True)
def dist_matrix_slow(pts, p: int):
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
def dist_matrix_great(pts, p: int):
    n = len(pts)
    dists = np.empty((n, n), dtype="float32")
    np.fill_diagonal(dists, 0)

    for x in range(n):
        for y in range(x, n):
            if p % 2 == 0:
                dists[x, y] = np.sum((pts[x] - pts[y]) ** p) ** (1 / p)
            else:
                dists[x, y] = np.sum(np.abs(pts[x] - pts[y]) ** p) ** (1 / p)
            dists[y, x] = dists[x, y]

    return dists


@njit(boundscheck=False, fastmath=True)
def dist_matrix(pts, p: int):
    n = len(pts)
    dists = np.empty((n, n), dtype="float32")
    np.fill_diagonal(dists, 0)

    if p % 2 == 0:
        for x in range(n):
            for y in range(x, n):
                d = np.pow((pts[x] - pts[y]), p)
                d = np.sqrt(np.sum(d))
                dists[x, y] = d
                dists[y, x] = d
    else:
        for x in range(n):
            for y in range(x, n):
                d = np.abs(pts[x] - pts[y]) ** p
                d = np.sum(d) ** (1 / p)
                dists[x, y] = d
                dists[y, x] = d

    return dists


@njit(boundscheck=False, fastmath=True)
def dist_matrix_euclid(pts):
    n = len(pts)
    dists = np.empty((n, n), dtype="float32")
    np.fill_diagonal(dists, 0)
    for x in range(n):
        for y in range(x, n):
            res = np.sqrt(np.sum((pts[x] - pts[y]) ** 2))
            dists[x, y] = res
            dists[y, x] = res

    return dists


class EuclidNumba(Euclid):
    def distance_matrix(self, a, b, threshold=1000000, rank_only=False):
        if a is b:
            return dist_matrix_euclid(a)
        else:
            return super().distance_matrix(a, b, threshold, rank_only)


class PNormNumba(PNorm):
    def distance_matrix(self, a, b, threshold=1000000, rank_only=False):
        if a is b:
            return dist_matrix(a, self.p)
        else:
            return super().distance_matrix(a, b, threshold, rank_only)


class PNormNumbaSlow(PNorm):
    def distance_matrix(self, a, b, threshold=1000000, rank_only=False):
        if a is b:
            return dist_matrix_slow(a, self.p)
        else:
            return super().distance_matrix(a, b, threshold, rank_only)


class PNormNumbaGreat(PNorm):
    def distance_matrix(self, a, b, threshold=1000000, rank_only=False):
        if a is b:
            return dist_matrix_great(a, self.p)
        else:
            return super().distance_matrix(a, b, threshold, rank_only)


def run(seed, metric, n_samples=20):
    rng = np.random.default_rng(seed)
    samples = range(n_samples)
    # if seed % 10 == 0:
    #     samples = tqdm(samples)
    for _ in samples:
        points = point_generator.generate_gaussian_points(rng, 3000, 7, False)
        res = metric.distance_matrix(points, points)
    return res


metrics = [
    # Euclid(),
    EuclidNumba(),
    # PNormNumba(),
    # PNormNumbaSlow(),
    # PNormNumbaGreat(),
]
results = []

# warm-up
for m in metrics:
    desired = run(0, metric=Euclid(), n_samples=1)
    actual = run(0, metric=m, n_samples=1)
    assert np.allclose(desired, actual)

for _ in range(2):
    np.random.shuffle(metrics)
    for m in tqdm(metrics):
        start_time = time.perf_counter()
        Parallel(16)(delayed(run)(i, m, 10) for i in range(64))
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        results.append(dict(name=m.__class__.__name__, time=execution_time))

results = pd.DataFrame(results)
print(results.groupby("name").agg(["sum", "std"]))

#                       time
#                        sum       std
# name
# Euclid           49.728331  0.023555
# EuclidNumba      20.615095  0.350627
# PNormNumba       73.326092  0.302942
# PNormNumbaGreat  35.278475  0.313421
# PNormNumbaSlow   35.627108  1.860825
