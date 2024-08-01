import numpy as np
from tqdm import tqdm

from generate import point_generator


import numexpr


def _preproc_points(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    # do we have two points, or lists of points?
    dim = max(len(a.shape), len(b.shape))
    return a, b, dim


class Metric:
    def __init__(self, name):
        self.name = name

    def _calc_distance(self, a: np.ndarray, b: np.ndarray, is_list: bool) -> np.array:
        pass

    def __call__(self, a, b):
        a, b, dim = _preproc_points(a, b)
        return self._calc_distance(a, b, dim == 2)

    def distance_matrix(self, x, y, threshold=1000000):
        """like (and copied from) scipy.spatial.distance_matrix"""
        x = np.asarray(x)
        m, k = x.shape
        y = np.asarray(y)
        n, kk = y.shape

        if k != kk:
            raise ValueError(
                f"x contains {k}-dimensional vectors but y contains "
                f"{kk}-dimensional vectors"
            )

        if m * n * k <= threshold:
            return self(x[:, np.newaxis, :], y[np.newaxis, :, :])
        else:
            result = np.empty((m, n), dtype=float)  # FIXME: figure out the best dtype
            if m < n:
                for i in range(m):
                    result[i, :] = self(x[i], y)
            else:
                for j in range(n):
                    result[:, j] = self(x, y[j])
            return result


_NUMEXPR_EUCLID = (
    "sum((a-b)**2, axis=0)",
    "sum((a-b)**2, axis=1)",
    "sum((a-b)**2, axis=2)",
)

from metric.metric import PNorm, Euclid

from numba import njit


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


from joblib import delayed, Parallel


def run(seed, metric, n_samples=20):
    rng = np.random.default_rng(seed)
    samples = range(n_samples)
    if seed % 10 == 0:
        samples = tqdm(samples)
    for _ in samples:
        points = point_generator.generate_gaussian_points(rng, 3000, 7, False)
        res = metric.distance_matrix(points, points)
    return res


import time

# warm-up
actual = run(0, metric=EuclidNumba(), n_samples=1)
actual = run(0, metric=PNormNumba(), n_samples=1)
actual = run(0, metric=PNormNumbaSlow(), n_samples=1)
desired = run(0, metric=Euclid(), n_samples=1)
assert np.allclose(desired, actual)

for m in [
    EuclidNumba(),
    PNormNumba(),
    EuclidNumba(),
    PNormNumbaSlow(),
    PNormNumba(),
]:  # Euclid(), PNorm()]):
    start_time = time.perf_counter()
    Parallel(10)(delayed(run)(i, m) for i in range(10))
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"{m} Execution time: {execution_time:.2f} seconds")
