import numpy as np
import sys
from tqdm import tqdm

import tetrahedron
import proj_quality

import pivot_selection
import point_generator


from scipy import spatial
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


class Euclid(Metric):
    """Numexpr implementation of the Euclidean distance, with a 2x speedup"""

    def __init__(self, p=2):
        if p != 2:
            raise ArgumentError()
        name = r"\|\cdot\|"
        super().__init__(name)

    def _calc_distance(self, a: np.ndarray, b: np.ndarray, is_list: bool) -> np.array:
        n_axis = max((len(a.shape), len(b.shape)))
        return np.sqrt(numexpr.evaluate(_NUMEXPR_EUCLID[n_axis - 1]))

    def distance_matrix(self, a, b, threshold=1000000, rank_only=False):
        if a is not b:
            return super().distance_matrix(a, b)

        b = a[np.newaxis, :, :]
        a = a[:, np.newaxis, :]

        if rank_only:
            return numexpr.evaluate(_NUMEXPR_EUCLID[2])
        else:
            return np.sqrt(numexpr.evaluate(_NUMEXPR_EUCLID[2]))


from metric.metric import PNorm, Euclid

metric = PNorm(2)
metric = Euclid(2)

from joblib import delayed, Parallel


def run(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        points = point_generator.generate_gaussian_points(rng, 3000, 7, False)
        res = metric.distance_matrix(points, points)
    return res.sum()


import time

start_time = time.perf_counter()
Parallel(10)(delayed(run)(i) for i in range(20))
end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
