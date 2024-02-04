"""Numerically test whether distances are a metric"""

import numpy as np
from tqdm import tqdm
from itertools import chain

from joblib import Parallel, delayed


class NoVarianceError(ValueError):
    pass


class MetricTest:
    def __init__(self, distance_func, vector_generators_func, atol=1e-4) -> None:
        self.d = distance_func
        self.vector_generators_func = vector_generators_func
        self.atol = atol

    def triangle_violation(self, x, y, z):
        d = self.d
        xz = d(x, z)
        xy = d(x, y)
        yz = d(y, z)

        # calculate all three possible triangles
        max_violation = min(
            [
                xz + xy - yz,
                xz - xy + yz,
                -xz + xy + yz,
            ]
        )

        return -max_violation

    def pto_violation(self, x, y, z, k):
        d = self.d
        return -(d(x, y) * d(z, k) + d(y, z) * d(k, x) - d(x, z) * d(y, k))

    def is_pseudo(self, x, y):
        diff = np.sum(np.abs(x - y))
        return diff > self.atol and self.d(x, y) < self.atol

    def run_test(self, n_samples=10000, multicore=True):
        n_jobs = 20 if multicore else 1
        executor = Parallel(n_jobs=n_jobs, verbose=3)

        points = self.vector_generators_func(n_samples, 4)
        jobs = [delayed(self._test_single)(*p) for p in points]
        result = executor(jobs)
        return list(chain.from_iterable(result)) 

    def _test_single(self, x, y, z, k):
        messages = []
        try:
            if self.triangle_violation(x, y, z) > self.atol:
                messages.append(("triangle violation", [x, y, z]))
            if self.pto_violation(x, y, z, k) > 1e-3:
                messages.append(("pto violation", [x, y, z, k]))
            if self.is_pseudo(x, y):
                messages.append(("distance 0", (x, y)))
        except ValueError:
            messages.append(("skipped", (x, y, z, k)))

        return messages