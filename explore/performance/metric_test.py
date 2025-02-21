"""Numerically test whether distances are a metric"""

import numpy as np
from tqdm import tqdm
from itertools import chain
import os

import joblib
from joblib import Parallel, delayed


class NoVarianceError(ValueError):
    pass


def histogram_overlap(data_a, data_b, bins=50):
    both = np.hstack((data_a, data_b))
    bins = np.histogram_bin_edges(both, bins=bins)
    a, _ = np.histogram(data_a, bins)
    b, _ = np.histogram(data_b, bins)
    overlap_absolute = np.vstack((a, b)).min(axis=0).sum()
    # bin_width = bins[1] - bins[0]
    return overlap_absolute / len(data_a)


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
        step_size = 1000
        n_jobs = joblib.cpu_count() if multicore else 1
        executor = Parallel(n_jobs=n_jobs, verbose=3)

        jobs = [
            delayed(self._run_test_unit)(step_size)
            for _ in range(0, n_samples, step_size)
        ]
        result = executor(list(jobs))
        return list(chain.from_iterable(result))

    def _run_test_unit(self, n_samples):
        seed = int.from_bytes(os.urandom(4))
        rng = np.random.default_rng(seed)

        points = self.vector_generators_func(n_samples, 4, rng=rng)
        result = [self._test_single(*p) for p in points]
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
