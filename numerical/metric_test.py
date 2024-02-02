"""Numerically test whether distances are a metric"""

import numpy as np
from tqdm import tqdm

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
        return -(d(x, z) + d(z, y) - d(x, y))

    def pto_violation(self, x, y, z, k):
        d = self.d
        return -(d(x, y) * d(z, k) + d(y, z) * d(k, x) - d(x, z) * d(y, k))

    def is_pseudo(self, x, y):
        diff = np.sum(np.abs(x - y))
        return diff > self.atol and self.d(x, y) < self.atol

    def run_test(self, n_samples=10000, multicore=True):
        # XXX: doesn't work yet
        n_jobs = 20 if multicore else 1
        executor = Parallel(n_jobs=n_jobs)
        job = delayed(self._run_test)(n_samples // n_jobs)
        return executor(job)

    def _run_test(self, n_samples=1000, use_tqdm=True):
        messages = []
        samples = tqdm(range(n_samples)) if use_tqdm else range(n_samples)
        for _ in samples:
            x, y, z, k = self.vector_generators_func(4)
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
