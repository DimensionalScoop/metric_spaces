from run import run

import numpy as np
from numpy.testing import assert_array_almost_equal
from meters.metric.fast_distance_matrix import euclidean_distance_matrix
from meters.metric import Euclid
import scipy
from tqdm import tqdm
import line_profiler

# metric = Euclid(2)

# @line_profiler.profile
# def doit():
#     for _ in range(100):
#         a = np.random.rand(512,12)
#         b = np.random.rand(1024,12)

#         r1 = euclidean_distance_matrix(a,b)
#         r2 = metric.distance_matrix(a, b)
#         r3 = scipy.spatial.distance_matrix(a,b)

#         assert_array_almost_equal(r1, r3)
#         assert_array_almost_equal(r1, r2)

# doit()


run(
    n_runs=1,
    n_samples=512,
    n_queries=512,
    dims=[5],
    n_cpus=1,
    datasets=["gaussian, eliptic"],
    algorithms=["optimal"],
    verbose=True,
)
