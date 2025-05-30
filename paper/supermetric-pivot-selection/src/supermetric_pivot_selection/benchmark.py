from supermetric_pivot_selection.run import run

import numpy as np
from numpy.testing import assert_array_almost_equal
import meters.metric.fast_distance_matrix as fast
from meters.metric import Euclid
import scipy
from tqdm import tqdm
import line_profiler

# metric = Euclid(2)

# @line_profiler.profile
# def doit():
#     for _ in tqdm(range(5)):
#         points = np.random.rand(512*10, 512,12)
#         queries = np.random.rand(512*10, 128,12)
#         r = 0.33

#         batched = fast.euclidean_range_query_hits_batched(points, queries, r)
#         unbatched = unb(points,queries,r)
#         unbatched2 = unb2(points,queries,r)

#         assert_array_almost_equal(batched, unbatched)
#         print(np.sum(batched))

#         # assert_array_almost_equal(r3, r1)
#         # assert_array_almost_equal(r3, r2)
#         # assert_array_almost_equal(r3, r4)

# def  unb(points,queries,r):
#     unbatched = np.zeros([len(points)], dtype=int)
#     for i in range(len(points)):
#         unbatched[i] = fast.euclidean_range_query_hits(points[i], queries[i], r)
#     return unbatched


# def unb2(points,queries,r):
#     unbatched = np.zeros([len(points)], dtype=int)
#     for i in range(len(points)):
#         unbatched[i] = fast.euclidean_range_query_hits_v4(points[i], queries[i], r)
#     return unbatched

# doit()


run(
    n_runs=1,
    # n_samples=512,
    # n_queries=128,
    n_samples=128,
    n_queries=128,
    dims=list(range(2, 18)),
    n_cpus=-1,
    # datasets=["gaussian, eliptic"],
    algorithms=["optimal"],
    verbose=True,
)
