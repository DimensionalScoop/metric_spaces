import numpy as np

from ..tetrahedron import tetrahedron, proj_quality
from .common import METRIC


def __mask_diag(a):
    diag = np.diag(np.ones(a.shape[0]))
    return np.ma.array(a, mask=diag)


OPTIMAL_METHODS = [
    "optimal_candidate_set_size",
    "optimal_hyperplane_quality",
    "optimal_partition_usability",
]


def optimize_pivots(points, queries, r, return_full=False) -> dict:
    return {
        name: points[pivot_idx]
        for name, pivot_idx in zip(
            OPTIMAL_METHODS, list(_optimize_pivots(points, queries, r, return_full))
        )
    }


def _optimize_pivots(points, queries, r, return_full=False):
    all_quality = np.nan * np.ones([3, len(points), len(points)], float)

    for i, p0 in enumerate(points):
        for j, p1 in enumerate(points):
            if j <= i:
                continue
            try:
                points_p = tetrahedron.project_to_2d_euclidean(points, p0, p1, METRIC)
                queries_p = tetrahedron.project_to_2d_euclidean(queries, p0, p1, METRIC)

                all_quality[0, i, j] = proj_quality.candidate_set_size(
                    points_p, queries_p, r, METRIC
                )

                part = proj_quality.HilbertPartitioner(points_p)
                all_quality[1, i, j] = part.hyperplane_quality(points_p, r)
                all_quality[2, i, j] = part.is_query_in_one_partition(queries_p, r)

            except:
                pass

    for c in range(len(all_quality)):
        quality = all_quality[c]
        # make symmetric
        lower_tri = np.tril_indices(quality.shape[0], -1)
        quality[lower_tri] = quality.T[lower_tri]

        quality = __mask_diag(quality)

        if return_full:
            yield quality
        else:
            yield np.array(np.unravel_index(np.argmax(quality), quality.shape))


def optimize_pivot(points, p0, criterion, rng=None):
    quality = []
    r = proj_quality.get_average_k_nn_dist(points, METRIC, k=10)
    for p1 in points:
        try:
            points_p = tetrahedron.project_to_2d_euclidean(points, p0, p1, METRIC)
            quality.append(criterion(points_p, r))
        except KeyError:
            quality.append(0)
        except AssertionError:
            quality.append(0)
    return points[np.argmax(quality)]


#
# def hilbert_almost_optimal(ps, rng=None):
#     p0, _ = two_least_central(ps)
#     p1 = optimize_pivot(ps, p0, _hilbert_quality)
#     return p0, p1
#


# def ccs_optimal_pivot(ps, queries, r, rng=None):
#     quality = lambda x: -proj_quality.candidate_set_size(x, queries, r, METRIC)
#     pivots_idx = optimize_pivots(ps, [quality], False)[0]
#     return ps[pivots_idx]


# def hilbert_optimal_pivots(ps, queries, r, rng=None):
#     pivots_idx = optimize_pivots(
#         ps, lambda x: proj_quality.hilbert_quality(x, r), False
#     )
#     return ps[pivots_idx]


# def combined_optimality(points, queries, r):
#     qual_metrics = {
#         "optimal_candidate_set_size": lambda x, q: -proj_quality.candidate_set_size(
#             x, q, r, METRIC
#         ),
#         "optimal_hilbert_quality": lambda x, q: proj_quality.hilbert_quality(x, r),
#         "optimal_query_usable_partitioning": lambda x,
#         q,
#         q: proj_quality.query_usable_partitioning(x, q, r),
#     }
#     optimize_pivots
