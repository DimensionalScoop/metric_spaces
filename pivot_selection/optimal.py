import numpy as np

from tetrahedron import tetrahedron, proj_quality
from metric.metric import Euclid

METRIC = Euclid(2)


def __mask_diag(a):
    diag = np.diag(np.ones(a.shape[0]))
    return np.ma.array(a, mask=diag)


def optimize_pivots(points, criterion, return_full=False):
    quality = np.nan * np.ones([len(points), len(points)], float)

    for i, p0 in enumerate(points):
        for j, p1 in enumerate(points):
            if j <= i:
                continue
            try:
                points_p = tetrahedron.project_to_2d_euclidean(points, p0, p1, METRIC)
                quality[i, j] = criterion(points_p)
            except:
                quality[i, j] = np.nan

    # make symmetric
    lower_tri = np.tril_indices(quality.shape[0], -1)
    quality[lower_tri] = quality.T[lower_tri]

    quality = __mask_diag(quality)

    if return_full:
        return quality
    else:
        return np.array(np.unravel_index(np.argmax(quality), quality.shape))


def hilbert_optimal_pivots(ps, rng=None):
    r = proj_quality.get_average_k_nn_dist(ps, METRIC, k=10)
    pivots_idx = optimize_pivots(
        ps, lambda x: proj_quality.hilbert_quality(x, r), False
    )
    return ps[pivots_idx]


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


def ccs_optimal_pivot(ps, rng):
    r = proj_quality.get_average_k_nn_dist(ps, METRIC, k=10)

    quality = lambda x: -proj_quality.candidate_set_size(x, r, METRIC)
    pivots_idx = optimize_pivots(ps, quality, False)
    return ps[pivots_idx]
