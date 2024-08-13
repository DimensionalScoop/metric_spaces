import numpy as np

from .common import choose_reasonably_remote_partner, METRIC


def __mask_diag(a):
    diag = np.diag(np.ones(a.shape[0]))
    return np.ma.array(a, mask=diag)


def max_dist_points(ps, rng=None):
    dists = METRIC.distance_matrix(ps, ps)
    flat_index = dists.argmax()
    row_index, col_index = np.unravel_index(flat_index, dists.shape)
    return ps[row_index], ps[col_index]


def min_dist_points(ps, rng=None):
    dists = METRIC.distance_matrix(ps, ps)
    diag = np.diag_indices(len(ps))
    dists[diag] = np.inf

    flat_index = dists.argmin()
    row_index, col_index = np.unravel_index(flat_index, dists.shape)
    return ps[row_index], ps[col_index]


def get_most_central_points_idx(ps, rng=None):
    dists = METRIC.distance_matrix(ps, ps)
    dists = __mask_diag(dists)

    return np.argsort(dists.std(axis=1))


def two_most_central(ps, rng=None):
    cent = get_most_central_points_idx(ps)
    return ps[cent[:2]]


def two_least_central(ps, rng=None):
    cent = get_most_central_points_idx(ps)
    return ps[cent[-2:]]


def low_centrality_and_far_away(ps, rng=None):
    cent = get_most_central_points_idx(ps)
    p0 = ps[cent[-1]]
    p1 = choose_reasonably_remote_partner(ps[cent], p0)
    return p0, p1


def two_remote_points(ps, rng=None):
    dists = METRIC.distance_matrix(ps, ps)
    dists = __mask_diag(dists)
    remoteness = (dists**2).sum(axis=1)
    remoteness_idx = np.argsort(remoteness)
    return ps[remoteness_idx[-2:]]


def central_and_distant(ps, rng=None):
    cent = get_most_central_points_idx(ps)
    p0 = ps[cent[0]]
    most_dist = np.argmax(METRIC(p0, ps))
    p1 = ps[most_dist]
    return p0, p1


def find_close_center(ps, cluster_member, rng=None):
    dists = METRIC(cluster_member, ps)
    close_points = ps[dists < np.median(dists)]

    dists = METRIC.distance_matrix(close_points, close_points)
    dists = __mask_diag(dists)

    center_idx = np.argmin(dists.std(axis=1))
    return close_points[center_idx]


def different_cluster_centers(ps, rng=None):
    # a,b are likely in different clusters
    a, b = max_dist_points(ps)
    center_a = find_close_center(ps, a)
    center_b = find_close_center(ps, b)

    while np.allclose(center_a, center_b):
        # we need two distinct pivots!
        center_b = rng.choice(ps, size=1)
    return center_a, center_b


def random_pivots(ps, rng=None):
    return rng.choice(ps, size=2, replace=False)


# TODO: Implement complete version of incremental selection heuristic
