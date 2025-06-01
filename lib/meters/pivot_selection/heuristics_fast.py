import numpy as np
from numba import njit

from .common import choose_reasonably_remote_partner, METRIC
from . import heuristics_complete as heu_com

# calculating a distance matrix between all k points costs:
# ((k-1)^2 + k - 1) / 2


# setting the budget at O(n^1.5) therefore means:
# n^1.5 distance calculations, or
# samples points for which a full distance matrix will be calculated:
def dist_calc_budget_to_points(pt):
    n = len(pt)
    return int(0.5 * (np.sqrt(8 * n ** (3 / 2) + 1) + 1))


def dist_matrix_at_a_budget(ps, budget, rng):
    """returns a distance matrix that has missing elements, so that at most `budget`
    distances are in the matrix.
    """
    # instead of implementing this, simulate this by throwing away entries
    # from a full distance matrix
    dists = METRIC.distance_matrix(ps, ps)

    diag_elements = len(dists)
    # dists is symmetric, so each value has a partner
    dist_evaluations = (dists.size - diag_elements) // 2
    count_discards = dist_evaluations - budget
    discard = rng.choice(range(dist_evaluations), replace=False, size=count_discards)

    # lower triangle matrix with the diagonal excluded
    keep = np.tri(len(dists), k=-1, dtype=bool)
    discard_pairs = np.argwhere(keep == 1)[discard]
    discard_idx = discard_pairs.T
    keep[*discard_idx] = False
    dists[~keep] = -1

    actual_evaluations = dists.size - (dists == -1).sum()
    assert np.allclose(actual_evaluations, budget, atol=2), (
        f"`dists` has {actual_evaluations:e} entries, but it should have no more than {budget:e}!"
    )

    # make matrix symmetric again
    dists = np.maximum(dists, dists.T)
    assert np.allclose(dists, dists.T)
    return dists


def max_dist_GNAT(ps, rng: np.random.Generator, budget=np.sqrt):
    """Chooses the most distant pair of points from a subsample of the database.

    This implement the strategy used for GNAT in its original paper.
    The basic idea is that this biases the selection towards cluster centers,
    as long as those centers have a high point density.

    Original paper: brinNeighborSearchLarge1995
    """
    # empirical factor from the paper
    OVERSAMPLING_FACTOR = 3

    # Here, we have to depart a bit from the original paper:
    # Because they build an entire index and not just one projection, they
    # need way more pivot candidates.
    # We simulate this by considering a number of candidates "proprtional" to
    # the number of points in the dataset.
    n_candidates = int(OVERSAMPLING_FACTOR * budget(len(ps)))
    pivot_candidates = rng.choice(ps, size=n_candidates, replace=False)
    return heu_com.max_dist_points(pivot_candidates)


def fair_max_dist(ps, rng: np.random.Generator):
    """Chooses the most distant pair of points from a subsample of the database."""
    n_dist_calcs = int(len(ps) ** (3 / 2))
    dists = dist_matrix_at_a_budget(ps, n_dist_calcs, rng)

    flat_index = dists.argmax()
    row_index, col_index = np.unravel_index(flat_index, dists.shape)
    return ps[row_index], ps[col_index]


def remote_points(ps, rng: np.random.Generator):
    n_candidates = dist_calc_budget_to_points(ps)
    pivot_candidates = rng.choice(ps, size=n_candidates, replace=False)
    return heu_com.two_remote_points(pivot_candidates)


def two_least_central_heuristically(ps, rng: np.random.Generator):
    """
    Find the two pivots that approximately maximize `Var[d(piv, .)]`.

    ps: points in the space
    """
    dist_budget = int(len(ps) ** (3 / 2))

    dists = dist_matrix_at_a_budget(ps, dist_budget, rng)
    dists = np.ma.array(dists, mask=dists == -1)
    centricity = np.argsort(dists.std(axis=1))

    # the two pivots with the highest std might be close together
    p0 = ps[centricity[-1]]
    p1 = choose_reasonably_remote_partner(ps[centricity], p0)

    return p0, p1
