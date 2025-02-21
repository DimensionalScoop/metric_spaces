import numpy as np
from numba import njit

from .common import choose_reasonably_remote_partner, METRIC
from . import heuristics_complete as heu_com


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


def two_least_central_heuristically(ps, rng: np.random.Generator, budget=np.sqrt):
    """
    Find the two pivots that approximately maximize `Var[d(piv, .)]`.

    ps: points in the space
    budget: limit to O(n*budget(n)) distance method calls.
            Default: O(n^1.5)
    """
    dist_budget = len(ps) * int(budget(len(ps)))

    # instead of implementing this, simulate this by throwing away entries
    # from a full distance matrix
    dists = METRIC.distance_matrix(ps, ps)

    diag_elements = len(dists)
    # dists is symmetric, so each value has a partner
    dist_evaluations = (dists.size - diag_elements) // 2
    count_discards = dist_evaluations - dist_budget
    discard = rng.choice(range(dist_evaluations), replace=False, size=count_discards)

    # lower triangle matrix with the diagonal excluded
    keep = np.tri(len(dists), k=-1, dtype=bool)
    discard_pairs = np.argwhere(keep == 1)[discard]
    discard_idx = discard_pairs.T
    keep[*discard_idx] = False
    dists[~keep] = -1

    actual_evaluations = dists.size - (dists == -1).sum()
    assert np.allclose(actual_evaluations, dist_budget, atol=2), (
        f"`dists` has {actual_evaluations:e} entries, but it should have no more than {dist_budget:e}!"
    )

    # make matrix symmetric again
    dists = np.maximum(dists, dists.T)
    assert np.allclose(dists, dists.T)

    dists = np.ma.array(dists, mask=dists == -1)
    centricity = np.argsort(dists.std(axis=1))

    # the two pivots with the highest std might be close together
    p0 = ps[centricity[-1]]
    p1 = choose_reasonably_remote_partner(ps[centricity], p0)

    return p0, p1
