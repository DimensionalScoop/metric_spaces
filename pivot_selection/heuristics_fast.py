import numpy as np
import sys
from numba import njit

from metric.metric import Euclid
from .common import choose_reasonably_remote_partner
from . import heuristics_complete as heu_com


METRIC = Euclid(2)


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
    pivot_candidates = rng.choice(rng, size=n_candidates, replace=False)
    return heu_com.max_dist_points(pivot_candidates)


def _select_IS_candidates(ps, budget):
    # TODO: select initial candidates, object relations
    piv_candidates = []

    objects_lhs = []
    objects_rhs = []
    return piv_candidates, objects_lhs, objects_rhs


def triangular_incremental_selection(ps, rng: np.random.Generator, budget=np.sqrt):
    """Chooses pivots that maximize the sum of the best lower bounds.

    This implements the strategy of the same name from a review paper.

    Review paper: zhuPivotSelectionAlgorithms2022
    """
    piv_candidates, objects_lhs, objects_rhs = _select_IS_candidates(ps, budget)

    lb_lhs = METRIC.distance_matrix(piv_candidates, objects_lhs)
    lb_rhs = METRIC.distance_matrix(piv_candidates, objects_rhs)
    lower_bounds = np.abs(lb_lhs - lb_rhs)

    lbs_quality = np.sum(lower_bounds, axis=0)
    best_pivot_idx = np.argmax(lbs_quality)

    lb_of_best_pivot = lower_bounds[best_pivot_idx]
    lb_of_other_pivots = (
        lower_bounds  # we can ignore that `best_pivot_idx` is included here
    )

    # choose the best available LB: use either the best pivot or the other pivot, for each pivot
    best_lbs = np.fmax(lb_of_best_pivot.reshape(1, -1), lb_of_other_pivots, axis=1)
    lbs_quality = best_lbs.sum(axis=1)
    second_best_piv_idx = np.argmax(lbs_quality)

    return piv_candidates[best_pivot_idx], piv_candidates[second_best_piv_idx]


def _argamax(a):
    """Return index of largest scalar in array.
    Like np.argmax, but for amax (the maximum along all axis)"""
    return np.unravel_index(np.argmax(a.flatten()), a.shape)


def ptolemys_incremental_selection(ps, rng: np.random.Generator, budget=np.sqrt):
    """Chooses pivots that maximize the sum of the best lower bounds."""
    piv_candidates, objects_lhs, objects_rhs = _select_IS_candidates(ps, budget)

    piv_lhs = METRIC.distance_matrix(piv_candidates, objects_lhs)
    piv_rhs = METRIC.distance_matrix(piv_candidates, objects_rhs)
    piv_piv = METRIC.distance_matrix(piv_candidates, piv_candidates)

    lb_quality = _ptolemy_scores(piv_candidates, piv_lhs, piv_rhs, piv_piv)

    p1, p2 = np.argamax(lb_quality)
    return piv_candidates[p1], piv_candidates[p2]


@njit
def _ptolemy_scores(piv_candidates, piv_lhs, piv_rhs, piv_piv):
    """Calculate the sum of Ptolemy's lower bounds"""
    k = len(piv_candidates)
    lb_quality = np.zeros((k, k))
    for p1 in range(k):
        for p2 in range(p1, k):
            lb_quality[p1, p2] = (
                np.abs(
                    piv_lhs[p1, :] * piv_rhs[p2, :] - piv_lhs[p2, :] * piv_rhs[p1, :]
                )
                / piv_piv[p1, p2]
            ).sum()
    return lb_quality


# TODO: find best scores simultaneously instead of iteratively


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
    assert np.allclose(
        actual_evaluations, dist_budget, atol=2
    ), f"`dists` has {actual_evaluations:e} entries, but it should have no more than {dist_budget:e}!"

    # make matrix symmetric again
    dists = np.maximum(dists, dists.T)
    assert np.allclose(dists, dists.T)

    dists = np.ma.array(dists, mask=dists == -1)
    centricity = np.argsort(dists.std(axis=1))

    # the two pivots with the highest std might be close together
    p0 = ps[centricity[-1]]
    p1 = choose_reasonably_remote_partner(ps[centricity], p0)

    return p0, p1
