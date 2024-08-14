"""Pivot selection techniques trying to maximize the lower bound approximation of distances."""

import numpy as np
from numba import njit
from itertools import permutations
from warnings import warn

from .common import METRIC


def _argamax(a):
    """Return index of largest scalar in array.
    Like np.argmax, but for amax (the maximum along all axis)"""
    return np.unravel_index(np.argmax(a.flatten()), a.shape)


def _all_pairs_distances(ps):
    all_dists = METRIC.distance_matrix(ps, ps)
    # look at all pairs from the POV of all pivots
    all_pairs_idx = np.array(list(permutations(range(len(ps)), 2)))
    dist_lhs = all_dists[:, all_pairs_idx[:, 0]]
    dist_rhs = all_dists[:, all_pairs_idx[:, 1]]
    return dist_lhs, dist_rhs, all_dists


def _select_IS_candidates(ps, n_pivs, n_points, rng):
    piv_candidates = rng.choice(ps, size=n_pivs, replace=False)

    # choose distance pairs without using any pair twice
    valid_permutations = np.array(np.triu_indices(len(ps), k=-1)).T
    pairs_idx = rng.choice(range(len(valid_permutations)), size=n_points, replace=False)
    lhs = ps[valid_permutations[pairs_idx, 0]]
    rhs = ps[valid_permutations[pairs_idx, 1]]
    return piv_candidates, lhs, rhs


def optimal_triangular_incremental_selection(ps, rng=None):
    """Chooses two pivots that maximize the sum of the best lower bounds."""
    # XXX: this uses way more memory than it needs to
    dist_lhs, dist_rhs, _ = _all_pairs_distances(ps)
    n_pivs, n_sampels = dist_lhs.shape
    assert n_sampels > len(ps) ** 1.4 and n_sampels < len(ps) ** 2, n_sampels

    chosen_pivots = _find_incremental_triangular_pair(dist_lhs, dist_rhs)
    return ps[chosen_pivots]


# TODO: change budget to total budget


def triangular_incremental_selection(ps, rng: np.random.Generator, budget=np.sqrt):
    """Chooses two pivots that maximize the sum of the best lower bounds.
    Subsamples from all points `ps` according to the budget.

    This implements the strategy of the same name from a review paper.

    Review paper: zhuPivotSelectionAlgorithms2022
    """
    # runtime: O(pivots * points), distribute evenly
    budget = len(ps) * budget(len(ps))
    n_pivs = int(np.sqrt(budget))
    n_points = int(np.sqrt(budget))
    piv_candidates, objects_lhs, objects_rhs = _select_IS_candidates(
        ps, n_pivs, n_points, rng
    )

    dist_lhs = METRIC.distance_matrix(piv_candidates, objects_lhs)
    dist_rhs = METRIC.distance_matrix(piv_candidates, objects_rhs)
    assert dist_lhs.shape == (len(piv_candidates), len(objects_lhs))

    chosen_pivots = _find_incremental_triangular_pair(dist_lhs, dist_rhs)
    return piv_candidates[chosen_pivots]


def _find_incremental_triangular_pair(dist_lhs, dist_rhs):
    n_pivs, n_samples = dist_lhs.shape

    lower_bounds = np.abs(dist_lhs - dist_rhs)
    lbs_quality = np.sum(lower_bounds, axis=1)
    assert len(lbs_quality) == n_pivs
    best_pivot_idx = np.argmax(lbs_quality)

    lb_of_best_pivot = lower_bounds[best_pivot_idx]
    lb_of_other_pivots = (
        lower_bounds  # we can ignore that `best_pivot_idx` is included here
    )

    # choose the best available LB: use either the best pivot or the other pivot, for each pivot
    best_lbs = np.fmax(lb_of_best_pivot.reshape(1, -1), lb_of_other_pivots)
    assert best_lbs.shape == (n_pivs, n_samples)
    lbs_quality = best_lbs.sum(axis=1)
    second_best_piv_idx = np.argmax(lbs_quality)
    assert second_best_piv_idx != best_pivot_idx

    return np.array([best_pivot_idx, second_best_piv_idx])


def ptolemy_optimal_selection(ps, rng=None):
    warn("this method takes too long: it uses O(n^4) operations (n=len(ps))")
    dist_matrix = METRIC.distance_matrix(ps, ps)

    dist_lhs, dist_rhs, dist_matrix = _all_pairs_distances(ps)
    lb_quality = _ptolemy_scores(dist_lhs, dist_rhs, dist_matrix)

    p1, p2 = _argamax(lb_quality)
    return ps[p1], ps[p2]


def ptolemys_incremental_selection(ps, rng: np.random.Generator, budget=np.sqrt):
    """Chooses pivots that maximize the sum of the best lower bounds."""

    # runtime: O(pivots^2 + pivots * points) distances
    #          O(pivots^2 * points) other ops
    # → for budget = np.sqrt, complexity should be O(n^1.5)
    # pivots = sqrt(n), points = n, other ops in O(n^2)

    budget = (len(ps) * budget(len(ps))) ** (1 / 1.5)
    n_pivs = int(np.sqrt(budget))
    n_points = int(budget)
    piv_candidates, objects_lhs, objects_rhs = _select_IS_candidates(
        ps, n_pivs, n_points, rng
    )

    # XXX: this uses way more memory than it needs to
    piv_lhs = METRIC.distance_matrix(piv_candidates, objects_lhs)
    piv_rhs = METRIC.distance_matrix(piv_candidates, objects_rhs)
    piv_piv = METRIC.distance_matrix(piv_candidates, piv_candidates)

    lb_quality = _ptolemy_scores(piv_lhs, piv_rhs, piv_piv)

    p1, p2 = _argamax(lb_quality)
    return piv_candidates[p1], piv_candidates[p2]


@njit
def _ptolemy_scores(piv_lhs, piv_rhs, piv_piv):
    """Calculate the sum of Ptolemy's lower bounds"""
    k, _ = piv_piv.shape
    lb_quality = np.zeros((k, k))

    for p1 in range(k):
        for p2 in range(p1 + 1, k):
            lb_quality[p1, p2] = (
                np.abs(
                    piv_lhs[p1, :] * piv_rhs[p2, :] - piv_lhs[p2, :] * piv_rhs[p1, :]
                )
            ).sum() / piv_piv[p1, p2]
    return lb_quality
