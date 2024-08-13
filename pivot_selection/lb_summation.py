"""Pivot selection techniques trying to maximize the lower bound approximation of distances."""

import numpy as np
from numba import njit

from .common import METRIC


def _select_IS_candidates(ps, budget, rng):
    n_candidates = int(budget(len(ps)))
    piv_candidates = rng.choice(ps, size=n_candidates, replace=False)

    # choose distance pairs without using any pair twice
    valid_permutations = np.array(np.triu_indices(len(ps), k=-1)).T
    pairs_idx = rng.choice(
        range(len(valid_permutations)), size=n_candidates, replace=False
    )
    lhs = ps[valid_permutations[pairs_idx, 0]]
    rhs = ps[valid_permutations[pairs_idx, 1]]
    return piv_candidates, lhs, rhs


def triangular_incremental_selection(ps, rng: np.random.Generator, budget=np.sqrt):
    """Chooses pivots that maximize the sum of the best lower bounds.

    This implements the strategy of the same name from a review paper.

    Review paper: zhuPivotSelectionAlgorithms2022
    """
    piv_candidates, objects_lhs, objects_rhs = _select_IS_candidates(ps, budget, rng)

    lb_lhs = METRIC.distance_matrix(piv_candidates, objects_lhs)
    lb_rhs = METRIC.distance_matrix(piv_candidates, objects_rhs)
    lower_bounds = np.abs(lb_lhs - lb_rhs)

    # TODO: reafactor after here to extract main method, use in fast and complete version

    lbs_quality = np.sum(lower_bounds, axis=0)
    best_pivot_idx = np.argmax(lbs_quality)

    lb_of_best_pivot = lower_bounds[best_pivot_idx]
    lb_of_other_pivots = (
        lower_bounds  # we can ignore that `best_pivot_idx` is included here
    )

    # choose the best available LB: use either the best pivot or the other pivot, for each pivot
    best_lbs = np.fmax(lb_of_best_pivot.reshape(1, -1), lb_of_other_pivots)
    lbs_quality = best_lbs.sum(axis=1)
    second_best_piv_idx = np.argmax(lbs_quality)

    return piv_candidates[best_pivot_idx], piv_candidates[second_best_piv_idx]


def _argamax(a):
    """Return index of largest scalar in array.
    Like np.argmax, but for amax (the maximum along all axis)"""
    return np.unravel_index(np.argmax(a.flatten()), a.shape)


def ptolemys_incremental_selection(ps, rng: np.random.Generator, budget=np.sqrt):
    """Chooses pivots that maximize the sum of the best lower bounds."""
    piv_candidates, objects_lhs, objects_rhs = _select_IS_candidates(ps, budget, rng)

    piv_lhs = METRIC.distance_matrix(piv_candidates, objects_lhs)
    piv_rhs = METRIC.distance_matrix(piv_candidates, objects_rhs)
    piv_piv = METRIC.distance_matrix(piv_candidates, piv_candidates)

    lb_quality = _ptolemy_scores(piv_candidates, piv_lhs, piv_rhs, piv_piv)

    p1, p2 = _argamax(lb_quality)
    return piv_candidates[p1], piv_candidates[p2]


@njit
def _ptolemy_scores(piv_candidates, piv_lhs, piv_rhs, piv_piv):
    """Calculate the sum of Ptolemy's lower bounds"""
    k = len(piv_candidates)
    lb_quality = np.zeros((k, k))
    for p1 in range(k):
        for p2 in range(p1 + 1, k):
            lb_quality[p1, p2] = (
                np.abs(
                    piv_lhs[p1, :] * piv_rhs[p2, :] - piv_lhs[p2, :] * piv_rhs[p1, :]
                )
                / piv_piv[p1, p2]
            ).sum()
    return lb_quality


# TODO: find best scores simultaneously instead of iteratively
