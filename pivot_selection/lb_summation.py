"""Pivot selection techniques trying to maximize the lower bound approximation of distances."""

import numpy as np
from numba import njit
from itertools import permutations
from warnings import warn

from typing import Literal

from .common import METRIC


def IS(
    ps,
    n_pivs,
    n_pairs,
    rng: np.random.Generator,
    lb_type: Literal["tri", "pto"] = "tri",
    fixed_first_pivot=False,
):
    """Maximize the sum of distances of the lower-bound approximation, using two pivots.
    Returns the best pivot pair.

    n_pivs: number of pivot candidates
    n_objects: number of objects to approximate the distance between
    lb_type: Triangle or Ptolemaic lower bound?
    fixed_fist_pivot: fix one pivot for speedup?

    if lb_type == "tri": dist_calcs = O(n_pivs * n_pairs)
    if lb_type == "pto": dist-calcs = O(n_pivs * n_pairs + (fixed_fist_pivot) * n_pivs**2)

    match (lb_type, fixed_first_pivot):
    if fixed_first_pivot:
        complexity: O(n_pivs * n_pairs)
    else:
        complexity: O(n_pivs**2 * n_pairs)

    Review paper: zhuPivotSelectionAlgorithms2022
    """
    piv_cand, lhs, rhs = _choose_pivs_and_pairs(rng, ps, n_pivs, n_pairs)

    cached_distances = dict(
        d_piv_lhs=METRIC.distance_matrix(piv_cand, lhs),
        d_piv_rhs=METRIC.distance_matrix(piv_cand, rhs),
    )
    if lb_type == "pto":
        cached_distances["d_piv_piv"] = METRIC.distance_matrix(piv_cand, piv_cand)

    def choose_algo():
        match (lb_type, fixed_first_pivot):
            case "tri", False:
                return _IS_tri
            case "pto", False:
                return _IS_pto
            case "tri", True:
                return _IS_tri_fixed_first_pivot
            case "pto", True:
                return _IS_pto_fixed_first_pivot
            case _:
                raise ValueError(f"Unknown combination {lb_type}, {fixed_first_pivot}")

    get_best_pair = choose_algo()
    p0, p1 = get_best_pair(**cached_distances)
    assert p0 != p1

    return piv_cand[p0], piv_cand[p1]


def _choose_pivs_and_pairs(rng, ps, n_pivs, n_pairs):
    ps = rng.choice(ps, len(ps), replace=False)
    piv_candidates = ps[:n_pivs]
    objs = ps[n_pivs:]

    # choose pairs
    valid_permutations = np.array(np.triu_indices(len(objs), k=-1)).T
    pairs_idx = rng.choice(range(len(valid_permutations)), size=n_pairs, replace=False)
    lhs = ps[valid_permutations[pairs_idx, 0]]
    rhs = ps[valid_permutations[pairs_idx, 1]]
    return piv_candidates, lhs, rhs


@njit
def _argamax(a):
    """Return index of largest scalar in array.
    Like np.argmax, but for amax (the maximum along all axis)"""
    return np.unravel_index(np.argmax(a.flatten()), a.shape)


@njit
def _IS_tri(d_piv_lhs, d_piv_rhs):
    """Calculate the quality of the triangle lower bound approximation.
    Returns the indices of the pivot pair with the highest quality.

    d_piv_lhs: distance matrix between pivots and left object
    d_piv_rhs: distance matrix between pivots and right object
    """
    n_pivs, n_pairs = d_piv_lhs.shape
    lb_quality = np.zeros([n_pivs] * 2)

    for piv0 in range(n_pivs):
        for piv1 in range(piv0 + 1, n_pivs):
            for pair in range(n_pairs):
                a = np.abs(d_piv_lhs[piv0, pair] - d_piv_rhs[piv0, pair])
                b = np.abs(d_piv_lhs[piv1, pair] - d_piv_rhs[piv1, pair])
                lb_quality[piv0, piv1] += max(a, b)

    p0, p1 = _argamax(lb_quality)
    return p0, p1


@njit
def _IS_tri_fixed_first_pivot(d_piv_lhs, d_piv_rhs):
    n_pivs, n_pairs = d_piv_lhs.shape
    lb_quality = np.zeros([1, n_pivs])

    p0 = _best_starting_pivot(d_piv_lhs, d_piv_rhs)
    for p1 in range(n_pivs):
        for pair in range(n_pairs):
            a = np.abs(d_piv_lhs[p0, pair] - d_piv_rhs[p0, pair])
            b = np.abs(d_piv_lhs[p1, pair] - d_piv_rhs[p1, pair])
            lb_quality[p0, p1] += max(a, b)

    _, p1 = _argamax(lb_quality)
    return p0, p1


@njit
def _best_starting_pivot(d_piv_lhs, d_piv_rhs):
    """Find the single pivot maximizing the triangle lower bound."""
    n_pivs, n_pairs = d_piv_lhs.shape
    lb_quality = np.zeros(n_pivs)

    for piv0 in range(0, n_pivs):
        for pair in range(n_pairs):
            lb_quality[piv0] = np.abs(d_piv_lhs[piv0, :] - d_piv_rhs[piv0, :]).sum()

    p0 = np.argmax(lb_quality)
    return p0


@njit
def _IS_pto(d_piv_lhs, d_piv_rhs, d_piv_piv):
    """Calculate the sum of Ptolemy's lower bounds"""
    n_pivs, n_pairs = d_piv_lhs.shape
    lb_quality = np.zeros((n_pivs, n_pivs))

    for p0 in range(n_pivs):
        for p1 in range(p0 + 1, n_pivs):
            lb_quality[p0, p1] = (
                np.abs(
                    d_piv_lhs[p0, :] * d_piv_rhs[p1, :]
                    - d_piv_lhs[p1, :] * d_piv_rhs[p0, :]
                )
            ).sum() / d_piv_piv[p0, p1]

    p0, p1 = _argamax(lb_quality)
    return p0, p1


@njit
def _IS_pto_fixed_first_pivot(d_piv_lhs, d_piv_rhs, d_piv_piv):
    """Calculate the sum of Ptolemy's lower bounds"""
    n_pivs, n_pairs = d_piv_lhs.shape
    lb_quality = np.zeros((1, n_pivs))

    p0 = _best_starting_pivot(d_piv_lhs, d_piv_rhs)
    for p1 in range(n_pivs):
        lb_quality[p0, p1] = (
            np.abs(
                d_piv_lhs[p0, :] * d_piv_rhs[p1, :]
                - d_piv_lhs[p1, :] * d_piv_rhs[p0, :]
            )
        ).sum() / d_piv_piv[p0, p1]

    _, p1 = _argamax(lb_quality)
    return p0, p1
