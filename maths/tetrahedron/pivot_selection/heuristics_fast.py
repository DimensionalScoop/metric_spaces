import numpy as np
import sys

sys.path.append("../../")

from metric.metric import Euclid
from .common import choose_reasonably_remote_partner


METRIC = Euclid(2)


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
