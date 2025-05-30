"""Different measures to gauge the quality of a projection.

E.g.: quality of a hyperplane projection
"""

from typing import *
import numpy as np
from sklearn import decomposition
from sklearn.neighbors import KDTree
from sklearn import config_context
from sklearn.utils._array_api import get_namespace
import scipy
from sympy.series import order

import line_profiler

from ..metric.metric import Metric, Euclid


def _faster_transform(pca, X):
    with config_context(assume_finite=True, skip_parameter_validation=True):
        xp, _ = get_namespace(X, pca.components_, pca.explained_variance_)
        return pca._transform(X, xp=xp, x_is_centered=False)


@line_profiler.profile
def candidate_set_size(
    points: np.ndarray,
    queries: np.ndarray,
    r: float,
    d: Metric,
    agg="mean",
    use_kdtree=True,
) -> Any:
    """Run a range query with `r` on all `query` points. How big is the candidate set?

    Args:
        points: [number of points, dimension of points]
        queries: [number of queryies, dimension of points]
        r: query distance
        agg: aggregation metric:
            None: return list of candidate set sizes,
            "mean": np.mean
            callable: function that takes an np.ndarray and returns a float
    """
    if use_kdtree:
        assert isinstance(d, Euclid), f"metric {d} not supported"
        tree = KDTree(points)
        counts = tree.query_radius(queries, r, count_only=True)
        return counts.sum() / len(queries)
    else:
        if agg is None:
            dist_matrix = d.distance_matrix(queries, points)
            n_neighbours = (dist_matrix < r).sum(axis=-1)
            return n_neighbours
        else:
            return d.count_query_hits(points, queries, r) / len(queries)


def get_average_k_nn_dist(points, queries, d: Metric, k=10, agg="mean"):
    dist_m = d.distance_matrix(queries, points)
    dist_m = np.sort(dist_m, axis=1)
    k_dist = dist_m[:, k - 1]

    if agg is None:
        return k_dist
    elif agg == "mean":
        agg = np.mean
    return agg(k_dist)


def hilbert_quality(points, r):
    part = HilbertPartitioner(points)
    return part.hyperplane_quality(points, r)


def query_usable_partitioning(points, queries, r):
    part = HilbertPartitioner(points)
    return part.is_query_in_one_partition(queries, r)


class HilbertPartitioner:
    """Finds a hyperplane bisecting the dataset which tries to maximize
    the number of partitioned points.

    Rotates the space to find the best (i.e. broadest) point spread.
    Points closer to the hyperplaen than `r` are considered to not profit from the partition.

    Returns:
        Percentage of points that are not too close to the hyperplane.
    """

    def __init__(self, points: np.ndarray, dummy_transform=False):
        # XXX: The PCA only works with very many points. We might not find the best orientation if there is noise
        # PCA is not exactly the same as finding the most faraway points.

        if not dummy_transform:
            try:
                self.pca = decomposition.PCA(
                    n_components=1
                )  # , svd_solver='randomized')
                # this takes a lot of time because sklearn validates the arrays
                projection = self.pca.fit_transform(points)
            except ValueError:
                return
        else:

            class DummyPCA:
                def transform(self, data):
                    return data[:, 0].reshape(-1, 1)

            self.pca = DummyPCA()
            projection = self.pca.transform(points)

        assert projection.shape == (len(points), 1)
        self.hyperplane = np.median(projection)

    def hyperplane_quality(self, points, r):
        """Returns the share of points that further than `r` away from the hyperplane"""
        try:
            left, right = self.get_partitions(points, r)

            count_partitioned_points = len(left) + len(right)
            return count_partitioned_points / len(points)
        except (ValueError, AttributeError):
            return -1

    def is_query_in_one_partition(self, queries, r):
        """Returns the share of range queries that only retrive elements from one partition.

        I.e. the partitioning can be used to speed up the query.
        `queries` should have the shape [number of queries, dimensionality]."""

        try:
            projection = _faster_transform(self.pca, queries)
            distance_to_border = np.abs(projection - self.hyperplane)
            n_far_away = (distance_to_border > r).sum()
            return n_far_away / len(queries)
        except (ValueError, AttributeError):
            return -1

    def get_partitions(self, points, r):
        """Return point indices that are in the (left, right) partition and further
        away from the boundary than `r`."""
        projection = _faster_transform(self.pca, points).flatten()
        assert len(projection) == len(points)
        mid = self.hyperplane
        left = np.argwhere(projection < mid - r).flatten()
        right = np.argwhere(projection > mid + r).flatten()

        intersec = set(left).intersection(right)
        assert len(intersec) == 0, (
            f"a point can only be on one side, but these are on both: {intersec}"
        )

        return left, right
