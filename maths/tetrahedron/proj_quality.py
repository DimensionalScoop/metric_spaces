"""Different measures to gauge the quality of a projection.

E.g.: quality of a hyperplane projection
"""

from typing import *
import numpy as np
from sklearn import decomposition
from sympy.series import order

from metric.metric import Metric, Euclid


def __count_neighbours(center_idx, radius, dist_matrix):
    return (dist_matrix[center_idx] < radius).sum() - 1


def candidate_set_size(points: np.ndarray, r: float, d: Metric, agg="mean") -> Any:
    """Assume every point in `points` is a proxy for
    query of range `r`. How big is the candidate set?

    Args:
        points: [number of points, dimension of point]
        r: query distance
        agg: aggregation metric:
            None: return list of candidate set sizes,
            "mean": np.mean
            callable: function that takes an np.ndarray and returns a float
    """
    dist_matrix = d.distance_matrix(points, points)
    neighbours = [__count_neighbours(i, r, dist_matrix) for i in range(len(points))]

    if agg is None:
        return neighbours
    elif agg == "mean":
        agg = np.mean
    return agg(neighbours)


def get_average_k_nn_dist(points, d: Metric, k=10, agg="mean"):
    dist_m = d.distance_matrix(points, points)
    dist_m = np.sort(dist_m, axis=1)
    k_dist = dist_m[:, k - 1]

    if agg is None:
        return k_dist
    elif agg == "mean":
        agg = np.mean
    return agg(k_dist)


class HilbertPartitioner:
    """Finds a hyperplane bisecting the dataset which tries to maximize
    the number of partitioned points.

    Rotates the space to find the best (i.e. broadest) point spread.
    Points closer to the hyperplaen than `r` are considered to not profit from the partition.

    Returns:
        Percentage of points that are not too close to the hyperplane.
    """

    def __init__(self, points: np.ndarray, r: float):
        # XXX: The PCA only works with very many points. We might not find the best orientation if there is noise
        # PCA is not exactly the same as finding the most faraway points.
        self.r = r
        self.pca = decomposition.PCA(1)
        projection = self.pca.fit_transform(points)
        assert projection.shape == (len(points), 1)
        self.hyperplane = np.median(projection)

    def hyperplane_quality(self, points):
        left, right = self.get_partitions(points)

        count_partitioned_points = len(left) + len(right)
        return count_partitioned_points / len(points)

    def get_partitions(self, points):
        """Return point indices that are in the (left, right) partition"""
        projection = self.pca.transform(points).flatten()
        assert len(projection) == len(points)
        mid = self.hyperplane
        r = self.r
        left = np.argwhere(projection < mid - r).flatten()
        right = np.argwhere(projection > mid + r).flatten()

        intersec = set(left).intersection(right)
        assert (
            len(intersec) == 0
        ), f"a point can only be on one side, but these are on both: {intersec}"

        return left, right
