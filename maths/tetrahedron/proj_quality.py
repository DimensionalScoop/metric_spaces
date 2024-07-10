"""Different measures to gauge the quality of a projection.

E.g.: quality of a hyperplane projection
"""

from typing import *
import numpy as np

from metric.metric import Metric


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
