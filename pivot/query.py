import numpy as np
from dataclasses import dataclass

from .transform import PivotSpace

Point = np.ndarray


@dataclass
class RangeQuery:
    center: Point
    range_: float
    transform: PivotSpace

    def get_candidates(self, points, lower_bound="exact"):
        """return all points that could lie inside the query range.

        If no lower_bound function is given, return the exact candidate set.
        """
        dist = np.zeros(points.shape[0])

        if lower_bound == "exact":
            dist = self.transform.metric(self.center, points)
        elif lower_bound == "triangle":
            dist = _triangle_lb(
                self.transform.metric,
                self.center,
                self.transform.pivots[0],
                points,
            )
        elif lower_bound == "triangles":
            dist = np.max(
                [
                    _triangle_lb(
                        self.transform.metric,
                        self.center,
                        piv,
                        points,
                    )
                    for piv in self.transform.pivots
                ],
                axis=0,
            )
        elif lower_bound == "ptolemy":
            dist = _ptolemy_lb(
                self.transform.metric,
                self.center,
                self.transform.pivots[0],
                self.transform.pivots[1],
                points,
                )
        else:
            raise NotImplementedError()

        return points[dist <= self.range_]


def _triangle_lb(d, q, piv, points):
    return np.abs(d(q, piv) - d(piv, points))

def _ptolemy_lb(d, q, p1, p2, points):
    numerator = d(points, p1) * d(q, p2) - d(points, p2) * d(q, p1)
    denominator = d(p1, p2)
    return np.abs(numerator) / denominator
