from scipy.spatial.transform import Rotation as R
import numpy as np

from .metric import Metric

def _rot_2d(radians):
    """returns a rotation matrix"""
    rot = R.from_rotvec(np.array([0,0,radians])).as_matrix()
    rot = rot[0:2,0:2]
    return rot

class PivotSpace:
    def __init__(self, metric:Metric, pivots:np.ndarray):
        self.metric = metric
        self.pivots = pivots

    def transform_single_point(self, point):
        np.asarray([self.metric(piv, point) for piv in self.pivots])

    def transform_points(self, points:np.ndarray):
        n_points = points.shape[0]
        n_dim = len(self.pivots)
        transformed = np.empty([n_points, n_dim])

        for dim, pv in enumerate(self.pivots):
            transformed[:, dim] = self.metric(pv, points)
        return transformed

    def rectify(self, points_in_ps):
        """rotates and translates points in the pivots space into a "normal" position.

        The end result is a rectangular pivot space that makes histogramming easier. 
        """ 
        assert len(self.pivots) == 2, "Not implemented for higher dimensions"
        piv_dist = self.metric(*self.pivots)
        base_length = np.sqrt(2) * piv_dist
        proj = points_in_ps - 0.5 * np.array([piv_dist, piv_dist])
        proj = proj @ _rot_2d(np.pi/4)
        proj += np.array([base_length / 2,0])
        return proj

    
