import numpy as np
import numpy.testing as npt
from scipy import spatial
import pytest
from sympy.geometry import point

from .tetrahedron import (
    euclidean2d_upper_bound_dist,
    project_to_2d_euclidean,
    upper_bound_dist_matrix,
)

# absolute tolerance
ATOL = 0.01


@pytest.fixture
def tri_h():
    # coordinates of two 2D test triangles
    return dict(
        p0=(0, 0),
        p1=(5, 0),
        o=(-3, 4),
        q=(4, 2),
        d_qo=7.28,
    )


@pytest.fixture
def points_10d():
    rng = np.random.default_rng()

    n_samples = 100
    dim = 10
    points = rng.integers(-37, 38, size=[n_samples, dim])
    return points


def test_projection_of_2d(tri_h):
    d = spatial.minkowski_distance
    points_p = project_to_2d_euclidean(
        points=np.array(
            [
                tri_h["o"],
                tri_h["q"],
            ]
        ),
        p0=tri_h["p0"],
        p1=tri_h["p1"],
        dist_func=d,
    )
    # the projection should do anything to two 2d points
    dist = d(*points_p)
    assert np.allclose(dist, tri_h["d_qo"], atol=ATOL)

    assert np.allclose(points_p[0], tri_h["o"], atol=ATOL)
    assert np.allclose(points_p[1], tri_h["q"], atol=ATOL)


def test_projection_of_10d(points_10d):
    dists = spatial.distance_matrix(points_10d, points_10d)

    points_p = project_to_2d_euclidean(
        points=points_10d,
        p0=points_10d[0],
        p1=points_10d[1],
        dist_func=spatial.minkowski_distance,
    )
    lbs = spatial.distance_matrix(points_p, points_p)
    ubs = upper_bound_dist_matrix(points_p, points_p)
    assert np.all(lbs <= dists + ATOL)
    assert np.all(ubs >= dists - ATOL)