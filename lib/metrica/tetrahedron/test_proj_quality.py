import pytest
import numpy as np
from scipy.optimize._lsq.common import left_multiplied_operator
from .proj_quality import HilbertPartitioner


def rotation_matrix(rads):
    return np.array([[np.cos(rads), -np.sin(rads)], [np.sin(rads), np.cos(rads)]])


QUERY_RANGE = 2


@pytest.fixture()
def gauss_2d_space():
    """Generate a gaussian point distribution with a non-trivial optimal partition boundary.
    Also keep track of what should be left and what should be right to the partition plane.
    """
    rng = np.random.default_rng(0x5EED)
    points = rng.standard_normal([1000, 2])
    points[:, 1] *= 5
    left_idx = np.argwhere(points[:, 1] >= QUERY_RANGE).flatten()
    right_idx = np.argwhere(points[:, 1] <= -QUERY_RANGE).flatten()
    assert len(left_idx)
    assert len(right_idx)
    rot = rotation_matrix(np.pi / 3)
    points = points @ rot
    points[:, 0] + 10
    return points, left_idx, right_idx


def test_hilbert_partition(gauss_2d_space):
    points, control_left, control_right = gauss_2d_space
    # be a bit more conservative, because the PCA might not find the best orientation
    part = HilbertPartitioner(points, 1.1 * QUERY_RANGE)

    qual = part.hyperplane_quality(points)
    assert 0 <= qual <= 1
    print(qual)

    test_left, test_right = part.get_partitions(points)
    tl = set(test_left)
    tr = set(test_right)
    cl = set(control_left)
    cr = set(control_right)

    # we don't know if the test_left corresponds to control left or right
    if tl.intersection(cr) > tl.intersection(cl):
        cr, cl = cl, cr

    assert tl.issubset(cl)
    assert tr.issubset(cr)
